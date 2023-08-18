// Copyright 2023 PingCAP, Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <Flash/Executor/PipelineExecutorContext.h>
#include <Flash/Executor/toRU.h>
#include <Flash/Pipeline/Schedule/TaskQueues/IOPriorityQueue.h>
#include <Flash/Pipeline/Schedule/TaskQueues/MultiLevelFeedbackQueue.h>
#include <Flash/Pipeline/Schedule/TaskQueues/ResourceControlQueue.h>
#include <Flash/Pipeline/Schedule/Tasks/TaskHelper.h>

namespace DB
{
template <typename NestedTaskQueueType>
void ResourceControlQueue<NestedTaskQueueType>::submit(TaskPtr && task)
{
    {
        std::lock_guard lock(mu);
        submitWithoutLock(std::move(task));
    }
    cv.notify_one();
}

template <typename NestedTaskQueueType>
void ResourceControlQueue<NestedTaskQueueType>::submit(std::vector<TaskPtr> & tasks)
{
    {
        std::lock_guard lock(mu);
        for (auto & task : tasks)
        {
            submitWithoutLock(std::move(task));
            cv.notify_one();
        }
    }
}

template <typename NestedTaskQueueType>
void ResourceControlQueue<NestedTaskQueueType>::submitWithoutLock(TaskPtr && task)
{
    if unlikely (is_finished)
    {
        FINALIZE_TASK(task);
        return;
    }
    const auto & query_id = task->getQueryId();
    if unlikely (cancel_query_id_cache.contains(query_id))
    {
        cancel_task_queue.push_back(std::move(task));
        return;
    }

    // name can be empty, it means resource control is disabled.
    const String & name = task->getResourceGroupName();

    auto iter = resource_group_task_queues.find(name);
    if (iter == resource_group_task_queues.end())
    {
        auto task_queue = std::make_shared<NestedTaskQueueType>();
        task_queue->submit(std::move(task));
        resource_group_infos.push({name, LocalAdmissionController::global_instance->getPriority(name), task_queue});
        resource_group_task_queues.insert({name, task_queue});
    }
    else
    {
        iter->second->submit(std::move(task));
    }
}

template <typename NestedTaskQueueType>
bool ResourceControlQueue<NestedTaskQueueType>::take(TaskPtr & task)
{
    assert(task == nullptr);
    std::unique_lock lock(mu);
    while (true)
    {
        if unlikely (is_finished)
            return false;

        if (popTask(cancel_task_queue, task))
            return true;

        updateResourceGroupInfosWithoutLock();

        while (!resource_group_infos.empty())
        {
            const ResourceGroupInfo & group_info = resource_group_infos.top();
            const bool ru_exhausted = LocalAdmissionController::isRUExhausted(group_info.priority);

            LOG_TRACE(
                logger,
                "trying to schedule task of resource group {}, priority: {}, ru exhausted: {}, is_finished: {}, "
                "task_queue.empty(): {}",
                group_info.name,
                group_info.priority,
                ru_exhausted,
                is_finished,
                group_info.task_queue->empty());

            // When highest priority of resource group is less than zero, means RU of all resource groups are exhausted.
            // Should not take any task from nested task queue for this situation.
            if (ru_exhausted)
                break;

            if (group_info.task_queue->empty())
            {
                // Nested task queue is empty, continue and try next resource group.
                size_t erase_num = resource_group_task_queues.erase(group_info.name);
                RUNTIME_CHECK_MSG(
                    erase_num == 1,
                    "cannot erase corresponding TaskQueue for task of resource group {}, erase_num: {}",
                    group_info.name,
                    erase_num);
                resource_group_infos.pop();
            }
            else
            {
                // Take task from nested task queue, and should always take succeed.
                // Because this task queue should not be finished inside lock_guard.
                RUNTIME_CHECK(group_info.task_queue->take(task));
                assert(task != nullptr);
                return true;
            }
        }

        assert(!task);
        // Wakeup when:
        // 1. finish() is called.
        // 2. refill_token_callback is called by LAC.
        cv.wait(lock);
    }
}

template <typename NestedTaskQueueType>
void ResourceControlQueue<NestedTaskQueueType>::updateStatistics(const TaskPtr & task, ExecTaskStatus, UInt64 inc_value)
{
    assert(task);
    const String & name = task->getResourceGroupName();

    std::lock_guard lock(mu);
    auto ru = toRU(inc_value);
    LOG_TRACE(logger, "resource group {} will consume {} RU(or {} cpu time in ns)", name, ru, inc_value);
    LocalAdmissionController::global_instance->consumeResource(name, ru, inc_value);
}

template <typename NestedTaskQueueType>
void ResourceControlQueue<NestedTaskQueueType>::updateResourceGroupInfosWithoutLock()
{
    std::priority_queue<ResourceGroupInfo> new_resource_group_infos;
    while (!resource_group_infos.empty())
    {
        const ResourceGroupInfo & group_info = resource_group_infos.top();
        auto new_priority = LocalAdmissionController::global_instance->getPriority(group_info.name);
        new_resource_group_infos.push({group_info.name, new_priority, group_info.task_queue});
        resource_group_infos.pop();
    }
    resource_group_infos = new_resource_group_infos;
}

template <typename NestedTaskQueueType>
bool ResourceControlQueue<NestedTaskQueueType>::empty() const
{
    std::lock_guard lock(mu);

    if (!cancel_task_queue.empty())
        return false;

    if (resource_group_task_queues.empty())
        return true;

    for (const auto & task_queue_iter : resource_group_task_queues)
    {
        if (!task_queue_iter.second->empty())
            return false;
    }
    return true;
}

template <typename NestedTaskQueueType>
void ResourceControlQueue<NestedTaskQueueType>::finish()
{
    {
        std::lock_guard lock(mu);
        is_finished = true;
        for (auto & ele : resource_group_task_queues)
            ele.second->finish();
    }

    cv.notify_all();
}

template <typename NestedTaskQueueType>
void ResourceControlQueue<NestedTaskQueueType>::cancel(const String & query_id, const String & resource_group_name)
{
    if unlikely (query_id.empty())
        return;

    std::lock_guard lock(mu);
    if (cancel_query_id_cache.add(query_id))
    {
        auto iter = resource_group_task_queues.find(resource_group_name);
        if (iter != resource_group_task_queues.end())
        {
            iter->second->collectCancelledTasks(cancel_task_queue, query_id);
        }
    }
}

template class ResourceControlQueue<CPUMultiLevelFeedbackQueue>;
// For now, io_task_thread_pool is not managed by ResourceControl mechanism.
template class ResourceControlQueue<IOPriorityQueue>;
} // namespace DB