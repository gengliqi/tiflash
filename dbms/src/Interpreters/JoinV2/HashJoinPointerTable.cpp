// Copyright 2024 PingCAP, Inc.
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

#include <Common/Stopwatch.h>
#include <Interpreters/JoinV2/HashJoinPointerTable.h>

namespace DB
{

void HashJoinPointerTable::init(size_t row_count, size_t probe_prefetch_threshold [[maybe_unused]])
{
    pointer_table_size = pointerTableCapacity(row_count);
    if (pointer_table_size > (1ULL << 32))
        pointer_table_size = 1ULL << 32;

    RUNTIME_ASSERT(isPowerOfTwo(pointer_table_size));
    pointer_table_size_degree = log2(pointer_table_size);
    RUNTIME_ASSERT(1ULL << pointer_table_size_degree == pointer_table_size);
    RUNTIME_ASSERT(pointer_table_size_degree <= 32);

    //enable_probe_prefetch = pointer_table_size >= probe_prefetch_threshold;
    enable_probe_prefetch = false;

    pointer_table_size_mask = (pointer_table_size - 1) << (32 - pointer_table_size_degree);

    pointer_table = reinterpret_cast<std::atomic<RowPtr> *>(
        alloc.alloc(pointer_table_size * sizeof(std::atomic<RowPtr>), sizeof(std::atomic<RowPtr>)));
}

template <typename HashValueType>
bool HashJoinPointerTable::build(
    const HashJoinRowLayout & row_layout,
    JoinBuildWorkerData & worker_data,
    std::vector<std::unique_ptr<MultipleRowContainer>> & multi_row_containers,
    size_t max_build_size)
{
    Stopwatch watch;
    size_t build_size = 0;
    bool is_end = false;
    while (true)
    {
        RowContainer * container = nullptr;
        if (worker_data.build_pointer_table_iter != -1)
            container = multi_row_containers[worker_data.build_pointer_table_iter]->getNext();
        if (container == nullptr)
        {
            {
                std::unique_lock lock(build_scan_table_lock);
                for (size_t i = 0; i < HJ_BUILD_PARTITION_COUNT; ++i)
                {
                    build_table_index = (build_table_index + i) % HJ_BUILD_PARTITION_COUNT;
                    container = multi_row_containers[build_table_index]->getNext();
                    if (container != nullptr)
                    {
                        worker_data.build_pointer_table_iter = build_table_index;
                        build_table_index = (build_table_index + 1) % HJ_BUILD_PARTITION_COUNT;
                        break;
                    }
                }
            }

            if (container == nullptr)
            {
                is_end = true;
                break;
            }
        }
        size_t size = container->size();
        build_size += size;
        for (size_t i = 0; i < size; ++i)
        {
            RowPtr row_ptr = container->getRowPtr(i);
            assert((reinterpret_cast<uintptr_t>(row_ptr) & (ROW_ALIGN - 1)) == 0);

            size_t hash = unalignedLoad<HashValueType>(row_ptr);
            size_t bucket = getBucketNum(hash);
            RowPtr head;
            do
            {
                head = pointer_table[bucket].load(std::memory_order_relaxed);
                unalignedStore<RowPtr>(row_ptr + row_layout.next_pointer_offset, head);
            } while (!std::atomic_compare_exchange_weak(&pointer_table[bucket], &head, row_ptr));
        }

        if (build_size >= max_build_size)
            break;
    }
    worker_data.build_pointer_table_size += build_size;
    worker_data.build_pointer_table_time += watch.elapsedMilliseconds();
    return !is_end;
}

template bool HashJoinPointerTable::build<UInt8>(
    const HashJoinRowLayout & row_layout,
    JoinBuildWorkerData & worker_data,
    std::vector<std::unique_ptr<MultipleRowContainer>> & multi_row_containers,
    size_t max_build_size);
template bool HashJoinPointerTable::build<UInt16>(
    const HashJoinRowLayout & row_layout,
    JoinBuildWorkerData & worker_data,
    std::vector<std::unique_ptr<MultipleRowContainer>> & multi_row_containers,
    size_t max_build_size);
template bool HashJoinPointerTable::build<UInt32>(
    const HashJoinRowLayout & row_layout,
    JoinBuildWorkerData & worker_data,
    std::vector<std::unique_ptr<MultipleRowContainer>> & multi_row_containers,
    size_t max_build_size);
template bool HashJoinPointerTable::build<size_t>(
    const HashJoinRowLayout & row_layout,
    JoinBuildWorkerData & worker_data,
    std::vector<std::unique_ptr<MultipleRowContainer>> & multi_row_containers,
    size_t max_build_size);

} // namespace DB