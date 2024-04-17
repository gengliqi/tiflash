// Copyright 2023 PingCAP, Inc.
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

#include <Operators/AggregateContext.h>
#include <Operators/AggregateConvergentSourceOp.h>

namespace DB
{
AggregateConvergentSourceOp::AggregateConvergentSourceOp(
    PipelineExecutorContext & exec_context_,
    const AggregateContextPtr & agg_context_,
    size_t index_,
    const String & req_id)
    : SourceOp(exec_context_, req_id)
    , agg_context(agg_context_)
    , index(index_)
{
    setHeader(agg_context->getHeader());
}

OperatorStatus AggregateConvergentSourceOp::readImpl(Block & block)
{
    Stopwatch watch;
    agg_context->convertPendingDataToTwoLevel();
    if (!agg_context->isAllConvertFinished())
        return OperatorStatus::WAITING;

    block = agg_context->readForConvergent(index);
    convergence_time += watch.elapsedFromLastTime();
    total_rows += block.rows();
    return OperatorStatus::HAS_OUTPUT;
}

OperatorStatus AggregateConvergentSourceOp::awaitImpl()
{
    return agg_context->isAllConvertFinished() ? OperatorStatus::HAS_OUTPUT : OperatorStatus::WAITING;
}

void AggregateConvergentSourceOp::operateSuffixImpl()
{
    LOG_INFO(log, "finish read {} rows from aggregate context, cost {}ns", total_rows, convergence_time);
}

} // namespace DB
