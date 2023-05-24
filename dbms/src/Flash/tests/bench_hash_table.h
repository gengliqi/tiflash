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

#pragma once

#include <benchmark/benchmark.h>
#include <Interpreters/Join.h>
#include <TestUtils/TiFlashTestEnv.h>
#include <TestUtils/FunctionTestUtils.h>

#include <random>

namespace DB
{
namespace tests
{

class BenchHashTable : public benchmark::Fixture
{
public:
    void SetUp(const benchmark::State &) override;
    void TearDown(const benchmark::State &) override;
};

} // namespace tests
} // namespace DB