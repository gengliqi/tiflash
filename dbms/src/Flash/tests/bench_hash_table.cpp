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

#include <Common/HashTable/HashMap.h>
#include <Flash/tests/bench_hash_table.h>
#include <fmt/core.h>

namespace DB
{
namespace tests
{

void BenchHashTable::SetUp(const benchmark::State &)
{

}

void BenchHashTable::TearDown(const benchmark::State &)
{

}

template<size_t payload>
struct Value
{
    char p[payload]{};
};

template<size_t payload>
struct KeyValue
{
    explicit KeyValue(uint64_t k):key(k){}
    uint64_t key;
    Value<payload> value;
};

template<size_t build_payload, size_t probe_payload>
std::tuple<std::vector<KeyValue<build_payload>>, std::vector<KeyValue<probe_payload>>> init(size_t build_size, size_t probe_size, size_t match_possibility)
{
    std::vector<KeyValue<build_payload>> build_kv;
    std::vector<KeyValue<probe_payload>> probe_kv;
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<uint32_t> int_dist;

    std::random_device rd2;
    std::mt19937 mt2(rd2());
    std::uniform_int_distribution<uint32_t> int_dist2;

    build_kv.reserve(build_size);
    for (size_t i = 0; i < build_size; ++i)
        build_kv.emplace_back(int_dist(mt));

    probe_kv.reserve(probe_size);
    for (size_t i = 0; i < probe_size; ++i)
    {
        size_t is_match = int_dist2(mt2) % 100 < match_possibility;
        if (is_match)
        {
            probe_kv.emplace_back(build_kv[int_dist(mt) % build_size].key);
        }
        else
        {
            probe_kv.emplace_back(int_dist(mt) + UINT32_MAX + 1);
        }
    }

    return {build_kv, probe_kv};
}

struct Cell
{
    uint64_t pos = 0;
    Cell * next = nullptr;
};

using CKHashTable = HashMap<uint64_t, Cell, HashCRC32<UInt32>>;

BENCHMARK_DEFINE_F(BenchHashTable, NoPartitionLinear)
(benchmark::State & state)
try
{
    const size_t build_size = state.range(0);
    const size_t probe_size = state.range(1);
    const size_t match_possibility = state.range(2);

    std::string head = std::to_string(build_size) + "/" + std::to_string(probe_size) + "/" + std::to_string(match_possibility);

    const size_t build_payload = 8;
    const size_t probe_payload = 8;

    auto [build_kv, probe_kv] = init<build_payload, probe_payload>(build_size, probe_size, match_possibility);

    Arena arena;
    CKHashTable hash_table;
    using MappedType = CKHashTable::mapped_type;

    Stopwatch watch;

    for (size_t i = 0; i < build_size; ++i)
    {
        CKHashTable::LookupResult it;
        bool inserted;
        hash_table.emplace(build_kv[i].key, it, inserted);
        if (inserted)
            new (&it->getMapped()) MappedType(Cell{i, nullptr});
        else
        {
            auto * elem = reinterpret_cast<MappedType *>(arena.alloc(sizeof(MappedType)));
            elem->next = it->getMapped().next;
            it->getMapped().next = elem;
            elem->pos = i;
        }
    }

    auto build_hash_time = watch.elapsedFromLastTime();

    printf("%s build hash table time %llu\n", head.c_str(), build_hash_time);

    std::vector<size_t> output_build;
    std::vector<size_t> probe_offset;
    probe_offset.reserve(probe_size);
    size_t offset = 0;
    for (size_t i = 0; i < probe_size; ++i)
    {
        auto * it = hash_table.find(probe_kv[i].key);
        if (it != hash_table.end())
        {
            for (auto * current = &it->getMapped(); current != nullptr; current = current->next)
            {
                output_build.push_back(current->pos);
                ++offset;
            }
        }
        probe_offset.push_back(offset);
    }

    auto probe_hash_time = watch.elapsedFromLastTime();

    printf("%s probe hash table time %llu\n", head.c_str(), probe_hash_time);
}
CATCH

BENCHMARK_DEFINE_F(BenchHashTable, PartitionLinear)
(benchmark::State & state)
try
{
    const size_t build_size = state.range(0);
    const size_t probe_size = state.range(1);
    const size_t match_possibility = state.range(2);

    const size_t build_payload = 8;
    const size_t probe_payload = 8;

    auto [build_kv, probe_kv] = init<build_payload, probe_payload>(build_size, probe_size, match_possibility);
}
CATCH


BENCHMARK_REGISTER_F(BenchHashTable, NoPartitionLinear)
    ->Args({100000, 1000000, 50})
    ->Args({100000, 1000000, 100})
    ->Args({1000000, 10000000, 50})
    ->Args({1000000, 10000000, 100})
    ->Args({10000000, 100000000, 50})
    ->Args({10000000, 100000000, 100})->Iterations(2);



} // namespace tests
} // namespace DB