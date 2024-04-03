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

#pragma once

#include <Columns/ColumnFixedString.h>
#include <Columns/ColumnNullable.h>
#include <Columns/ColumnString.h>
#include <Common/Arena.h>
#include <Common/Logger.h>
#include <Core/Spiller.h>
#include <DataStreams/IBlockInputStream.h>
#include <DataStreams/RuntimeFilter.h>
#include <Flash/Coprocessor/JoinInterpreterHelper.h>
#include <Flash/Coprocessor/RuntimeFilterMgr.h>
#include <Interpreters/AggregationCommon.h>
#include <Interpreters/ExpressionActions.h>
#include <Interpreters/HashJoinSpillContext.h>
#include <Interpreters/JoinHashMap.h>
#include <Interpreters/JoinPartition.h>
#include <Interpreters/ProbeProcessInfo.h>
#include <Interpreters/SettingsCommon.h>

#include <memory>
#include <shared_mutex>

namespace DB
{
struct JoinProfileInfo
{
    UInt64 peak_build_bytes_usage = 0;
    bool is_spill_enabled = false;
    bool is_spilled = false;
};
using JoinProfileInfoPtr = std::shared_ptr<JoinProfileInfo>;

class Join;
using JoinPtr = std::shared_ptr<Join>;

class AutoSpillTrigger;

struct RestoreInfo
{
    JoinPtr join;
    size_t stream_index;
    BlockInputStreamPtr scan_hash_map_stream;
    BlockInputStreamPtr build_stream;
    BlockInputStreamPtr probe_stream;

    RestoreInfo(
        const JoinPtr & join_,
        size_t stream_index_,
        BlockInputStreamPtr && scan_hash_map_stream_,
        BlockInputStreamPtr && build_stream_,
        BlockInputStreamPtr && probe_stream_)
        : join(join_)
        , stream_index(stream_index_)
        , scan_hash_map_stream(std::move(scan_hash_map_stream_))
        , build_stream(std::move(build_stream_))
        , probe_stream(std::move(probe_stream_))
    {}
};

struct PartitionBlock
{
    size_t partition_index;
    Block block;

    PartitionBlock()
        : partition_index(0)
        , block({})
    {}

    explicit PartitionBlock(Block && block_)
        : partition_index(0)
        , block(std::move(block_))
    {}

    PartitionBlock(size_t partition_index_, Block && block_)
        : partition_index(partition_index_)
        , block(std::move(block_))
    {}

    explicit operator bool() const { return static_cast<bool>(block); }
    bool operator!() const { return !block; }
};
using PartitionBlocks = std::list<PartitionBlock>;

struct RestoreConfig
{
    Int64 join_restore_concurrency;
    size_t restore_round;
    size_t restore_partition_id;
};

enum class SimpleColumnType
{
    Fixed,
    String,
    Other,
};

class SimpleColumnView
{
public:
    ConstNullMapPtr null_map;
    size_t fixed_size;
    StringRef fixed_data;
    void * other_data;
};

struct SimpleMutableColumn
{
    SimpleColumnType type;
    ColumnUInt8 * null_map;
    std::unique_ptr<SimplePaddedPODArray> pod_array;
    void * other_data;
};

const size_t PARTITION_SHIFT = 4;
const size_t PARTITION_COUNT = 1 << PARTITION_SHIFT;

inline size_t getPartitionNum(size_t hash)
{
    const size_t partition_mask = (PARTITION_COUNT - 1) << (32 - PARTITION_SHIFT);
    return (hash & partition_mask) >> (32 - PARTITION_SHIFT);
}

using RowPtr = char *;
using RowPtrs = PaddedPODArray<RowPtr>;
using MultipleRowPtrs = std::vector<RowPtrs>;

// TODO: change to UInt32
constexpr size_t pointer_offset = sizeof(size_t);
constexpr size_t key_offset = sizeof(size_t) + sizeof(RowPtr);

constexpr size_t row_align = alignof(std::atomic<RowPtr>);

ALWAYS_INLINE inline bool getRowPtrFlag(RowPtr ptr)
{
    static_assert(row_align <= UINT8_MAX + 1);
    auto * next = reinterpret_cast<std::atomic<RowPtr> *>(ptr + pointer_offset)->load(std::memory_order_relaxed);
    auto flag = static_cast<UInt8>(reinterpret_cast<uintptr_t>(next) & static_cast<uintptr_t>(row_align - 1));
    return flag & 0x01;
}

ALWAYS_INLINE inline void setRowPtrFlag(RowPtr ptr)
{
    std::atomic<RowPtr> * atomic_next;
    RowPtr next, new_next;
    do
    {
        atomic_next = reinterpret_cast<std::atomic<RowPtr> *>(ptr + pointer_offset);
        next = atomic_next->load(std::memory_order_relaxed);
        new_next = reinterpret_cast<RowPtr>(reinterpret_cast<uintptr_t>(next) | 0x01);
        if (new_next == next)
            break;
    } while (!std::atomic_compare_exchange_weak(atomic_next, &next, new_next));
}

void convertToSimpleMutableColumn(MutableColumnPtr & ptr, SimpleMutableColumn & simple_column);
void deserializeAndInsertFromPos(SimpleMutableColumn & simple_column, RowPtrs & pos);
void convertToMutableColumn(SimpleMutableColumn & simple_column, MutableColumnPtr & ptr);

struct alignas(ABSL_CACHELINE_SIZE) MultipleRowPtrsWithLock
{
    std::mutex mu;
    std::vector<RowPtrs> multi_row_ptrs;
    size_t partitioned_row_counts = 0;
    size_t build_table_index = 0;
    size_t scan_table_index = 0;

    void insert(RowPtrs && row_ptrs, size_t count)
    {
        std::unique_lock lock(mu);
        multi_row_ptrs.push_back(std::move(row_ptrs));
        partitioned_row_counts += count;
    }

    RowPtrs * getNext()
    {
        std::unique_lock lock(mu);
        if (build_table_index >= multi_row_ptrs.size())
            return nullptr;
        return &multi_row_ptrs[build_table_index++];
    }

    RowPtrs * getScanNext()
    {
        std::unique_lock lock(mu);
        if (scan_table_index >= multi_row_ptrs.size())
            return nullptr;
        return &multi_row_ptrs[scan_table_index++];
    }
};

class alignas(ABSL_CACHELINE_SIZE) BuildWorkerData
{
public:
    BuildWorkerData() = default;
    ~BuildWorkerData()
    {
        for (auto & c : row_memory)
            alloc.free(c.first, c.second);
        row_memory.clear();
    }

    Allocator<false> alloc;
    std::vector<std::pair<RowPtr, size_t>> row_memory;
    size_t row_count = 0;

    RowPtr null_rows_list_head = nullptr;

    Arena pool;
    PaddedPODArray<size_t> row_sizes;
    PaddedPODArray<size_t> hashes;
    RowPtrs row_ptrs;

    size_t build_time = 0;
    size_t convert_time = 0;
    size_t column_num = 0;

    size_t build_pointer_table_time = 0;
    size_t build_pointer_table_size = 0;

    ssize_t iter_index = -1;
};

class alignas(ABSL_CACHELINE_SIZE) ProbeWorkerData
{
public:
    size_t probe_time = 0;
    size_t row_count = 0;

    Arena pool;
    std::vector<SimpleMutableColumn> simple_added_columns;
    std::vector<std::string> sort_key_containers;

    size_t prefetch_iter = 0;
    std::unique_ptr<void, std::function<void(void *)>> prefetch_states;

    RowPtrs row_ptrs_buffer;
};

class JoinHashPointerTable
{
public:
    JoinHashPointerTable() = default;
    ~JoinHashPointerTable()
    {
        if (pointer_table != nullptr)
            alloc.free(reinterpret_cast<void *>(pointer_table), pointer_table_size * sizeof(std::atomic<RowPtr>));
    }

    size_t pointer_table_size = 0;
    size_t pointer_table_size_degree = 0;
    size_t pointer_table_size_mask = 0;
    std::atomic<RowPtr> * pointer_table = nullptr;
    Allocator<true> alloc;

    static size_t pointerTableCapacity(size_t count) { return std::max(roundUpToPowerOfTwoOrZero(count * 2), 1 << 10); }

    void init(size_t row_count)
    {
        pointer_table_size = pointerTableCapacity(row_count);
        if (pointer_table_size > (1ULL << 32))
        {
            pointer_table_size = 1ULL << 32;
        }
        RUNTIME_ASSERT(isPowerOfTwo(pointer_table_size));
        pointer_table_size_degree = log2(pointer_table_size);
        RUNTIME_ASSERT(1ULL << pointer_table_size_degree == pointer_table_size);
        RUNTIME_ASSERT(pointer_table_size_degree <= 32);

        pointer_table_size_mask = (pointer_table_size - 1) << (32 - pointer_table_size_degree);

        pointer_table = reinterpret_cast<std::atomic<RowPtr> *>(
            alloc.alloc(pointer_table_size * sizeof(std::atomic<RowPtr>), sizeof(std::atomic<RowPtr>)));
    }

    inline size_t getBucketNum(size_t hash) const
    {
        return (hash & pointer_table_size_mask) >> (32 - pointer_table_size_degree);
    }
};

/** Data structure for implementation of JOIN.
  * It is just a hash table: keys -> rows of joined ("right") table.
  * Additionally, CROSS JOIN is supported: instead of hash table, it use just set of blocks without keys.
  *
  * JOIN-s could be of nine types: ANY/ALL × LEFT/INNER/RIGHT/FULL, and also CROSS.
  *
  * If ANY is specified - then select only one row from the "right" table, (first encountered row), even if there was more matching rows.
  * If ALL is specified - usual JOIN, when rows are multiplied by number of matching rows from the "right" table.
  * ANY is more efficient.
  *
  * If INNER is specified - leave only rows that have matching rows from "right" table.
  * If LEFT is specified - in case when there is no matching row in "right" table, fill it with default values instead.
  * If RIGHT is specified - first process as INNER, but track what rows from the right table was joined,
  *  and at the end, add rows from right table that was not joined and substitute default values for columns of left table.
  * If FULL is specified - first process as LEFT, but track what rows from the right table was joined,
  *  and at the end, add rows from right table that was not joined and substitute default values for columns of left table.
  *
  * Thus, LEFT and RIGHT JOINs are not symmetric in terms of implementation.
  *
  * All JOINs (except CROSS) are done by equality condition on keys (equijoin).
  * Non-equality and other conditions are not supported.
  *
  * Implementation:
  *
  * 1. Build hash table in memory from "right" table.
  * This hash table is in form of keys -> row in case of ANY or keys -> [rows...] in case of ALL.
  * This is done in insertFromBlock method.
  *
  * 2. Process "left" table and join corresponding rows from "right" table by lookups in the map.
  * This is done in joinBlock methods.
  *
  * In case of ANY LEFT JOIN - form new columns with found values or default values.
  * This is the most simple. Number of rows in left table does not change.
  *
  * In case of ANY INNER JOIN - form new columns with found values,
  *  and also build a filter - in what rows nothing was found.
  * Then filter columns of "left" table.
  *
  * In case of ALL ... JOIN - form new columns with all found rows,
  *  and also fill 'offsets' array, describing how many times we need to replicate values of "left" table.
  * Then replicate columns of "left" table.
  *
  * How Nullable keys are processed:
  *
  * NULLs never join to anything, even to each other.
  * During building of map, we just skip keys with NULL value of any component.
  * During joining, we simply treat rows with any NULLs in key as non joined.
  *
  * Default values for outer joins (LEFT, RIGHT, FULL):
  *
  * Always generate Nullable column and substitute NULLs for non-joined rows,
  *  as in standard SQL.
  */

class Join
{
public:
    Join(
        const Names & key_names_left_,
        const Names & key_names_right_,
        ASTTableJoin::Kind kind_,
        const String & req_id,
        size_t fine_grained_shuffle_count_,
        size_t max_bytes_before_external_join_,
        const SpillConfig & build_spill_config_,
        const SpillConfig & probe_spill_config_,
        const RestoreConfig & restore_config_,
        const NamesAndTypes & output_columns_,
        const RegisterOperatorSpillContext & register_operator_spill_context_,
        AutoSpillTrigger * auto_spill_trigger_,
        const TiDB::TiDBCollators & collators_,
        const JoinNonEqualConditions & non_equal_conditions_,
        size_t max_block_size,
        size_t shallow_copy_cross_probe_threshold_,
        const String & match_helper_name_,
        const String & flag_mapped_entry_helper_name_,
        size_t probe_cache_column_threshold_,
        bool is_test,
        bool enable_new_hash_join_,
        size_t prefetch_threshold_,
        size_t prefetch_length_,
        size_t insert_buffer_size_,
        bool insert_enable_simd_,
        const std::vector<RuntimeFilterPtr> & runtime_filter_list_ = dummy_runtime_filter_list);

    RestoreConfig restore_config;

    /** Call `setBuildConcurrencyAndInitJoinPartition` and `setSampleBlock`.
      * You must call this method before subsequent calls to insertFromBlock.
      */
    void initBuild(const Block & sample_block, size_t build_concurrency_ = 1);

    void initProbe(const Block & sample_block, size_t probe_concurrency_ = 1);

    void insertFromBlock(const Block & block, size_t stream_index);

    /** Join data from the map (that was previously built by calls to insertFromBlock) to the block with data from "left" table.
      * Could be called from different threads in parallel.
      */
    Block joinBlock(ProbeProcessInfo & probe_process_info, size_t stream_index, bool dry_run = false) const;

    void checkTypes(const Block & block) const;

    /**
      * A stream that will scan and output rows from right table, might contain default values from left table
      * Use only after all calls to joinBlock was done.
      */
    BlockInputStreamPtr createScanHashMapAfterProbeStream(
        const Block & left_sample_block,
        size_t index,
        size_t step,
        size_t max_block_size) const;

    bool isEnableSpill() const;

    bool isRestoreJoin() const;

    bool getPartitionSpilled(size_t partition_index) const;

    bool hasPartitionToRestore();

    bool isSpilled() const { return hash_join_spill_context->isSpilled(); }

    bool enableNewHashJoin() const { return enable_new_hash_join; }

    std::optional<RestoreInfo> getOneRestoreStream(size_t max_block_size);

    void dispatchProbeBlock(Block & block, PartitionBlocks & partition_blocks_list, size_t stream_index);

    Blocks dispatchBlock(const Strings & key_columns_names, const Block & from_block);

    /// Number of keys in all built JOIN maps.
    /// This function can only be used externally because it uses `shared_lock(rwlock)`, and `shared_lock` is not reentrant.
    size_t getTotalRowCount() const;
    /// Sum size in bytes of all buffers, used for JOIN maps and for all memory pools.
    size_t getTotalByteCount();
    /// The peak build bytes usage, if spill is not enabled, the same as getTotalByteCount
    size_t getPeakBuildBytesUsage();

    void checkAndMarkPartitionSpilledIfNeeded(size_t stream_index);

    void checkAndMarkPartitionSpilledIfNeededInternal(
        JoinPartition & join_partition,
        std::unique_lock<std::mutex> & partition_lock,
        size_t partition_index,
        size_t stream_index);

    size_t getTotalBuildInputRows() const { return total_input_build_rows; }

    ASTTableJoin::Kind getKind() const { return kind; }

    const Names & getLeftJoinKeys() const { return key_names_left; }

    void setInitActiveBuildThreads()
    {
        std::unique_lock lock(build_probe_mutex);
        active_build_threads = getBuildConcurrency();
    }

    size_t getProbeConcurrency() const
    {
        std::unique_lock lock(build_probe_mutex);
        return probe_concurrency;
    }
    void setProbeConcurrency(size_t concurrency)
    {
        std::unique_lock lock(build_probe_mutex);
        probe_concurrency = concurrency;
        active_probe_threads = probe_concurrency;
    }

    void wakeUpAllWaitingThreads();

    // Return true if it is the last build thread.
    bool finishOneBuild(size_t stream_index);
    void finalizeBuild();
    void waitUntilAllBuildFinished() const;

    // Return true if it is the last probe thread.
    bool finishOneProbe(size_t stream_index);
    void finalizeProbe();
    void waitUntilAllProbeFinished() const;
    bool quickCheckProbeFinished() const;

    bool quickCheckBuildFinished() const;

    void finishOneNonJoin(size_t partition_index);

    size_t getBuildConcurrency() const
    {
        if (unlikely(build_concurrency == 0))
            throw Exception(
                "Logical error: `setBuildConcurrencyAndInitPool` has not been called",
                ErrorCodes::LOGICAL_ERROR);
        return build_concurrency;
    }

    void meetError(const String & error_message);
    void meetErrorImpl(const String & error_message, std::unique_lock<std::mutex> & lock);

    // std::unordered_map<partition_index, Blocks>
    using MarkedSpillData = std::unordered_map<size_t, Blocks>;

    MarkedSpillData & getBuildSideMarkedSpillData(size_t stream_index);
    const MarkedSpillData & getBuildSideMarkedSpillData(size_t stream_index) const;
    bool hasBuildSideMarkedSpillData(size_t stream_index) const;
    void flushBuildSideMarkedSpillData(size_t stream_index);

    MarkedSpillData & getProbeSideMarkedSpillData(size_t stream_index);
    const MarkedSpillData & getProbeSideMarkedSpillData(size_t stream_index) const;
    bool hasProbeSideMarkedSpillData(size_t stream_index) const;
    void flushProbeSideMarkedSpillData(size_t stream_index);
    size_t getProbeCacheColumnThreshold() const { return probe_cache_column_threshold; }

    static const String match_helper_prefix;
    static const DataTypePtr match_helper_type;
    static const String flag_mapped_entry_helper_prefix;
    static const DataTypePtr flag_mapped_entry_helper_type;

    // only use for left outer semi joins.
    const String match_helper_name;
    // only use for right semi, right anti joins with other conditions,
    // used to name the column that records matched map entry before other conditions filter
    const String flag_mapped_entry_helper_name;

    const JoinProfileInfoPtr profile_info = std::make_shared<JoinProfileInfo>();
    HashJoinSpillContextPtr hash_join_spill_context;
    const Block & getOutputBlock() const { return finalized ? output_block_after_finalize : output_block; }
    const Names & getRequiredColumns() const { return required_columns; }
    void finalize(const Names & parent_require);

    bool buildPointerTable(size_t stream_index);

private:
    friend class ScanHashMapAfterProbeBlockInputStream;

    ASTTableJoin::Kind kind;
    ASTTableJoin::Strictness strictness;
    bool has_other_condition;
    String join_req_id;
    const bool may_probe_side_expanded_after_join;

    /// Names of key columns (columns for equi-JOIN) in "left" table (in the order they appear in USING clause).
    const Names key_names_left;
    /// Names of key columns (columns for equi-JOIN) in "right" table (in the order they appear in USING clause).
    const Names key_names_right;

    mutable std::mutex build_probe_mutex;

    mutable std::condition_variable build_cv;
    size_t build_concurrency;
    size_t active_build_threads;
    std::atomic_bool build_finished{false};

    mutable std::condition_variable probe_cv;
    size_t probe_concurrency;
    size_t active_probe_threads;
    std::atomic_bool probe_finished{false};

    bool skip_wait = false;
    bool meet_error = false;
    String error_message;

    /// collators for the join key
    const TiDB::TiDBCollators collators;

    const JoinNonEqualConditions non_equal_conditions;

    size_t max_block_size;
    /// Runtime Filter, optional
    std::vector<RuntimeFilterPtr> runtime_filter_list;

    /** Blocks of "right" table.
      */
    BlocksList blocks;
    Blocks original_blocks;
    /// mutex to protect concurrent insert to blocks
    std::mutex blocks_lock;

    JoinPartitions partitions;

    std::list<size_t> remaining_partition_indexes_to_restore;

    std::atomic<size_t> peak_build_bytes_usage{0};

    std::vector<RestoreInfo> restore_infos;
    Int64 restore_join_build_concurrency = -1;

    JoinPtr restore_join;

    RegisterOperatorSpillContext register_operator_spill_context;

    AutoSpillTrigger * auto_spill_trigger;

    /// Whether to directly check all blocks for row with null key.
    bool null_key_check_all_blocks_directly = false;

    /// For null-aware semi join with no other condition.
    /// Indicate if the right table is empty.
    std::atomic<bool> right_table_is_empty{true};
    /// Indicate if the right table has a all-key-null row.
    std::atomic<bool> right_has_all_key_null_row{false};

    bool has_build_data_in_memory = false;

    CrossProbeMode cross_probe_mode = CrossProbeMode::DEEP_COPY_RIGHT_BLOCK;
    size_t right_rows_to_be_added_when_matched_for_cross_join = 0;
    size_t shallow_copy_cross_probe_threshold;
    size_t probe_cache_column_threshold;

    JoinMapMethod join_map_method = JoinMapMethod::EMPTY;

    Sizes key_sizes;

    /// Block with columns from the right-side table except key columns.
    Block sample_block_without_keys;
    /// Block with key columns in the same order they appear in the right-side table.
    Block sample_block_only_keys;

    NamesAndTypes output_columns;
    Block output_block;
    NamesAndTypes output_columns_after_finalize;
    Block output_block_after_finalize;
    NameSet output_column_names_set_after_finalize;
    NameSet output_columns_names_set_for_other_condition_after_finalize;
    Names required_columns;
    bool finalized = false;

    bool is_test;

    Block build_sample_block;
    Block probe_sample_block;

    const LoggerPtr log;

    std::atomic<size_t> total_input_build_rows{0};

    /** Protect state for concurrent use in insertFromBlock and joinBlock.
      * Note that these methods could be called simultaneously only while use of StorageJoin,
      *  and StorageJoin only calls these two methods.
      * That's why another methods are not guarded.
      */
    mutable std::shared_mutex rwlock;

    bool initialized = false;
    bool enable_fine_grained_shuffle = false;
    size_t fine_grained_shuffle_count = 0;

    // the index of vector is the stream_index.
    std::vector<MarkedSpillData> build_side_marked_spilled_data;
    std::vector<MarkedSpillData> probe_side_marked_spilled_data;

    const bool enable_new_hash_join;
    const size_t prefetch_threshold;
    const size_t prefetch_length;
    bool enable_prefetch = false;
    size_t insert_buffer_size;
    bool insert_enable_simd;

    std::vector<std::unique_ptr<MultipleRowPtrsWithLock>> partitioned_multiple_row_ptrs;
    std::vector<std::unique_ptr<BuildWorkerData>> build_workers_data;

    std::vector<std::unique_ptr<ProbeWorkerData>> probe_workers_data;

    JoinHashPointerTable table;

    mutable std::mutex build_scan_table_lock;
    size_t build_table_index = 0;
    mutable size_t scan_table_index = 0;

private:
    /** Set information about structure of right hand of JOIN (joined data).
      * You must call this method before subsequent calls to insertFromBlock.
      */
    void setSampleBlock(const Block & block);

    /** Set Join build concurrency and init hash map.
      * You must call this method before subsequent calls to insertFromBlock.
      */
    void setBuildConcurrencyAndInitJoinPartition(size_t build_concurrency_);

    /// Throw an exception if blocks have different types of key columns.
    void checkTypesOfKeys(const Block & block_left, const Block & block_right) const;

    /** Add block of data from right hand of JOIN to the map.
      * Returns false, if some limit was exceeded and you should not insert more data.
      */
    void insertFromBlockInternal(Block * stored_block, size_t stream_index);

    Block joinBlockHash(ProbeProcessInfo & probe_process_info, size_t stream_index) const;

    Block doJoinBlockHash(ProbeProcessInfo & probe_process_info, const JoinBuildInfo & join_build_info) const;
    Block doJoinBlockHashNew(
        ProbeProcessInfo & probe_process_info,
        const JoinBuildInfo & join_build_info,
        size_t stream_index) const;

    Block joinBlockNullAwareSemi(ProbeProcessInfo & probe_process_info) const;

    Block joinBlockSemi(ProbeProcessInfo & probe_process_info) const;

    Block joinBlockCross(ProbeProcessInfo & probe_process_info) const;

    Block removeUselessColumn(Block & block) const;

    /** Handle non-equal join conditions
      *
      * @param block
      */
    void handleOtherConditions(Block & block, IColumn::Filter * filter, IColumn::Offsets * offsets_to_replicate) const;

    void handleOtherConditionsForOneProbeRow(Block & block, ProbeProcessInfo & probe_process_info) const;

    Block doJoinBlockCross(ProbeProcessInfo & probe_process_info) const;

    template <ASTTableJoin::Kind KIND, ASTTableJoin::Strictness STRICTNESS, typename Maps>
    Block joinBlockNullAwareSemiImpl(const ProbeProcessInfo & probe_process_info) const;

    template <ASTTableJoin::Kind KIND, ASTTableJoin::Strictness STRICTNESS, typename Maps>
    Block joinBlockSemiImpl(const JoinBuildInfo & join_build_info, const ProbeProcessInfo & probe_process_info) const;

    IColumn::Selector hashToSelector(const WeakHash32 & hash) const;
    IColumn::Selector selectDispatchBlock(const Strings & key_columns_names, const Block & from_block);

    void spillBuildSideBlocks(UInt64 part_id, Blocks && blocks) const;
    void spillProbeSideBlocks(UInt64 part_id, Blocks && blocks) const;

    void markBuildSideSpillData(UInt64 part_id, Blocks && blocks, size_t stream_index);
    void markProbeSideSpillData(UInt64 part_id, Blocks && blocks, size_t stream_index);

    /// use lock as the argument to force the caller acquire the lock before call them
    void releaseAllPartitions();

    void spillMostMemoryUsedPartitionIfNeed(size_t stream_index);
    std::shared_ptr<Join> createRestoreJoin(size_t max_bytes_before_external_join_, size_t restore_partition_id);

    void workAfterBuildFinish(size_t stream_index);
    void workAfterProbeFinish(size_t stream_index);

    void generateRuntimeFilterValues(const Block & block);
    void finalizeRuntimeFilter();
    void cancelRuntimeFilter(const String & reason);

    void finalizeProfileInfo();

    void finalizeNullAwareSemiFamilyBuild();

    void finalizeCrossJoinBuild();

    /// Sum size in bytes of all hash table and pools
    size_t getTotalHashTableAndPoolByteCount();
};

} // namespace DB
