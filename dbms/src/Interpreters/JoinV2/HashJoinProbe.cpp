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

#include <Columns/ColumnUtils.h>
#include <Common/Stopwatch.h>
#include <Interpreters/JoinUtils.h>
#include <Interpreters/JoinV2/HashJoinProbe.h>
#include <Interpreters/NullableUtils.h>

namespace DB
{

using enum ASTTableJoin::Kind;

bool JoinProbeContext::isCurrentProbeFinished() const
{
    return start_row_idx >= rows && prefetch_active_states == 0;
}

void JoinProbeContext::resetBlock(Block & block_)
{
    block = block_;
    orignal_block = block_;
    rows = block.rows();
    start_row_idx = 0;
    current_probe_row_ptr = nullptr;
    prefetch_active_states = 0;

    is_prepared = false;
    materialized_columns.clear();
    key_columns.clear();
    null_map = nullptr;
    null_map_holder = nullptr;
    current_row_is_matched = false;
}

void JoinProbeContext::prepareForHashProbe(
    HashJoinKeyMethod method,
    ASTTableJoin::Kind kind,
    const Names & key_names,
    const String & filter_column,
    const NameSet & probe_output_name_set,
    const TiDB::TiDBCollators & collators)
{
    if (is_prepared)
        return;

    key_columns = extractAndMaterializeKeyColumns(block, materialized_columns, key_names);
    /// Some useless columns maybe key columns so they must be removed after extracting key columns.
    for (size_t pos = 0; pos < block.columns();)
    {
        if (!probe_output_name_set.contains(block.getByPosition(pos).name))
            block.erase(pos);
        else
            ++pos;
    }

    /// Keys with NULL value in any column won't join to anything.
    extractNestedColumnsAndNullMap(key_columns, null_map_holder, null_map);
    /// reuse null_map to record the filtered rows, the rows contains NULL or does not
    /// match the join filter won't join to anything
    recordFilteredRows(block, filter_column, null_map_holder, null_map);

    if unlikely (!key_getter)
        key_getter = createHashJoinKeyGetter(method, collators);

    resetHashJoinKeyGetter(method, key_columns, key_getter);

    /** If you use FULL or RIGHT JOIN, then the columns from the "left" table must be materialized.
     * Because if they are constants, then in the "not joined" rows, they may have different values
     *  - default values, which can differ from the values of these constants.
     */
    if (getFullness(kind))
    {
        size_t existing_columns = block.columns();
        for (size_t i = 0; i < existing_columns; ++i)
        {
            auto & col = block.getByPosition(i).column;

            if (ColumnPtr converted = col->convertToFullColumnIfConst())
                col = converted;

            /// convert left columns (except keys) to Nullable
            if (std::end(key_names) == std::find(key_names.begin(), key_names.end(), block.getByPosition(i).name))
                convertColumnToNullable(block.getByPosition(i));
        }
    }

    is_prepared = true;
}

#define PREFETCH_READ(ptr) __builtin_prefetch((ptr), 0 /* rw==read */, 3 /* locality */)

enum class ProbePrefetchStage : UInt8
{
    None,
    FindHeader,
    FindNext,
};

template <typename KeyGetter, bool KeyTypeReference = KeyGetter::Type::isJoinKeyTypeReference()>
struct ProbePrefetchState;

template <typename KeyGetter>
struct ProbePrefetchState<KeyGetter, true>
{
    using KeyGetterType = typename KeyGetter::Type;
    using KeyType = typename KeyGetterType::KeyType;
    using HashValueType = typename KeyGetter::HashValueType;

    ProbePrefetchStage stage = ProbePrefetchStage::None;
    bool is_matched = false;
    UInt16 hash_tag = 0;
    HashValueType hash = 0;
    size_t index = 0;
    union
    {
        RowPtr ptr = nullptr;
        std::atomic<RowPtr> * pointer_ptr;
    };

    ALWAYS_INLINE KeyType getJoinKey(KeyGetterType & key_getter) { return key_getter.getJoinKeyWithBufferHint(index); }
};

template <typename KeyGetter>
struct ProbePrefetchState<KeyGetter, false>
{
    using KeyGetterType = typename KeyGetter::Type;
    using KeyType = typename KeyGetterType::KeyType;
    using HashValueType = typename KeyGetter::HashValueType;

    ProbePrefetchStage stage = ProbePrefetchStage::None;
    bool is_matched = false;
    UInt16 hash_tag = 0;
    HashValueType hash = 0;
    size_t index = 0;
    union
    {
        RowPtr ptr = nullptr;
        std::atomic<RowPtr> * pointer_ptr;
    };
    KeyType key{};

    ALWAYS_INLINE KeyType getJoinKey(KeyGetterType &) { return key; }
};

template <typename KeyGetter, bool has_null_map, AddRowType add_row_type, bool tagged_pointer>
class JoinProbeBlockHelper
{
public:
    using KeyGetterType = typename KeyGetter::Type;
    using KeyType = typename KeyGetterType::KeyType;
    using Hash = typename KeyGetter::Hash;
    using HashValueType = typename KeyGetter::HashValueType;

    JoinProbeBlockHelper(
        JoinProbeContext & context,
        JoinProbeWorkerData & wd,
        HashJoinKeyMethod method,
        ASTTableJoin::Kind kind,
        const JoinNonEqualConditions & non_equal_conditions,
        const HashJoinSettings & settings,
        const HashJoinPointerTable & pointer_table,
        const HashJoinRowLayout & row_layout,
        MutableColumns & added_columns)
        : context(context)
        , wd(wd)
        , method(method)
        , kind(kind)
        , non_equal_conditions(non_equal_conditions)
        , settings(settings)
        , pointer_table(pointer_table)
        , row_layout(row_layout)
        , added_columns(added_columns)
    {
        if constexpr (needRawKeyPtr<add_row_type>())
        {
            wd.insert_batch.clear();
            wd.insert_batch.reserve(settings.probe_insert_batch_size);
            //wd.insert_batch.reserve(context.rows);
        }
        if constexpr (needOtherColumnPtr<add_row_type>())
        {
            wd.insert_batch_other.clear();
            wd.insert_batch_other.reserve(settings.probe_insert_batch_size);
            //wd.insert_batch_other.reserve(context.rows);
        }

        if (pointer_table.enableProbePrefetch())
        {
            wd.selective_offsets.clear();
            wd.selective_offsets.reserve(context.rows);
            if (!wd.prefetch_states)
            {
                wd.prefetch_states = decltype(wd.prefetch_states)(
                    static_cast<void *>(new ProbePrefetchState<KeyGetter>[settings.probe_prefetch_step]),
                    [](void * ptr) { delete static_cast<ProbePrefetchState<KeyGetter> *>(ptr); });
            }
        }
        else
            wd.offsets_to_replicate.resize(context.rows);
    }

    void joinProbeBlockImpl();

    void NO_INLINE joinProbeBlockInner();
    void NO_INLINE joinProbeBlockInnerPrefetch();

    void NO_INLINE joinProbeBlockLeftOuter();
    void NO_INLINE joinProbeBlockLeftOuterPrefetch();

    void NO_INLINE joinProbeBlockSemi();
    void NO_INLINE joinProbeBlockSemiPrefetch();

    void NO_INLINE joinProbeBlockAnti();
    void NO_INLINE joinProbeBlockAntiPrefetch();

    template <bool has_other_condition>
    void NO_INLINE joinProbeBlockRightOuter();
    template <bool has_other_condition>
    void NO_INLINE joinProbeBlockRightOuterPrefetch();

    template <bool has_other_condition>
    void NO_INLINE joinProbeBlockRightSemi();
    template <bool has_other_condition>
    void NO_INLINE joinProbeBlockRightSemiPrefetch();

    template <bool has_other_condition>
    void NO_INLINE joinProbeBlockRightAnti();
    template <bool has_other_condition>
    void NO_INLINE joinProbeBlockRightAntiPrefetch();

private:
    bool ALWAYS_INLINE joinKeyIsEqual(
        KeyGetterType & key_getter,
        const KeyType & key1,
        const KeyType & key2,
        HashValueType hash1,
        RowPtr row_ptr) const
    {
        if constexpr (KeyGetterType::joinKeyCompareHashFirst())
        {
            auto hash2 = unalignedLoad<HashValueType>(row_ptr);
            if (hash1 != hash2)
                return false;
        }
        return key_getter.joinKeyIsEqual(key1, key2);
    }

    void ALWAYS_INLINE insertRowToBatch(RowPtr row_ptr, size_t key_size) const
    {
        if constexpr (needRawKeyPtr<add_row_type>())
        {
            wd.insert_batch.push_back(row_ptr);
            PREFETCH_READ(row_ptr + 64);
            PREFETCH_READ(row_ptr + 128);
        }
        if constexpr (needOtherColumnPtr<add_row_type>())
        {
            wd.insert_batch_other.push_back(row_ptr + key_size);
        }
        FlushBatchIfNecessary<false>();
    }

    template <bool force>
    void ALWAYS_INLINE FlushBatchIfNecessary() const
    {
        if constexpr (!force)
        {
            if likely (wd.insert_batch.size() < settings.probe_insert_batch_size)
                return;
        }
        if constexpr (add_row_type != AddRowType::NoNeeded)
        {
            size_t rows;
            if constexpr (needRawKeyPtr<add_row_type>())
                rows = wd.insert_batch.size();
            else
                rows = wd.insert_batch_other.size();

            /*if constexpr (needRawKeyPtr<add_row_type>())
            {
                for (auto [column_index, is_nullable] : row_layout.raw_required_key_column_indexes)
                {
                    IColumn * column = added_columns[column_index].get();
                    if (has_null_map && is_nullable)
                        column = &static_cast<ColumnNullable &>(*added_columns[column_index]).getNestedColumn();
                    column->deserializeAndInsertFromPos(wd.insert_batch, 0, rows);
                }
            }

            if constexpr (add_row_type == AddRowType::KeyAllRawNeeded || needOtherColumnPtr<add_row_type>())
            {
                for (auto [column_index, _] : row_layout.other_required_column_indexes)
                {
                    if constexpr (add_row_type == AddRowType::KeyAllRawNeeded)
                        added_columns[column_index]->deserializeAndInsertFromPos(wd.insert_batch, 0, rows);
                    else
                        added_columns[column_index]->deserializeAndInsertFromPos(wd.insert_batch_other, 0, rows);
                }
            }*/
            /*for (size_t i = 0; i < rows; ++i)
            {
                auto * pos = wd.insert_batch[i];

                auto d = unalignedLoad<Int64>(pos);
                pos += 8;
                static_cast<ColumnVector<Int64>*>(added_columns[0].get())->insert(d);

                d = unalignedLoad<Int64>(pos);
                pos += 8;
                static_cast<ColumnVector<Int64>*>(added_columns[1].get())->insert(d);

                auto sz = unalignedLoad<size_t>(pos);
                pos += 8;
                static_cast<ColumnString*>(added_columns[2].get())->insertDataImpl<false>(reinterpret_cast<char*>(pos), sz);
                pos += sz;

                d = unalignedLoad<Int64>(pos);
                pos += 8;
                static_cast<ColumnDecimal<Decimal64>*>(added_columns[3].get())->insert(Decimal64(d));

                auto d2 = unalignedLoad<UInt64>(pos);
                pos += 8;
                static_cast<ColumnVector<UInt64>*>(added_columns[4].get())->insert(d2);

                sz = unalignedLoad<size_t>(pos);
                pos += 8;
                static_cast<ColumnString*>(added_columns[5].get())->insertDataImpl<false>(reinterpret_cast<char*>(pos), sz);
                pos += sz;

                sz = unalignedLoad<size_t>(pos);
                pos += 8;
                static_cast<ColumnString*>(added_columns[6].get())->insertDataImpl<false>(reinterpret_cast<char*>(pos), sz);
                pos += sz;

                d = unalignedLoad<Int64>(pos);
                pos += 8;
                static_cast<ColumnVector<Int64>*>(added_columns[7].get())->insert(d);

                sz = unalignedLoad<size_t>(pos);
                pos += 8;
                static_cast<ColumnString*>(added_columns[8].get())->insertDataImpl<false>(reinterpret_cast<char*>(pos), sz);
            }*/
            size_t loop = 1;
            if constexpr (force)
            {
                if (rows != 0)
                    loop = 2;
            }
            for (size_t i = 0; i < loop; ++i)
            {
                static_cast<ColumnVector<Int64> *>(added_columns[0].get())
                    ->deserializeAndInsertFromPos(wd.insert_batch, 0, rows, wd.buffers[0]);
                static_cast<ColumnVector<Int64> *>(added_columns[1].get())
                    ->deserializeAndInsertFromPos(wd.insert_batch, 0, rows, wd.buffers[1]);
                static_cast<ColumnString *>(added_columns[2].get())
                    ->deserializeAndInsertFromPos(wd.insert_batch, 0, rows, wd.buffers[2]);
                static_cast<ColumnDecimal<Decimal64> *>(added_columns[3].get())
                    ->deserializeAndInsertFromPos(wd.insert_batch, 0, rows, wd.buffers[3]);
                static_cast<ColumnVector<UInt64> *>(added_columns[4].get())
                    ->deserializeAndInsertFromPos(wd.insert_batch, 0, rows, wd.buffers[4]);
                static_cast<ColumnString *>(added_columns[5].get())
                    ->deserializeAndInsertFromPos(wd.insert_batch, 0, rows, wd.buffers[5]);
                static_cast<ColumnString *>(added_columns[6].get())
                    ->deserializeAndInsertFromPos(wd.insert_batch, 0, rows, wd.buffers[6]);
                static_cast<ColumnVector<Int64> *>(added_columns[7].get())
                    ->deserializeAndInsertFromPos(wd.insert_batch, 0, rows, wd.buffers[7]);
                static_cast<ColumnString *>(added_columns[8].get())
                    ->deserializeAndInsertFromPos(wd.insert_batch, 0, rows, wd.buffers[8]);
                rows = 0;
            }

            //size_t step = settings.probe_insert_batch_size;
            //for (size_t i = 0; i < rows; i += step)
            //{
            //    size_t start = i;
            //    size_t end = i + step > rows ? rows : i + step;

            //    size_t next_end = end + step > rows ? rows : end + step;
            //    for (size_t j = end; j < next_end; ++j)
            //    {
            //        RowPtr ptr;
            //        if constexpr (needRawKeyPtr<add_row_type>())
            //            ptr = wd.insert_batch[j];
            //        else
            //            ptr = wd.insert_batch_other[j];
            //        __builtin_prefetch((ptr), 0 /* rw==read */, 1 /* locality */);
            //    }
            //    if constexpr (needRawKeyPtr<add_row_type>())
            //    {
            //        for (auto [column_index, is_nullable] : row_layout.raw_required_key_column_indexes)
            //        {
            //            IColumn * column = added_columns[column_index].get();
            //            if (has_null_map && is_nullable)
            //                column = &static_cast<ColumnNullable &>(*added_columns[column_index]).getNestedColumn();
            //            column->deserializeAndInsertFromPos(wd.insert_batch, start, end);
            //        }
            //    }

            //    if constexpr (add_row_type == AddRowType::KeyAllRawNeeded || needOtherColumnPtr<add_row_type>())
            //    {
            //        for (auto [column_index, _] : row_layout.other_required_column_indexes)
            //        {
            //            if constexpr (add_row_type == AddRowType::KeyAllRawNeeded)
            //                added_columns[column_index]->deserializeAndInsertFromPos(wd.insert_batch, start, end);
            //            else
            //                added_columns[column_index]->deserializeAndInsertFromPos(wd.insert_batch_other, start, end);
            //        }
            //    }
            //}

            if constexpr (needRawKeyPtr<add_row_type>())
                wd.insert_batch.clear();
            if constexpr (needOtherColumnPtr<add_row_type>())
                wd.insert_batch_other.clear();
        }
    }

    void ALWAYS_INLINE FillNullMap(size_t size) const
    {
        if constexpr (has_null_map)
        {
            for (auto [column_index, is_nullable] : row_layout.raw_required_key_column_indexes)
            {
                if (is_nullable)
                {
                    auto & null_map_vec
                        = static_cast<ColumnNullable &>(*added_columns[column_index]).getNullMapColumn().getData();
                    null_map_vec.resize_fill(null_map_vec.size() + size, 0);
                }
            }
        }
    }

private:
    JoinProbeContext & context;
    JoinProbeWorkerData & wd;
    const HashJoinKeyMethod method;
    const ASTTableJoin::Kind kind;
    const JoinNonEqualConditions & non_equal_conditions;
    const HashJoinSettings & settings;
    const HashJoinPointerTable & pointer_table;
    const HashJoinRowLayout & row_layout;
    MutableColumns & added_columns;
};

template <typename KeyGetter, bool has_null_map, AddRowType add_row_type, bool tagged_pointer>
void NO_INLINE JoinProbeBlockHelper<KeyGetter, has_null_map, add_row_type, tagged_pointer>::joinProbeBlockInner()
{
    auto & key_getter = *static_cast<KeyGetterType *>(context.key_getter.get());
    size_t current_offset = 0;
    auto & offsets_to_replicate = wd.offsets_to_replicate;
    size_t & idx = context.start_row_idx;
    RowPtr & ptr = context.current_probe_row_ptr;
    Stopwatch watch;
    for (; idx < context.rows; ++idx)
    {
        if (has_null_map && (*context.null_map)[idx])
        {
            offsets_to_replicate[idx] = current_offset;
            continue;
        }
        const auto & key = key_getter.getJoinKey(idx);
        auto hash = static_cast<HashValueType>(Hash()(key));
        UInt16 hash_tag = hash & ROW_PTR_TAG_MASK;
        if likely (ptr == nullptr)
        {
            ptr = pointer_table.getHeadPointer(hash);
            if (ptr == nullptr)
            {
                offsets_to_replicate[idx] = current_offset;
                continue;
            }
            if constexpr (tagged_pointer)
            {
                if (!containOtherTag(ptr, hash_tag))
                {
                    ptr = nullptr;
                    offsets_to_replicate[idx] = current_offset;
                    continue;
                }
                ptr = removeRowPtrTag(ptr);
            }
        }
        while (true)
        {
            const auto & key2 = key_getter.deserializeJoinKey(ptr + row_layout.key_offset);
            bool is_equal = joinKeyIsEqual(key_getter, key, key2, hash, ptr);
            if (is_equal)
            {
                ++current_offset;
                insertRowToBatch(ptr + row_layout.key_offset, key_getter.getJoinKeySize(key2));
                if unlikely (current_offset >= context.rows)
                    break;
            }
            wd.collision += !is_equal;
            ptr = row_layout.getNextRowPtr(ptr);
            if (ptr == nullptr)
                break;
            if constexpr (tagged_pointer)
            {
                if (!containOtherTag(ptr, hash_tag))
                {
                    ptr = nullptr;
                    break;
                }
                ptr = removeRowPtrTag(ptr);
            }
        }
        offsets_to_replicate[idx] = current_offset;
        if unlikely (ptr != nullptr)
        {
            ptr = row_layout.getNextRowPtr(ptr);
            if (ptr == nullptr)
                ++idx;
            else
            {
                if constexpr (tagged_pointer)
                {
                    if (!containOtherTag(ptr, hash_tag))
                    {
                        ptr = nullptr;
                        ++idx;
                    }
                    else
                        ptr = removeRowPtrTag(ptr);
                }
            }
            break;
        }
    }
    wd.probe_hash_table_time += watch.elapsedFromLastTime();
    FlushBatchIfNecessary<true>();
    wd.insert_time += watch.elapsedFromLastTime();
    FillNullMap(current_offset);
}

template <typename KeyGetter, bool has_null_map, AddRowType add_row_type, bool tagged_pointer>
void NO_INLINE
JoinProbeBlockHelper<KeyGetter, has_null_map, add_row_type, tagged_pointer>::joinProbeBlockInnerPrefetch()
{
    auto & key_getter = *static_cast<KeyGetterType *>(context.key_getter.get());
    auto & selective_offsets = wd.selective_offsets;
    auto * states = static_cast<ProbePrefetchState<KeyGetter> *>(wd.prefetch_states.get());

    size_t & idx = context.start_row_idx;
    size_t & active_states = context.prefetch_active_states;
    size_t & k = wd.prefetch_iter;
    size_t current_offset = 0;
    Stopwatch watch;
    while (idx < context.rows || active_states > 0)
    {
        k = k == settings.probe_prefetch_step ? 0 : k;
        auto * state = &states[k];
        if (state->stage == ProbePrefetchStage::FindNext)
        {
            RowPtr ptr = state->ptr;
            RowPtr next_ptr = row_layout.getNextRowPtr(ptr);
            if (next_ptr)
            {
                bool forward = true;
                if constexpr (tagged_pointer)
                {
                    if (containOtherTag(next_ptr, state->hash_tag))
                    {
                        next_ptr = removeRowPtrTag(next_ptr);
                    }
                    else
                    {
                        next_ptr = nullptr;
                        forward = false;
                    }
                }
                if (forward)
                {
                    state->ptr = next_ptr;
                    PREFETCH_READ(next_ptr + row_layout.next_pointer_offset);
                }
            }

            const auto & key = state->getJoinKey(key_getter);
            const auto & key2 = key_getter.deserializeJoinKey(ptr + row_layout.key_offset);
            bool is_equal = joinKeyIsEqual(key_getter, key, key2, state->hash, ptr);
            if (is_equal)
            {
                ++current_offset;
                selective_offsets.push_back(state->index);
                insertRowToBatch(ptr + row_layout.key_offset, key_getter.getJoinKeySize(key2));
                if unlikely (current_offset >= context.rows)
                {
                    if (!next_ptr)
                    {
                        state->stage = ProbePrefetchStage::None;
                        --active_states;
                    }
                    break;
                }
            }
            wd.collision += !is_equal;
            if (next_ptr)
            {
                ++k;
                continue;
            }

            state->stage = ProbePrefetchStage::None;
            --active_states;
        }
        else if (state->stage == ProbePrefetchStage::FindHeader)
        {
            RowPtr ptr = state->pointer_ptr->load(std::memory_order_relaxed);
            if (ptr)
            {
                bool forward = true;
                if constexpr (tagged_pointer)
                {
                    if (containOtherTag(ptr, state->hash_tag))
                        ptr = removeRowPtrTag(ptr);
                    else
                        forward = false;
                }
                if (forward)
                {
                    PREFETCH_READ(ptr + row_layout.next_pointer_offset);
                    state->ptr = ptr;
                    state->stage = ProbePrefetchStage::FindNext;
                    ++k;
                    continue;
                }
            }

            state->stage = ProbePrefetchStage::None;
            --active_states;
        }

        assert(state->stage == ProbePrefetchStage::None);

        if constexpr (has_null_map)
        {
            while (idx < context.rows)
            {
                if (!(*context.null_map)[idx])
                    break;
                ++idx;
            }
        }

        if unlikely (idx >= context.rows)
        {
            ++k;
            continue;
        }

        const auto & key = key_getter.getJoinKeyWithBufferHint(idx);
        size_t hash = static_cast<HashValueType>(Hash()(key));
        size_t bucket = pointer_table.getBucketNum(hash);
        state->pointer_ptr = pointer_table.getPointerTable() + bucket;
        PREFETCH_READ(state->pointer_ptr);

        if constexpr (!KeyGetterType::isJoinKeyTypeReference())
            state->key = key;
        if constexpr (tagged_pointer)
            state->hash_tag = hash & ROW_PTR_TAG_MASK;
        if constexpr (KeyGetterType::joinKeyCompareHashFirst())
            state->hash = hash;
        state->index = idx;
        state->stage = ProbePrefetchStage::FindHeader;
        ++active_states;
        ++idx;
        ++k;
    }
    wd.probe_hash_table_time += watch.elapsedFromLastTime();
    FlushBatchIfNecessary<true>();
    wd.insert_time += watch.elapsedFromLastTime();
    FillNullMap(current_offset);
}

template <typename KeyGetter, bool has_null_map, AddRowType add_row_type, bool tagged_pointer>
void NO_INLINE JoinProbeBlockHelper<KeyGetter, has_null_map, add_row_type, tagged_pointer>::joinProbeBlockLeftOuter()
{}

template <typename KeyGetter, bool has_null_map, AddRowType add_row_type, bool tagged_pointer>
void NO_INLINE
JoinProbeBlockHelper<KeyGetter, has_null_map, add_row_type, tagged_pointer>::joinProbeBlockLeftOuterPrefetch()
{}

template <typename KeyGetter, bool has_null_map, AddRowType add_row_type, bool tagged_pointer>
void NO_INLINE JoinProbeBlockHelper<KeyGetter, has_null_map, add_row_type, tagged_pointer>::joinProbeBlockSemi()
{}

template <typename KeyGetter, bool has_null_map, AddRowType add_row_type, bool tagged_pointer>
void NO_INLINE JoinProbeBlockHelper<KeyGetter, has_null_map, add_row_type, tagged_pointer>::joinProbeBlockSemiPrefetch()
{}

template <typename KeyGetter, bool has_null_map, AddRowType add_row_type, bool tagged_pointer>
void NO_INLINE JoinProbeBlockHelper<KeyGetter, has_null_map, add_row_type, tagged_pointer>::joinProbeBlockAnti()
{}

template <typename KeyGetter, bool has_null_map, AddRowType add_row_type, bool tagged_pointer>
void NO_INLINE JoinProbeBlockHelper<KeyGetter, has_null_map, add_row_type, tagged_pointer>::joinProbeBlockAntiPrefetch()
{}

template <typename KeyGetter, bool has_null_map, AddRowType add_row_type, bool tagged_pointer>
template <bool has_other_condition>
void NO_INLINE JoinProbeBlockHelper<KeyGetter, has_null_map, add_row_type, tagged_pointer>::joinProbeBlockRightOuter()
{}

template <typename KeyGetter, bool has_null_map, AddRowType add_row_type, bool tagged_pointer>
template <bool has_other_condition>
void NO_INLINE
JoinProbeBlockHelper<KeyGetter, has_null_map, add_row_type, tagged_pointer>::joinProbeBlockRightOuterPrefetch()
{}

template <typename KeyGetter, bool has_null_map, AddRowType add_row_type, bool tagged_pointer>
template <bool has_other_condition>
void NO_INLINE JoinProbeBlockHelper<KeyGetter, has_null_map, add_row_type, tagged_pointer>::joinProbeBlockRightSemi()
{}

template <typename KeyGetter, bool has_null_map, AddRowType add_row_type, bool tagged_pointer>
template <bool has_other_condition>
void NO_INLINE
JoinProbeBlockHelper<KeyGetter, has_null_map, add_row_type, tagged_pointer>::joinProbeBlockRightSemiPrefetch()
{}

template <typename KeyGetter, bool has_null_map, AddRowType add_row_type, bool tagged_pointer>
template <bool has_other_condition>
void NO_INLINE JoinProbeBlockHelper<KeyGetter, has_null_map, add_row_type, tagged_pointer>::joinProbeBlockRightAnti()
{}

template <typename KeyGetter, bool has_null_map, AddRowType add_row_type, bool tagged_pointer>
template <bool has_other_condition>
void NO_INLINE
JoinProbeBlockHelper<KeyGetter, has_null_map, add_row_type, tagged_pointer>::joinProbeBlockRightAntiPrefetch()
{}

template <typename KeyGetter, bool has_null_map, AddRowType add_row_type, bool tagged_pointer>
void JoinProbeBlockHelper<KeyGetter, has_null_map, add_row_type, tagged_pointer>::joinProbeBlockImpl()
{
#define CALL(JoinType)                        \
    if (pointer_table.enableProbePrefetch())  \
        joinProbeBlock##JoinType##Prefetch(); \
    else                                      \
        joinProbeBlock##JoinType();

#define CALL2(JoinType, has_other_condition)                       \
    if (pointer_table.enableProbePrefetch())                       \
        joinProbeBlock##JoinType##Prefetch<has_other_condition>(); \
    else                                                           \
        joinProbeBlock##JoinType<has_other_condition>();

    bool has_other_condition = non_equal_conditions.other_cond_expr != nullptr;
    if (kind == Inner)
        CALL(Inner)
    else if (kind == LeftOuter)
        CALL(LeftOuter)
    else if (kind == Semi && !has_other_condition)
        CALL(Semi)
    else if (kind == Anti && !has_other_condition)
        CALL(Anti)
    else if (kind == RightOuter && has_other_condition)
        CALL2(RightOuter, true)
    else if (kind == RightOuter)
        CALL2(RightOuter, false)
    else if (kind == RightSemi && has_other_condition)
        CALL2(RightSemi, true)
    else if (kind == RightSemi)
        CALL2(RightSemi, false)
    else if (kind == RightAnti && has_other_condition)
        CALL2(RightAnti, true)
    else if (kind == RightAnti)
        CALL2(RightAnti, false)
    else
        throw Exception("Logical error: unknown combination of JOIN", ErrorCodes::LOGICAL_ERROR);

#undef CALL2
#undef CALL
}

void joinProbeBlock(
    JoinProbeContext & context,
    JoinProbeWorkerData & wd,
    HashJoinKeyMethod method,
    ASTTableJoin::Kind kind,
    const JoinNonEqualConditions & non_equal_conditions,
    const HashJoinSettings & settings,
    const HashJoinPointerTable & pointer_table,
    const HashJoinRowLayout & row_layout,
    MutableColumns & added_columns)
{
    if (context.rows == 0)
        return;

    switch (method)
    {
    case HashJoinKeyMethod::Empty:
    case HashJoinKeyMethod::Cross:
        break;

#define CALL(KeyGetter, has_null_map, add_row_type, tagged_pointer)              \
    JoinProbeBlockHelper<KeyGetter, has_null_map, add_row_type, tagged_pointer>( \
        context,                                                                 \
        wd,                                                                      \
        method,                                                                  \
        kind,                                                                    \
        non_equal_conditions,                                                    \
        settings,                                                                \
        pointer_table,                                                           \
        row_layout,                                                              \
        added_columns)                                                           \
        .joinProbeBlockImpl();

#define CALL3(KeyGetter, has_null_map, add_row_type)        \
    if (pointer_table.enableTaggedPointer())                \
    {                                                       \
        CALL(KeyGetter, has_null_map, add_row_type, true);  \
    }                                                       \
    else                                                    \
    {                                                       \
        CALL(KeyGetter, has_null_map, add_row_type, false); \
    }

#define CALL2(KeyGetter, has_null_map)                                     \
    switch (row_layout.add_row_type)                                       \
    {                                                                      \
    case AddRowType::KeyAllRawNeeded:                                      \
    {                                                                      \
        CALL3(KeyGetter, has_null_map, AddRowType::KeyAllRawNeeded);       \
        break;                                                             \
    }                                                                      \
    case AddRowType::KeyRawNeededOnly:                                     \
    {                                                                      \
        CALL3(KeyGetter, has_null_map, AddRowType::KeyRawNeededOnly);      \
        break;                                                             \
    }                                                                      \
    case AddRowType::OtherColumnNeededOnly:                                \
    {                                                                      \
        CALL3(KeyGetter, has_null_map, AddRowType::OtherColumnNeededOnly); \
        break;                                                             \
    }                                                                      \
    case AddRowType::AllNeeded:                                            \
    {                                                                      \
        CALL3(KeyGetter, has_null_map, AddRowType::AllNeeded);             \
        break;                                                             \
    }                                                                      \
    case AddRowType::NoNeeded:                                             \
    {                                                                      \
        CALL3(KeyGetter, has_null_map, AddRowType::NoNeeded);              \
        break;                                                             \
    }                                                                      \
    }

#define CALL1(KeyGetter)         \
    if (context.null_map)        \
    {                            \
        CALL2(KeyGetter, true);  \
    }                            \
    else                         \
    {                            \
        CALL2(KeyGetter, false); \
    }

#define M(METHOD)                                                                          \
    case HashJoinKeyMethod::METHOD:                                                        \
        using KeyGetterType##METHOD = HashJoinKeyGetterForType<HashJoinKeyMethod::METHOD>; \
        CALL1(KeyGetterType##METHOD);                                                      \
        break;
        APPLY_FOR_HASH_JOIN_VARIANTS(M)
#undef M

#undef CALL1
#undef CALL2
#undef CALL3
#undef CALL

    default:
        throw Exception("Unknown JOIN keys variant.", ErrorCodes::UNKNOWN_SET_DATA_VARIANT);
    }
}

} // namespace DB