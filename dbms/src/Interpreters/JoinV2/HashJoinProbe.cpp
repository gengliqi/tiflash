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
#include <Common/Exception.h>
#include <Common/Stopwatch.h>
#include <Interpreters/JoinUtils.h>
#include <Interpreters/JoinV2/HashJoinProbe.h>
#include <Interpreters/NullableUtils.h>
#include <common/memcpy.h>

#include <cstdint>

#include "Interpreters/JoinV2/HashJoinRowLayout.h"
#include "Storages/KVStore/Utils.h"

#ifdef TIFLASH_ENABLE_AVX_SUPPORT
ASSERT_USE_AVX2_COMPILE_FLAG
#endif

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
    const TiDB::TiDBCollators & collators,
    const HashJoinRowLayout & row_layout)
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

    resetHashJoinKeyGetter(method, key_getter, key_columns, row_layout);

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
    CopyNext,
};

template <typename KeyGetter, bool KeyTypeReference = KeyGetter::Type::isJoinKeyTypeReference()>
struct ProbePrefetchState;

template <typename KeyGetter>
struct alignas(CPU_CACHE_LINE_SIZE) ProbePrefetchState<KeyGetter, true>
{
    using KeyGetterType = typename KeyGetter::Type;
    using KeyType = typename KeyGetterType::KeyType;
    using HashValueType = typename KeyGetter::HashValueType;

    ProbePrefetchStage stage = ProbePrefetchStage::None;
    bool is_matched = false;
    UInt16 hash_tag = 0;
    UInt16 remaining_length = 0;
    UInt16 buffer_offset = 0;
    UInt32 index = 0;
    HashValueType hash = 0;
    void * ptr;
    RowPtr next_ptr = nullptr;

    ALWAYS_INLINE KeyType getJoinKey(KeyGetterType & key_getter) { return key_getter.getJoinKeyWithBufferHint(index); }
};

template <typename KeyGetter>
struct alignas(CPU_CACHE_LINE_SIZE) ProbePrefetchState<KeyGetter, false>
{
    using KeyGetterType = typename KeyGetter::Type;
    using KeyType = typename KeyGetterType::KeyType;
    using HashValueType = typename KeyGetter::HashValueType;

    ProbePrefetchStage stage = ProbePrefetchStage::None;
    bool is_matched = false;
    UInt16 hash_tag = 0;
    UInt16 remaining_length = 0;
    UInt16 buffer_offset = 0;
    UInt32 index = 0;
    HashValueType hash = 0;
    void * ptr;
    RowPtr next_ptr = nullptr;
    KeyType key{};

    ALWAYS_INLINE KeyType getJoinKey(KeyGetterType &) { return key; }
};


template <typename KeyGetter, bool has_null_map, bool tagged_pointer>
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
        wd.insert_batch.clear();
        wd.insert_batch.reserve(settings.probe_insert_batch_size);

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
            auto hash2 = unalignedLoad<HashValueType>(row_ptr + sizeof(RowPtr));
            if (hash1 != hash2)
                return false;
        }
        return key_getter.joinKeyIsEqual(key1, key2);
    }

    void ALWAYS_INLINE insertRowToBatch(KeyGetterType & key_getter, RowPtr row_ptr, const KeyType & key) const
    {
        wd.insert_batch.push_back(row_ptr + key_getter.getRequiredKeyOffset(key));
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
        wd.align_buffer.resetIndex(force);
        for (auto [column_index, is_nullable] : row_layout.raw_required_key_column_indexes)
        {
            IColumn * column = added_columns[column_index].get();
            if (has_null_map && is_nullable)
                column = &static_cast<ColumnNullable &>(*added_columns[column_index]).getNestedColumn();
            column->deserializeAndInsertFromPos(wd.insert_batch, wd.align_buffer);
        }
        for (auto [column_index, _] : row_layout.other_required_column_indexes)
        {
            added_columns[column_index]->deserializeAndInsertFromPos(wd.insert_batch, wd.align_buffer);
        }

        wd.insert_batch.clear();
    }

    template <bool force>
    void ALWAYS_INLINE FlushBatchIfNecessary2() const
    {
        wd.align_buffer.resetIndex(force);
        for (auto [column_index, is_nullable] : row_layout.raw_required_key_column_indexes)
        {
            IColumn * column = added_columns[column_index].get();
            if (has_null_map && is_nullable)
                column = &static_cast<ColumnNullable &>(*added_columns[column_index]).getNestedColumn();
            column->deserializeAndInsertFromPos(wd.insert_batch, wd.align_buffer);
        }
        for (auto [column_index, _] : row_layout.other_required_column_indexes)
        {
            added_columns[column_index]->deserializeAndInsertFromPos(wd.insert_batch, wd.align_buffer);
        }

        wd.insert_batch.clear();
    }

    /*template <bool force>
    void ALWAYS_INLINE FlushBatchIfNecessary3(size_t probe_buffer_size) const
    {
        const size_t prefetch_step = settings.probe_buffer_prefetch_step;
        wd.probe_buffer_states.resize(prefetch_step);

        size_t i = 0, k = 0;
        size_t active_states = 0;
        size_t probe_buffer2_size = 0;
        while (i < probe_buffer_size || active_states != 0)
        {
            k = k == prefetch_step ? 0 : k;
            auto * state = &wd.probe_buffer_states[k];
            if likely (state->has_work)
            {
                if (state->remaining_len > CPU_CACHE_LINE_SIZE)
                {
                    tiflash_compiler_builtin_memcpy(&wd.probe_buffer2[state->offset], state->ptr, CPU_CACHE_LINE_SIZE);
                    state->remaining_len -= CPU_CACHE_LINE_SIZE;
                    state->offset += CPU_CACHE_LINE_SIZE;
                    state->ptr = reinterpret_cast<RowPtr>(state->ptr) + CPU_CACHE_LINE_SIZE;

                    PREFETCH_READ(state->ptr);
                    ++k;
                    continue;
                }

                switch (state->remaining_len)
                {
                case 16:
                    tiflash_compiler_builtin_memcpy(&wd.probe_buffer[state->offset], state->ptr, 16);
                case 32:
                    tiflash_compiler_builtin_memcpy(&wd.probe_buffer[state->offset], state->ptr, 32);
                case 48:
                    tiflash_compiler_builtin_memcpy(&wd.probe_buffer[state->offset], state->ptr, 48);
                case 64:
                    tiflash_compiler_builtin_memcpy(&wd.probe_buffer[state->offset], state->ptr, 64);
                default:
                    assert(false);
                }
                --active_states;
            }

            state->has_work = true;

            ++k;
        }

        wd.align_buffer.resetIndex(force);
        for (auto [column_index, is_nullable] : row_layout.raw_required_key_column_indexes)
        {
            IColumn * column = added_columns[column_index].get();
            if (has_null_map && is_nullable)
                column = &static_cast<ColumnNullable &>(*added_columns[column_index]).getNestedColumn();
            column->deserializeAndInsertFromPos(wd.insert_batch, wd.align_buffer);
        }
        for (auto [column_index, _] : row_layout.other_required_column_indexes)
        {
            added_columns[column_index]->deserializeAndInsertFromPos(wd.insert_batch, wd.align_buffer);
        }

        wd.insert_batch.clear();
        wd.probe_buffer_size = 0;
    }*/

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
                    null_map_vec.resize_fill_zero(null_map_vec.size() + size);
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

template <typename KeyGetter, bool has_null_map, bool tagged_pointer>
void NO_INLINE JoinProbeBlockHelper<KeyGetter, has_null_map, tagged_pointer>::joinProbeBlockInner()
{
    auto & key_getter = *static_cast<KeyGetterType *>(context.key_getter.get());
    size_t current_offset = 0;
    auto & offsets_to_replicate = wd.offsets_to_replicate;
    size_t idx = context.start_row_idx;
    RowPtr ptr = context.current_probe_row_ptr;
    size_t collision = 0;
    size_t key_offset = sizeof(RowPtr);
    if constexpr (KeyGetterType::joinKeyCompareHashFirst())
    {
        key_offset += sizeof(HashValueType);
    }
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
            const auto & key2 = key_getter.deserializeJoinKey(ptr + key_offset);
            bool key_is_equal = joinKeyIsEqual(key_getter, key, key2, hash, ptr);
            collision += !key_is_equal;
            if (key_is_equal)
            {
                ++current_offset;
                insertRowToBatch(key_getter, ptr + key_offset, key2);
                if unlikely (current_offset >= context.rows)
                    break;
            }

            ptr = HashJoinRowLayout::getNextRowPtr(ptr);
            ptr = removeRowPtrTag(ptr);
            if (ptr == nullptr)
                break;
        }
        offsets_to_replicate[idx] = current_offset;
        if unlikely (ptr != nullptr)
        {
            ptr = HashJoinRowLayout::getNextRowPtr(ptr);
            ptr = removeRowPtrTag(ptr);
            if (ptr == nullptr)
                ++idx;
            break;
        }
    }
    FlushBatchIfNecessary<true>();
    FillNullMap(current_offset);

    context.start_row_idx = idx;
    context.current_probe_row_ptr = ptr;
    wd.collision += collision;
}

template <typename KeyGetter, bool has_null_map, bool tagged_pointer>
void NO_INLINE JoinProbeBlockHelper<KeyGetter, has_null_map, tagged_pointer>::joinProbeBlockInnerPrefetch()
{
    if (settings.probe_buffer_size == 0)
    {
        auto & key_getter = *static_cast<KeyGetterType *>(context.key_getter.get());
        auto & selective_offsets = wd.selective_offsets;
        auto * states = static_cast<ProbePrefetchState<KeyGetter> *>(wd.prefetch_states.get());

        size_t idx = context.start_row_idx;
        size_t active_states = context.prefetch_active_states;
        size_t k = wd.prefetch_iter;
        size_t current_offset = 0;
        size_t collision = 0;
        size_t key_offset = sizeof(RowPtr);
        if constexpr (KeyGetterType::joinKeyCompareHashFirst())
        {
            key_offset += sizeof(HashValueType);
        }
        const size_t probe_prefetch_step = settings.probe_prefetch_step;
        while (idx < context.rows || active_states > 0)
        {
            k = k == probe_prefetch_step ? 0 : k;
            auto * state = &states[k];
            if (state->stage == ProbePrefetchStage::FindNext)
            {
                RowPtr ptr = reinterpret_cast<RowPtr>(state->ptr);
                RowPtr next_ptr = HashJoinRowLayout::getNextRowPtr(ptr);
                next_ptr = removeRowPtrTag(next_ptr);
                if (next_ptr)
                {
                    state->ptr = next_ptr;
                    PREFETCH_READ(next_ptr);
                }

                const auto & key = state->getJoinKey(key_getter);
                const auto & key2 = key_getter.deserializeJoinKey(ptr + key_offset);
                bool key_is_equal = joinKeyIsEqual(key_getter, key, key2, state->hash, ptr);
                collision += !key_is_equal;
                if (key_is_equal)
                {
                    ++current_offset;
                    selective_offsets.push_back(state->index);
                    insertRowToBatch(key_getter, ptr + key_offset, key2);
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
                RowPtr ptr = *reinterpret_cast<RowPtr *>(state->ptr);
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
                        PREFETCH_READ(ptr);
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
            state->ptr = pointer_table.getPointerTable() + bucket;
            PREFETCH_READ(state->ptr);

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

        FlushBatchIfNecessary<true>();
        FillNullMap(current_offset);

        context.start_row_idx = idx;
        context.prefetch_active_states = active_states;
        wd.prefetch_iter = k;
        wd.collision += collision;
    }
    else if (settings.probe_buffer2_size == 0)
    {
        RUNTIME_CHECK_MSG(
            settings.probe_buffer_size <= UINT16_MAX,
            "probe_buffer_size {} > UINT16_MAX",
            settings.probe_buffer_size);
        if (wd.probe_buffer.empty())
            wd.probe_buffer.resize(settings.probe_buffer_size, CPU_CACHE_LINE_SIZE);
        size_t probe_buffer_size = 0;
        const size_t buffer_row_align = 16;
        static_assert(CPU_CACHE_LINE_SIZE % buffer_row_align == 0);

        auto & key_getter = *static_cast<KeyGetterType *>(context.key_getter.get());
        auto & selective_offsets = wd.selective_offsets;
        auto * states = static_cast<ProbePrefetchState<KeyGetter> *>(wd.prefetch_states.get());

        size_t idx = context.start_row_idx;
        size_t active_states = context.prefetch_active_states;
        size_t k = wd.prefetch_iter;
        size_t current_offset = 0;
        size_t collision = 0;
        size_t key_offset = sizeof(RowPtr);
        if constexpr (KeyGetterType::joinKeyCompareHashFirst())
        {
            key_offset += sizeof(HashValueType);
        }
        const size_t probe_prefetch_step = settings.probe_prefetch_step;
        std::deque<UInt8> prefetch_id_queue;
        assert(settings.probe_prefetch_step <= UINT8_MAX);
        while (idx < context.rows || active_states > 0)
        {
            k = k == probe_prefetch_step ? 0 : k;
            auto * state = &states[k];
            if (state->stage == ProbePrefetchStage::CopyNext)
            {
                assert(reinterpret_cast<std::uintptr_t>(state->ptr) % CPU_CACHE_LINE_SIZE == 0);
                if (state->remaining_length > CPU_CACHE_LINE_SIZE)
                {
                    tiflash_compiler_builtin_memcpy(
                        &wd.probe_buffer[state->buffer_offset],
                        state->ptr,
                        CPU_CACHE_LINE_SIZE);
                    state->buffer_offset += CPU_CACHE_LINE_SIZE;
                    state->ptr = reinterpret_cast<RowPtr>(state->ptr) + CPU_CACHE_LINE_SIZE;
                    state->remaining_length -= CPU_CACHE_LINE_SIZE;

                    PREFETCH_READ(state->ptr);
                    ++k;
                    continue;
                }
                switch (state->remaining_length)
                {
                case 16:
                    tiflash_compiler_builtin_memcpy(&wd.probe_buffer[state->buffer_offset], state->ptr, 16);
                    break;
                case 32:
                    tiflash_compiler_builtin_memcpy(&wd.probe_buffer[state->buffer_offset], state->ptr, 32);
                    break;
                case 48:
                    tiflash_compiler_builtin_memcpy(&wd.probe_buffer[state->buffer_offset], state->ptr, 48);
                    break;
                case 64:
                    tiflash_compiler_builtin_memcpy(&wd.probe_buffer[state->buffer_offset], state->ptr, 64);
                    break;
                default:
                    assert(false);
                }
                //do
                //{
                //    tiflash_compiler_builtin_memcpy(&wd.probe_buffer[offset], ptr, buffer_row_align);
                //    offset += buffer_row_align;
                //    ptr += buffer_row_align;
                //    remaining_length -= buffer_row_align;
                //} while (remaining_length > 0);

                if (state->next_ptr)
                {
                    state->stage = ProbePrefetchStage::FindNext;
                    state->ptr = state->next_ptr;
                    PREFETCH_READ(state->ptr);
                    ++k;
                    continue;
                }
                state->stage = ProbePrefetchStage::None;
                --active_states;
            }
            else if (state->stage == ProbePrefetchStage::FindNext)
            {
                RowPtr ptr = reinterpret_cast<RowPtr>(state->ptr);
                RowPtr next_ptr = HashJoinRowLayout::getNextRowPtr(ptr);
                UInt16 len = getRowPtrTag(next_ptr);
                next_ptr = removeRowPtrTag(next_ptr);

                const auto & key = state->getJoinKey(key_getter);
                const auto & key2 = key_getter.deserializeJoinKey(ptr + key_offset);
                bool key_is_equal = joinKeyIsEqual(key_getter, key, key2, state->hash, ptr);
                collision += !key_is_equal;
                if (key_is_equal)
                {
                    ++current_offset;
                    selective_offsets.push_back(state->index);
                    if (len > 0)
                    {
                        RowPtr start = ptr + key_offset + key_getter.getRequiredKeyOffset(key2);
                        RowPtr copy_start = reinterpret_cast<RowPtr>(
                            reinterpret_cast<std::uintptr_t>(start) / buffer_row_align * buffer_row_align);
                        UInt16 diff = start - copy_start;
                        len += diff;
                        UInt16 align_len = (len + buffer_row_align - 1) / buffer_row_align * buffer_row_align;
                        if unlikely (
                            probe_buffer_size + align_len > settings.probe_buffer_size
                            || wd.insert_batch.size() >= settings.probe_insert_batch_size)
                        {
                            prefetch_id_queue.clear();
                            for (size_t i = k + 1; i < probe_prefetch_step; ++i)
                                if (states[i].stage == ProbePrefetchStage::CopyNext)
                                    prefetch_id_queue.push_back(i);
                            for (size_t i = 0; i < k; ++i)
                                if (states[i].stage == ProbePrefetchStage::CopyNext)
                                    prefetch_id_queue.push_back(i);

                            while (!prefetch_id_queue.empty())
                            {
                                auto id = prefetch_id_queue.front();
                                prefetch_id_queue.pop_front();
                                auto * current_state = &states[id];
                                assert(reinterpret_cast<std::uintptr_t>(current_state->ptr) % CPU_CACHE_LINE_SIZE == 0);
                                if (current_state->remaining_length > CPU_CACHE_LINE_SIZE)
                                {
                                    tiflash_compiler_builtin_memcpy(
                                        &wd.probe_buffer[current_state->buffer_offset],
                                        current_state->ptr,
                                        CPU_CACHE_LINE_SIZE);
                                    current_state->buffer_offset += CPU_CACHE_LINE_SIZE;
                                    current_state->ptr
                                        = reinterpret_cast<RowPtr>(current_state->ptr) + CPU_CACHE_LINE_SIZE;
                                    current_state->remaining_length -= CPU_CACHE_LINE_SIZE;

                                    PREFETCH_READ(current_state->ptr);
                                    prefetch_id_queue.push_back(id);
                                    continue;
                                }
                                switch (current_state->remaining_length)
                                {
                                case 16:
                                    tiflash_compiler_builtin_memcpy(
                                        &wd.probe_buffer[current_state->buffer_offset],
                                        current_state->ptr,
                                        16);
                                    break;
                                case 32:
                                    tiflash_compiler_builtin_memcpy(
                                        &wd.probe_buffer[current_state->buffer_offset],
                                        current_state->ptr,
                                        32);
                                    break;
                                case 48:
                                    tiflash_compiler_builtin_memcpy(
                                        &wd.probe_buffer[current_state->buffer_offset],
                                        current_state->ptr,
                                        48);
                                    break;
                                case 64:
                                    tiflash_compiler_builtin_memcpy(
                                        &wd.probe_buffer[current_state->buffer_offset],
                                        current_state->ptr,
                                        64);
                                    break;
                                default:
                                    assert(false);
                                }

                                if (current_state->next_ptr)
                                {
                                    current_state->stage = ProbePrefetchStage::FindNext;
                                    current_state->ptr = current_state->next_ptr;
                                }
                                else
                                {
                                    current_state->stage = ProbePrefetchStage::None;
                                    --active_states;
                                }
                            }

                            //for (size_t i = k + 1; i < probe_prefetch_step; ++i)
                            //{
                            //    if (states[i].stage == ProbePrefetchStage::CopyNext)
                            //    {
                            //        inline_memcpy(
                            //            &wd.probe_buffer[states[i].buffer_offset],
                            //            states[i].ptr,
                            //            states[i].remaining_length);
                            //        if (states[i].next_ptr)
                            //        {
                            //            states[i].stage = ProbePrefetchStage::FindNext;
                            //            states[i].ptr = states[i].next_ptr;
                            //            PREFETCH_READ(states[i].ptr);
                            //        }
                            //        else
                            //        {
                            //            states[i].stage = ProbePrefetchStage::None;
                            //            --active_states;
                            //        }
                            //    }
                            //}
                            //for (size_t i = 0; i < k; ++i)
                            //{
                            //    if (states[i].stage == ProbePrefetchStage::CopyNext)
                            //    {
                            //        inline_memcpy(
                            //            &wd.probe_buffer[states[i].buffer_offset],
                            //            states[i].ptr,
                            //            states[i].remaining_length);
                            //        if (states[i].next_ptr)
                            //        {
                            //            states[i].stage = ProbePrefetchStage::FindNext;
                            //            states[i].ptr = states[i].next_ptr;
                            //            PREFETCH_READ(states[i].ptr);
                            //        }
                            //        else
                            //        {
                            //            states[i].stage = ProbePrefetchStage::None;
                            //            --active_states;
                            //        }
                            //    }
                            //}

                            FlushBatchIfNecessary2<false>();
                            probe_buffer_size = 0;

                            PREFETCH_READ(copy_start);
                            for (size_t i = k + 1; i < probe_prefetch_step; ++i)
                                if (states[i].stage != ProbePrefetchStage::None)
                                    PREFETCH_READ(states[i].ptr);
                            for (size_t i = 0; i < k; ++i)
                                if (states[i].stage != ProbePrefetchStage::None)
                                    PREFETCH_READ(states[i].ptr);
                        }
                        wd.insert_batch.push_back(&wd.probe_buffer[probe_buffer_size + diff]);
                        RowPtr copy_end = reinterpret_cast<RowPtr>(
                            (reinterpret_cast<std::uintptr_t>(start) + CPU_CACHE_LINE_SIZE - 1) / CPU_CACHE_LINE_SIZE
                            * CPU_CACHE_LINE_SIZE);
                        UInt16 buffer_offset = probe_buffer_size;
                        probe_buffer_size += align_len;
                        auto copy_len = std::min(copy_end - copy_start, align_len);
                        switch (copy_len)
                        {
                        case 16:
                            tiflash_compiler_builtin_memcpy(&wd.probe_buffer[buffer_offset], copy_start, 16);
                            break;
                        case 32:
                            tiflash_compiler_builtin_memcpy(&wd.probe_buffer[buffer_offset], copy_start, 32);
                            break;
                        case 48:
                            tiflash_compiler_builtin_memcpy(&wd.probe_buffer[buffer_offset], copy_start, 48);
                            break;
                        case 64:
                            tiflash_compiler_builtin_memcpy(&wd.probe_buffer[buffer_offset], copy_start, 64);
                            break;
                        default:
                            assert(false);
                        }
                        if (copy_end - copy_start < align_len)
                        {
                            buffer_offset += copy_len;
                            align_len -= copy_len;

                            //while (copy_start < copy_end)
                            //{
                            //    tiflash_compiler_builtin_memcpy(&wd.probe_buffer[buffer_offset], copy_start, buffer_row_align);
                            //    buffer_offset += buffer_row_align;
                            //    copy_start += buffer_row_align;
                            //    align_len -= buffer_row_align;
                            //}

                            assert(reinterpret_cast<std::uintptr_t>(copy_end) % CPU_CACHE_LINE_SIZE == 0);
                            PREFETCH_READ(copy_end);
                            state->stage = ProbePrefetchStage::CopyNext;
                            state->remaining_length = align_len;
                            state->buffer_offset = buffer_offset;
                            state->ptr = copy_end;
                            state->next_ptr = next_ptr;
                            ++k;
                            if unlikely (current_offset >= context.rows)
                                break;
                            continue;
                        }
                    }
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

                if (next_ptr)
                {
                    PREFETCH_READ(next_ptr);
                    state->ptr = next_ptr;
                    ++k;
                    continue;
                }

                state->stage = ProbePrefetchStage::None;
                --active_states;
            }
            else if (state->stage == ProbePrefetchStage::FindHeader)
            {
                RowPtr ptr = *reinterpret_cast<RowPtr *>(state->ptr);
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
                        PREFETCH_READ(ptr);
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
            state->ptr = pointer_table.getPointerTable() + bucket;
            PREFETCH_READ(state->ptr);

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

        for (size_t i = 0; i < probe_prefetch_step; ++i)
        {
            if (states[i].stage == ProbePrefetchStage::CopyNext)
            {
                inline_memcpy(&wd.probe_buffer[states[i].buffer_offset], states[i].ptr, states[i].remaining_length);
                if (states[i].next_ptr)
                {
                    states[i].stage = ProbePrefetchStage::FindNext;
                    states[i].ptr = states[i].next_ptr;
                }
                else
                {
                    states[i].stage = ProbePrefetchStage::None;
                    --active_states;
                }
            }
        }
        FlushBatchIfNecessary2<true>();
        FillNullMap(current_offset);

        context.start_row_idx = idx;
        context.prefetch_active_states = active_states;
        wd.prefetch_iter = k;
        wd.collision += collision;
    }
    else
    {
        if (wd.probe_buffer.empty())
            wd.probe_buffer.resize(settings.probe_buffer_size, CPU_CACHE_LINE_SIZE);
        if (wd.probe_buffer2.empty())
            wd.probe_buffer2.resize(settings.probe_buffer2_size, CPU_CACHE_LINE_SIZE);
        size_t probe_buffer_size = 0;
        const size_t buffer_row_align = 16;
        static_assert(CPU_CACHE_LINE_SIZE % buffer_row_align == 0);
        wd.probe_buffer_info.clear();
        wd.probe_buffer_info.reserve(settings.probe_buffer_size * 2 / CPU_CACHE_LINE_SIZE);

        auto & key_getter = *static_cast<KeyGetterType *>(context.key_getter.get());
        auto & selective_offsets = wd.selective_offsets;
        auto * states = static_cast<ProbePrefetchState<KeyGetter> *>(wd.prefetch_states.get());

        size_t idx = context.start_row_idx;
        size_t active_states = context.prefetch_active_states;
        size_t k = wd.prefetch_iter;
        size_t current_offset = 0;
        size_t collision = 0;
        size_t key_offset = sizeof(RowPtr);
        if constexpr (KeyGetterType::joinKeyCompareHashFirst())
        {
            key_offset += sizeof(HashValueType);
        }
        const size_t probe_prefetch_step = settings.probe_prefetch_step;
        while (idx < context.rows || active_states > 0)
        {
            k = k == probe_prefetch_step ? 0 : k;
            auto * state = &states[k];
            if (state->stage == ProbePrefetchStage::FindNext)
            {
                RowPtr ptr = reinterpret_cast<RowPtr>(state->ptr);
                RowPtr next_ptr = HashJoinRowLayout::getNextRowPtr(ptr);
                UInt16 len = getRowPtrTag(next_ptr);
                next_ptr = removeRowPtrTag(next_ptr);
                if (next_ptr)
                {
                    state->ptr = next_ptr;
                    PREFETCH_READ(next_ptr);
                }

                const auto & key = state->getJoinKey(key_getter);
                const auto & key2 = key_getter.deserializeJoinKey(ptr + key_offset);
                bool key_is_equal = joinKeyIsEqual(key_getter, key, key2, state->hash, ptr);
                collision += !key_is_equal;
                if (key_is_equal)
                {
                    ++current_offset;
                    selective_offsets.push_back(state->index);
                    if (len > 0)
                    {
                        RowPtr start = ptr + key_offset + key_getter.getRequiredKeyOffset(key2);
                        RowPtr copy_start = reinterpret_cast<RowPtr>(
                            reinterpret_cast<std::uintptr_t>(start) / buffer_row_align * buffer_row_align);

                        UInt16 diff = start - copy_start;
                        len += diff;
                        UInt16 align_len = (len + buffer_row_align - 1) / buffer_row_align * buffer_row_align;

                        RowPtr copy_end = reinterpret_cast<RowPtr>(
                            (reinterpret_cast<std::uintptr_t>(copy_start) + CPU_CACHE_LINE_SIZE - 1)
                            / CPU_CACHE_LINE_SIZE * CPU_CACHE_LINE_SIZE);
                        UInt16 buffer_offset = probe_buffer_size;
                        UInt16 copy_len = std::min(copy_end - copy_start, align_len);
                        switch (copy_len)
                        {
                        case 16:
                            tiflash_compiler_builtin_memcpy(&wd.probe_buffer[buffer_offset], copy_start, 16);
                        case 32:
                            tiflash_compiler_builtin_memcpy(&wd.probe_buffer[buffer_offset], copy_start, 32);
                        case 48:
                            tiflash_compiler_builtin_memcpy(&wd.probe_buffer[buffer_offset], copy_start, 48);
                        default:
                            assert(false);
                        }
                        //wd.probe_buffer_info.emplace_back(diff, align_len, align_len - copy_len, copy_end);
                        if unlikely (probe_buffer_size + CPU_CACHE_LINE_SIZE > settings.probe_buffer_size)
                        {
                            //FlushBatchIfNecessary3<false>(probe_buffer_size);
                            probe_buffer_size = 0;
                        }
                    }
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
                RowPtr ptr = *reinterpret_cast<RowPtr *>(state->ptr);
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
                        PREFETCH_READ(ptr);
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
            state->ptr = pointer_table.getPointerTable() + bucket;
            PREFETCH_READ(state->ptr);

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

        //FlushBatchIfNecessary3<true>(probe_buffer_size);
        FillNullMap(current_offset);

        context.start_row_idx = idx;
        context.prefetch_active_states = active_states;
        wd.prefetch_iter = k;
        wd.collision += collision;
    }
}

template <typename KeyGetter, bool has_null_map, bool tagged_pointer>
void NO_INLINE JoinProbeBlockHelper<KeyGetter, has_null_map, tagged_pointer>::joinProbeBlockLeftOuter()
{}

template <typename KeyGetter, bool has_null_map, bool tagged_pointer>
void NO_INLINE JoinProbeBlockHelper<KeyGetter, has_null_map, tagged_pointer>::joinProbeBlockLeftOuterPrefetch()
{}

template <typename KeyGetter, bool has_null_map, bool tagged_pointer>
void NO_INLINE JoinProbeBlockHelper<KeyGetter, has_null_map, tagged_pointer>::joinProbeBlockSemi()
{}

template <typename KeyGetter, bool has_null_map, bool tagged_pointer>
void NO_INLINE JoinProbeBlockHelper<KeyGetter, has_null_map, tagged_pointer>::joinProbeBlockSemiPrefetch()
{}

template <typename KeyGetter, bool has_null_map, bool tagged_pointer>
void NO_INLINE JoinProbeBlockHelper<KeyGetter, has_null_map, tagged_pointer>::joinProbeBlockAnti()
{}

template <typename KeyGetter, bool has_null_map, bool tagged_pointer>
void NO_INLINE JoinProbeBlockHelper<KeyGetter, has_null_map, tagged_pointer>::joinProbeBlockAntiPrefetch()
{}

template <typename KeyGetter, bool has_null_map, bool tagged_pointer>
template <bool has_other_condition>
void NO_INLINE JoinProbeBlockHelper<KeyGetter, has_null_map, tagged_pointer>::joinProbeBlockRightOuter()
{}

template <typename KeyGetter, bool has_null_map, bool tagged_pointer>
template <bool has_other_condition>
void NO_INLINE JoinProbeBlockHelper<KeyGetter, has_null_map, tagged_pointer>::joinProbeBlockRightOuterPrefetch()
{}

template <typename KeyGetter, bool has_null_map, bool tagged_pointer>
template <bool has_other_condition>
void NO_INLINE JoinProbeBlockHelper<KeyGetter, has_null_map, tagged_pointer>::joinProbeBlockRightSemi()
{}

template <typename KeyGetter, bool has_null_map, bool tagged_pointer>
template <bool has_other_condition>
void NO_INLINE JoinProbeBlockHelper<KeyGetter, has_null_map, tagged_pointer>::joinProbeBlockRightSemiPrefetch()
{}

template <typename KeyGetter, bool has_null_map, bool tagged_pointer>
template <bool has_other_condition>
void NO_INLINE JoinProbeBlockHelper<KeyGetter, has_null_map, tagged_pointer>::joinProbeBlockRightAnti()
{}

template <typename KeyGetter, bool has_null_map, bool tagged_pointer>
template <bool has_other_condition>
void NO_INLINE JoinProbeBlockHelper<KeyGetter, has_null_map, tagged_pointer>::joinProbeBlockRightAntiPrefetch()
{}

template <typename KeyGetter, bool has_null_map, bool tagged_pointer>
void JoinProbeBlockHelper<KeyGetter, has_null_map, tagged_pointer>::joinProbeBlockImpl()
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

#define CALL(KeyGetter, has_null_map, tagged_pointer)              \
    JoinProbeBlockHelper<KeyGetter, has_null_map, tagged_pointer>( \
        context,                                                   \
        wd,                                                        \
        method,                                                    \
        kind,                                                      \
        non_equal_conditions,                                      \
        settings,                                                  \
        pointer_table,                                             \
        row_layout,                                                \
        added_columns)                                             \
        .joinProbeBlockImpl();

#define CALL2(KeyGetter, has_null_map)        \
    if (pointer_table.enableTaggedPointer())  \
    {                                         \
        CALL(KeyGetter, has_null_map, true);  \
    }                                         \
    else                                      \
    {                                         \
        CALL(KeyGetter, has_null_map, false); \
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
