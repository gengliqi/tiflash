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

#include <Columns/ColumnConst.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnsNumber.h>
#include <Common/typeid_cast.h>
#include <Core/Defines.h>
#include <DataTypes/DataTypeFactory.h>
#include <DataTypes/DataTypeString.h>
#include <IO/ReadHelpers.h>
#include <IO/VarInt.h>
#include <IO/WriteHelpers.h>

#include <ext/scope_guard.h>

#ifdef TIFLASH_ENABLE_AVX_SUPPORT
ASSERT_USE_AVX2_COMPILE_FLAG
#endif

#if __SSE2__
#include <emmintrin.h>
#endif


namespace DB
{
void DataTypeString::serializeBinary(const Field & field, WriteBuffer & ostr) const
{
    const auto & s = get<const String &>(field);
    writeVarUInt(s.size(), ostr);
    writeString(s, ostr);
}


void DataTypeString::deserializeBinary(Field & field, ReadBuffer & istr) const
{
    UInt64 size;
    readVarUInt(size, istr);
    field = String();
    auto & s = get<String &>(field);
    s.resize(size);
    istr.readStrict(&s[0], size);
}


void DataTypeString::serializeBinary(const IColumn & column, size_t row_num, WriteBuffer & ostr) const
{
    const StringRef & s = static_cast<const ColumnString &>(column).getDataAt(row_num);
    writeVarUInt(s.size, ostr);
    writeString(s, ostr);
}


void DataTypeString::deserializeBinary(IColumn & column, ReadBuffer & istr) const
{
    auto & column_string = static_cast<ColumnString &>(column);
    ColumnString::Chars_t & data = column_string.getChars();
    ColumnString::Offsets & offsets = column_string.getOffsets();

    UInt64 size;
    readVarUInt(size, istr);

    size_t old_chars_size = data.size();
    size_t offset = old_chars_size + size + 1;
    offsets.push_back(offset);

    try
    {
        data.resize(offset);
        istr.readStrict(reinterpret_cast<char *>(&data[offset - size - 1]), size);
        data.back() = 0;
    }
    catch (...)
    {
        offsets.pop_back();
        data.resize_assume_reserved(old_chars_size);
        throw;
    }
}


void DataTypeString::serializeBinaryBulk(const IColumn & column, WriteBuffer & ostr, size_t offset, size_t limit) const
{
    const ColumnString & column_string = typeid_cast<const ColumnString &>(column);
    const ColumnString::Chars_t & data = column_string.getChars();
    const ColumnString::Offsets & offsets = column_string.getOffsets();

    size_t size = column_string.size();
    if (!size)
        return;

    size_t end = limit && offset + limit < size ? offset + limit : size;

    if (offset == 0)
    {
        UInt64 str_size = offsets[0] - 1;
        writeVarUInt(str_size, ostr);
        ostr.write(reinterpret_cast<const char *>(data.data()), str_size);

        ++offset;
    }

    for (size_t i = offset; i < end; ++i)
    {
        UInt64 str_size = offsets[i] - offsets[i - 1] - 1;
        writeVarUInt(str_size, ostr);
        ostr.write(reinterpret_cast<const char *>(&data[offsets[i - 1]]), str_size);
    }
}


template <int UNROLL_TIMES>
static NO_INLINE void deserializeBinarySSE2(
    ColumnString::Chars_t & data,
    ColumnString::Offsets & offsets,
    ReadBuffer & istr,
    size_t limit)
{
    size_t offset = data.size();
    for (size_t i = 0; i < limit; ++i)
    {
        if (istr.eof())
            break;

        UInt64 size;
        readVarUInt(size, istr);

        offset += size + 1;
        offsets.push_back(offset);

        data.resize(offset);

        if (size)
        {
#ifdef __SSE2__
            /// An optimistic branch in which more efficient copying is possible.
            if (offset + 16 * UNROLL_TIMES <= data.capacity()
                && istr.position() + size + 16 * UNROLL_TIMES <= istr.buffer().end())
            {
                const auto * sse_src_pos = reinterpret_cast<const __m128i *>(istr.position());
                const __m128i * sse_src_end
                    = sse_src_pos + (size + (16 * UNROLL_TIMES - 1)) / 16 / UNROLL_TIMES * UNROLL_TIMES;
                auto * sse_dst_pos = reinterpret_cast<__m128i *>(&data[offset - size - 1]);

                while (sse_src_pos < sse_src_end)
                {
                    for (size_t j = 0; j < UNROLL_TIMES; ++j)
                        _mm_storeu_si128(sse_dst_pos + j, _mm_loadu_si128(sse_src_pos + j));

                    sse_src_pos += UNROLL_TIMES;
                    sse_dst_pos += UNROLL_TIMES;
                }

                istr.position() += size;
            }
            else
#endif
            {
                istr.readStrict(reinterpret_cast<char *>(&data[offset - size - 1]), size);
            }
        }

        data[offset - 1] = 0;
    }
}


void DataTypeString::deserializeBinaryBulk(
    IColumn & column,
    ColumnsAlignBufferAVX2 * align_buffer [[maybe_unused]],
    ReadBuffer & istr,
    size_t limit,
    double avg_value_size_hint) const
{
    ColumnString & column_string = typeid_cast<ColumnString &>(column);
    ColumnString::Chars_t & chars = column_string.getChars();
    ColumnString::Offsets & offsets = column_string.getOffsets();

    double avg_chars_size = 1; /// By default reserve only for empty strings.

    if (avg_value_size_hint > 0.0 && avg_value_size_hint > sizeof(offsets[0]))
    {
        /// Randomly selected.
        constexpr auto avg_value_size_hint_reserve_multiplier = 1.2;

        avg_chars_size = (avg_value_size_hint - sizeof(offsets[0])) * avg_value_size_hint_reserve_multiplier;
    }

    size_t size_to_reserve = chars.size() + static_cast<size_t>(std::ceil(limit * avg_chars_size));
#ifdef TIFLASH_ENABLE_AVX_SUPPORT
    chars.reserve(size_to_reserve, FULL_VECTOR_SIZE_AVX2);
#else
    chars.reserve(size_to_reserve);
#endif

#ifdef TIFLASH_ENABLE_AVX_SUPPORT
    offsets.reserve(offsets.size() + limit, FULL_VECTOR_SIZE_AVX2);
#else
    offsets.reserve(offsets.size() + limit);
#endif

#ifdef TIFLASH_ENABLE_AVX_SUPPORT
    if (align_buffer)
    {
        size_t prev_size = offsets.size();
        size_t char_size = chars.size();

        bool is_offset_aligned = reinterpret_cast<std::uintptr_t>(&offsets[prev_size]) % FULL_VECTOR_SIZE_AVX2 == 0;
        bool is_char_aligned = reinterpret_cast<std::uintptr_t>(&chars[char_size]) % FULL_VECTOR_SIZE_AVX2 == 0;

        /// Get two next indexes first then get the references to avoid the hang pointer issue
        size_t char_buffer_index = align_buffer->nextIndex();
        size_t offset_buffer_index = align_buffer->nextIndex();

        AlignBufferAVX2 & saved_char_buffer = align_buffer->getAlignBuffer(char_buffer_index);
        UInt8 & char_buffer_size_ref = align_buffer->getSize(char_buffer_index);
        /// Better use a register rather than a reference for a frequently-updated variable
        UInt8 char_buffer_size = char_buffer_size_ref;
        SCOPE_EXIT({ char_buffer_size_ref = char_buffer_size; });

        AlignBufferAVX2 & offset_buffer = align_buffer->getAlignBuffer(offset_buffer_index);
        UInt8 & offset_buffer_size_ref = align_buffer->getSize(offset_buffer_index);
        /// Better use a register rather than a reference for a frequently-updated variable
        UInt8 offset_buffer_size = offset_buffer_size_ref;
        SCOPE_EXIT({ offset_buffer_size_ref = offset_buffer_size; });

        if likely (is_offset_aligned && is_char_aligned)
        {
            struct
            {
                AlignBufferAVX2 buffer;
                char padding[15];
            } tmp_char_buf;

            AlignBufferAVX2 & char_buffer = tmp_char_buf.buffer;

            tiflash_compiler_builtin_memcpy(&char_buffer, &saved_char_buffer, sizeof(AlignBufferAVX2));
            SCOPE_EXIT({ tiflash_compiler_builtin_memcpy(&saved_char_buffer, &char_buffer, sizeof(AlignBufferAVX2)); });

            for (size_t i = 0; i < limit; ++i)
            {
                if (istr.eof())
                    break;

                UInt64 str_size;
                readVarUInt(str_size, istr);

                auto * p = istr.position();
                if likely (p + str_size + 15 <= istr.buffer().end())
                {
                    do
                    {
                        UInt8 remain = FULL_VECTOR_SIZE_AVX2 - char_buffer_size;
                        UInt8 copy_bytes = static_cast<UInt8>(std::min(static_cast<UInt32>(remain), str_size));
                        memcpySmallAllowReadWriteOverflow15(&char_buffer.data[char_buffer_size], p, copy_bytes);
                        p += copy_bytes;
                        char_buffer_size += copy_bytes;
                        str_size -= copy_bytes;
                        if (char_buffer_size == FULL_VECTOR_SIZE_AVX2)
                        {
                            chars.resize(char_size + FULL_VECTOR_SIZE_AVX2, FULL_VECTOR_SIZE_AVX2);
                            _mm256_stream_si256(reinterpret_cast<__m256i *>(&chars[char_size]), char_buffer.v[0]);
                            _mm256_stream_si256(
                                reinterpret_cast<__m256i *>(&chars[char_size + VECTOR_SIZE_AVX2]),
                                char_buffer.v[1]);
                            char_size += FULL_VECTOR_SIZE_AVX2;
                            char_buffer_size = 0;
                        }
                    } while (str_size > 0);
                }
                else
                {
                    do
                    {
                        UInt8 remain = FULL_VECTOR_SIZE_AVX2 - char_buffer_size;
                        UInt8 copy_bytes = static_cast<UInt8>(std::min(static_cast<UInt32>(remain), str_size));
                        inline_memcpy(&char_buffer.data[char_buffer_size], p, copy_bytes);
                        p += copy_bytes;
                        char_buffer_size += copy_bytes;
                        str_size -= copy_bytes;
                        if (char_buffer_size == FULL_VECTOR_SIZE_AVX2)
                        {
                            chars.resize(char_size + FULL_VECTOR_SIZE_AVX2, FULL_VECTOR_SIZE_AVX2);
                            _mm256_stream_si256(reinterpret_cast<__m256i *>(&chars[char_size]), char_buffer.v[0]);
                            _mm256_stream_si256(
                                reinterpret_cast<__m256i *>(&chars[char_size + VECTOR_SIZE_AVX2]),
                                char_buffer.v[1]);
                            char_size += FULL_VECTOR_SIZE_AVX2;
                            char_buffer_size = 0;
                        }
                    } while (str_size > 0);
                }

                istr.position() = p;

                char_buffer.data[char_buffer_size] = 0;
                ++char_buffer_size;
                if unlikely (char_buffer_size == FULL_VECTOR_SIZE_AVX2)
                {
                    chars.resize(char_size + FULL_VECTOR_SIZE_AVX2, FULL_VECTOR_SIZE_AVX2);
                    _mm256_stream_si256(reinterpret_cast<__m256i *>(&chars[char_size]), char_buffer.v[0]);
                    char_size += VECTOR_SIZE_AVX2;
                    _mm256_stream_si256(reinterpret_cast<__m256i *>(&chars[char_size]), char_buffer.v[1]);
                    char_size += VECTOR_SIZE_AVX2;
                    char_buffer_size = 0;
                }

                size_t offset = char_size + char_buffer_size;
                tiflash_compiler_builtin_memcpy(&offset_buffer.data[offset_buffer_size], &offset, sizeof(size_t));
                offset_buffer_size += sizeof(size_t);
                if unlikely (offset_buffer_size == FULL_VECTOR_SIZE_AVX2)
                {
                    offsets.resize(prev_size + FULL_VECTOR_SIZE_AVX2 / sizeof(size_t), FULL_VECTOR_SIZE_AVX2);
                    _mm256_stream_si256(reinterpret_cast<__m256i *>(&offsets[prev_size]), offset_buffer.v[0]);
                    prev_size += VECTOR_SIZE_AVX2 / sizeof(size_t);
                    _mm256_stream_si256(reinterpret_cast<__m256i *>(&offsets[prev_size]), offset_buffer.v[1]);
                    prev_size += VECTOR_SIZE_AVX2 / sizeof(size_t);
                    offset_buffer_size = 0;
                }
            }
            return;
        }
        throw Exception("AlignBuffer is not used due to unaligned data");
    }
#endif
    if (avg_chars_size >= 64)
        deserializeBinarySSE2<4>(chars, offsets, istr, limit);
    else if (avg_chars_size >= 48)
        deserializeBinarySSE2<3>(chars, offsets, istr, limit);
    else if (avg_chars_size >= 32)
        deserializeBinarySSE2<2>(chars, offsets, istr, limit);
    else
        deserializeBinarySSE2<1>(chars, offsets, istr, limit);
}


void DataTypeString::serializeText(const IColumn & column, size_t row_num, WriteBuffer & ostr) const
{
    writeString(static_cast<const ColumnString &>(column).getDataAt(row_num), ostr);
}


void DataTypeString::serializeTextEscaped(const IColumn & column, size_t row_num, WriteBuffer & ostr) const
{
    writeEscapedString(static_cast<const ColumnString &>(column).getDataAt(row_num), ostr);
}


template <typename Reader>
static inline void read(IColumn & column, Reader && reader)
{
    auto & column_string = static_cast<ColumnString &>(column);
    ColumnString::Chars_t & data = column_string.getChars();
    ColumnString::Offsets & offsets = column_string.getOffsets();

    size_t old_chars_size = data.size();
    size_t old_offsets_size = offsets.size();

    try
    {
        reader(data);
        data.push_back(0);
        offsets.push_back(data.size());
    }
    catch (...)
    {
        offsets.resize_assume_reserved(old_offsets_size);
        data.resize_assume_reserved(old_chars_size);
        throw;
    }
}


void DataTypeString::deserializeTextEscaped(IColumn & column, ReadBuffer & istr) const
{
    read(column, [&](ColumnString::Chars_t & data) { readEscapedStringInto(data, istr); });
}


void DataTypeString::serializeTextQuoted(const IColumn & column, size_t row_num, WriteBuffer & ostr) const
{
    writeQuotedString(static_cast<const ColumnString &>(column).getDataAt(row_num), ostr);
}


void DataTypeString::deserializeTextQuoted(IColumn & column, ReadBuffer & istr) const
{
    read(column, [&](ColumnString::Chars_t & data) { readQuotedStringInto<true>(data, istr); });
}


void DataTypeString::serializeTextJSON(
    const IColumn & column,
    size_t row_num,
    WriteBuffer & ostr,
    const FormatSettingsJSON &) const
{
    writeJSONString(static_cast<const ColumnString &>(column).getDataAt(row_num), ostr);
}


void DataTypeString::deserializeTextJSON(IColumn & column, ReadBuffer & istr) const
{
    read(column, [&](ColumnString::Chars_t & data) { readJSONStringInto(data, istr); });
}


void DataTypeString::serializeTextXML(const IColumn & column, size_t row_num, WriteBuffer & ostr) const
{
    writeXMLString(static_cast<const ColumnString &>(column).getDataAt(row_num), ostr);
}


void DataTypeString::serializeTextCSV(const IColumn & column, size_t row_num, WriteBuffer & ostr) const
{
    writeCSVString<>(static_cast<const ColumnString &>(column).getDataAt(row_num), ostr);
}


void DataTypeString::deserializeTextCSV(IColumn & column, ReadBuffer & istr, const char /*delimiter*/) const
{
    read(column, [&](ColumnString::Chars_t & data) { readCSVStringInto(data, istr); });
}


MutableColumnPtr DataTypeString::createColumn() const
{
    return ColumnString::create();
}


bool DataTypeString::equals(const IDataType & rhs) const
{
    return typeid(rhs) == typeid(*this);
}


void registerDataTypeString(DataTypeFactory & factory)
{
    auto creator = static_cast<DataTypePtr (*)()>([] { return DataTypePtr(std::make_shared<DataTypeString>()); });

    factory.registerSimpleDataType("String", creator);

    /// These synonims are added for compatibility.

    factory.registerSimpleDataType("CHAR", creator, DataTypeFactory::CaseInsensitive);
    factory.registerSimpleDataType("VARCHAR", creator, DataTypeFactory::CaseInsensitive);
    factory.registerSimpleDataType("TEXT", creator, DataTypeFactory::CaseInsensitive);
    factory.registerSimpleDataType("TINYTEXT", creator, DataTypeFactory::CaseInsensitive);
    factory.registerSimpleDataType("MEDIUMTEXT", creator, DataTypeFactory::CaseInsensitive);
    factory.registerSimpleDataType("LONGTEXT", creator, DataTypeFactory::CaseInsensitive);
    factory.registerSimpleDataType("BLOB", creator, DataTypeFactory::CaseInsensitive);
    factory.registerSimpleDataType("TINYBLOB", creator, DataTypeFactory::CaseInsensitive);
    factory.registerSimpleDataType("MEDIUMBLOB", creator, DataTypeFactory::CaseInsensitive);
    factory.registerSimpleDataType("LONGBLOB", creator, DataTypeFactory::CaseInsensitive);
}

} // namespace DB
