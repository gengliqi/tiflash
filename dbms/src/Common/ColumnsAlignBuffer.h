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

#pragma once

#include <Common/PODArray.h>

#ifdef TIFLASH_ENABLE_AVX_SUPPORT
#include <immintrin.h>
#endif

namespace DB
{
#ifdef TIFLASH_ENABLE_AVX_SUPPORT
struct AlignBufferAVX2
{
    static constexpr size_t vector_size = sizeof(__m256i);
    static constexpr size_t buffer_size = 2 * vector_size;

    union
    {
        char data[buffer_size]{};
        __m256i v[2];
    };
};

class ColumnsAlignBufferAVX2
{
public:
    ColumnsAlignBufferAVX2() = default;

    void resize(size_t n)
    {
        buffers.resize(n, AlignBufferAVX2::buffer_size);
        sizes.resize_fill_zero(n, AlignBufferAVX2::buffer_size);
    }

    void reset(bool need_flush_)
    {
        current_index = 0;
        need_flush = need_flush_;
    }

    size_t nextIndex()
    {
        if unlikely (current_index >= buffers.size())
            resize(current_index + 1);
        return current_index++;
    }

    AlignBufferAVX2 & getAlignBuffer(size_t index)
    {
        assert(index < buffers.size());
        return buffers[index];
    }

    UInt8 & getSize(size_t index)
    {
        assert(index < sizes.size());
        return sizes[index];
    }

    bool needFlush() const { return need_flush; }

private:
    size_t current_index = 0;
    bool need_flush = false;
    PaddedPODArray<AlignBufferAVX2> buffers;
    static_assert(UINT8_MAX >= AlignBufferAVX2::buffer_size);
    PaddedPODArray<UInt8> sizes;
};

#else
class ColumnsAlignBufferAVX2
{
};
#endif
} // namespace DB