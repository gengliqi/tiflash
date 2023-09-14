#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define MY_API __attribute ((visibility ("default")))

MY_API void runComputeEngine(char * config);

typedef struct RawString
{
    void * data;
    uint32_t size;
} RawString;

MY_API RawString newRawString(void * data, uint32_t size);
MY_API void deleteRawString(RawString s);

// TODO: use local context instead of global context
extern void * global_flash_context;

MY_API int dispatchMPPTask(void * ctx, RawString raw_request, RawString * raw_response);

typedef struct MPPStreamResponse MPPStreamResponse;

MY_API int establishMPPConnection(void * ctx, RawString raw_request, MPPStreamResponse ** stream_response);
MY_API int nextResponse(MPPStreamResponse * stream_response, RawString * raw_response);
MY_API void deleteMPPStreamResponse(MPPStreamResponse * stream_response);

MY_API int cancelMPPTask(void * ctx, RawString raw_request, RawString * raw_response);

#ifdef __cplusplus
}
#endif
