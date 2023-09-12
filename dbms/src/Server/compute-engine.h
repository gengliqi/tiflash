#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

void runComputeEngineDaemon();

typedef struct RawString
{
    void * data;
    size_t size;
} RawString;

RawString newRawString(void * data, size_t size);
void deleteRawString(RawString s);

// FlashService
extern void * global_flash_context;

int dispatchMPPTask(void * ctx, RawString raw_request, RawString * raw_response);

typedef struct MPPStreamResponse MPPStreamResponse;

int establishMPPConnection(void * ctx, RawString raw_request, MPPStreamResponse ** stream_response);
bool nextResponse(MPPStreamResponse * stream_response, RawString * raw_response);
void deleteMPPStreamResponse(MPPStreamResponse * stream_response);

int cancelMPPTask(void * ctx, RawString raw_request, RawString * raw_response);

#ifdef __cplusplus
}
#endif
