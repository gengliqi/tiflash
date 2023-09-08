#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

void runComputeEngineDaemon();

typedef struct CRawString {
    char * data;
    size_t size;
} CRawString;

CRawString dispatchMPPTask(const char *, size_t);

typedef struct MPPResultStream MPPResultStream;

MPPResultStream * establishMPPConnection(char *, size_t);

CRawString next(MPPResultStream *);

CRawString cancelMPPTask(char *, size_t);

#ifdef __cplusplus
}
#endif
