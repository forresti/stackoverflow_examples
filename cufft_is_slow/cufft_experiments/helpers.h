#ifndef __HELPERS_H__
#define __HELPERS_H__
#include <vector>
#include <string>
#include <cufft.h>
#include <sys/time.h>
using namespace std;

#define CHECK_CUDART(x) do { \
  cudaError_t res = (x); \
  if(res != cudaSuccess) { \
    fprintf(stderr, "CUDART: %s = %d (%s) at (%s:%d)\n", #x, res, cudaGetErrorString(res),__FILE__,__LINE__); \
    exit(1); \
  } \
} while(0)

#define CHECK_CUFFT(x) do { \
  cufftResult res = (x); \
  if(res != CUFFT_SUCCESS) { \
    fprintf(stderr, "CUFFT: %s = %d (%s) at (%s:%d)\n", #x, res, cufftGetErrorString(res),__FILE__,__LINE__); \
    exit(1); \
  } \
} while(0)

double read_timer();
const char* cufftGetErrorString(cufftResult error);
float* readCsv_1dFloat(int length, char* fname);

#endif

