#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#define WIDTH 6
#define HEIGHT 2

#define CHECK_CUDART(x) do { \
  cudaError_t res = (x); \
  if(res != cudaSuccess) { \
    fprintf(stderr, "CUDART: %s = %d (%s) at (%s:%d)\n", #x, res, cudaGetErrorString(res),__FILE__,__LINE__); \
    exit(1); \
  } \
} while(0) 

__global__ void printGpu_tex(cudaTextureObject_t tex) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    if(tidx < WIDTH && tidy < HEIGHT){
        float x = tex2D<float>(tex, tidy, tidx);
        //float x = tex2D<float>(tex, float(tidx)+0.5, float(tidy)+0.5);
        printf("tex2D<float>(tex, %d, %d) = %f \n", tidy, tidx, x);
    }
}

__global__ void printGpu_vanilla(float* d_buffer, int pitch) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    if(tidx < WIDTH && tidy < HEIGHT){
        float x = d_buffer[tidy*pitch + tidx];
        printf("d_buffer[%d][%d] = %f \n", tidy, tidx, x);
    }
}

int main() {
    int width = WIDTH;
    int height = HEIGHT; 
    float h_buffer[12] = {1,2,3,4,5,6,7,8,9,10,11,12};

    float* d_buffer;
    size_t pitch;
    CHECK_CUDART(cudaMallocPitch(&d_buffer, &pitch, sizeof(float)*width, height));
    CHECK_CUDART(cudaMemset2D(d_buffer, pitch, 0, pitch, height));
    CHECK_CUDART(cudaMemcpy2D(d_buffer, pitch, &h_buffer, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));
    printf("pitch = %d \n", pitch);

    //CUDA 5 texture objects: https://developer.nvidia.com/content/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = d_buffer;
    resDesc.res.pitch2D.pitchInBytes =  pitch;
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.pitch2D.desc.x = 32; // bits per channel
    resDesc.res.pitch2D.desc.y = 32;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t tex;
    CHECK_CUDART(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    dim3 grid(1, 1, 1); //assume one small block
    dim3 block(WIDTH, HEIGHT, 1);
    printGpu_tex<<<grid, block>>>(tex);
    CHECK_CUDART(cudaGetLastError());
    printGpu_vanilla<<<grid, block>>>(d_buffer, pitch/sizeof(float));
    CHECK_CUDART(cudaGetLastError());
    cudaDestroyTextureObject(tex);
    cudaFree(d_buffer);
}


