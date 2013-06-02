#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cufft.h>
#include "helpers.h"
using namespace std;

void cufftForward_dpmData(){
    int howMany = 1;
    int maxRows = 4;
    int maxCols = 4;

    int n[2] = {maxRows, maxCols};
    cufftHandle forwardPlan; 
    float* d_in; cufftComplex* d_freq; 

    CHECK_CUFFT(cufftPlanMany(&forwardPlan,
                  2, //rank
                  n, //dimensions = {maxRows, maxCols}
                  0, //inembed
                  howMany, //istride
                  1, //idist
                  0, //onembed
                  howMany, //ostride
                  1, //odist
                  CUFFT_R2C, //cufftType
                  howMany /*batch*/));

    float* h_in = (float*)malloc(sizeof(float) * maxRows*maxCols*howMany);
    for(int i=0; i<(maxRows*maxCols*howMany); i++){
        h_in[i] = (float)i; //* rand();
        printf("h_in[%d] = %f \n", i, h_in[i]);
    }

    cufftComplex* h_freq = (cufftComplex*)malloc(sizeof(cufftComplex)*maxRows*maxCols*howMany);
    CHECK_CUDART(cudaMalloc(&d_in, sizeof(float)*maxRows*maxCols*howMany));
    CHECK_CUDART(cudaMemcpy(d_in, h_in, sizeof(float)*maxRows*maxCols*howMany, cudaMemcpyHostToDevice));
    CHECK_CUDART(cudaMalloc(&d_freq, sizeof(cufftComplex)*maxRows*maxCols*howMany));
    CHECK_CUDART(cudaMemset(d_freq, 0, sizeof(cufftComplex)*maxRows*maxCols*howMany));

    CHECK_CUFFT(cufftExecR2C(forwardPlan, d_in, d_freq));

    CHECK_CUDART(cudaMemcpy(h_freq, d_freq, sizeof(cufftComplex)*maxRows*maxCols*howMany, cudaMemcpyDeviceToHost));

    for(int i=0; i<(maxRows*maxCols*howMany); i++){ 
        printf("cufft h_freq[%d].(x,y) = %f,%f \n", i, h_freq[i].x, h_freq[i].y);
    }
}

void deviceStuff(){
    int device; cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("deviceName = %s \n", prop.name);
}

int main (int argc, char **argv){
    deviceStuff();
    CHECK_CUDART(cudaDeviceSynchronize());
    cufftForward_dpmData();
    return 0;
}
