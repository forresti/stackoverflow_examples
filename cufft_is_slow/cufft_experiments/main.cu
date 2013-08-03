//#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cufft.h>
#include "helpers.h"
using namespace std;

void cufftForward_dpmData(){
    int depth = 32;
    int maxRows = 1024;
    int maxCols = 1024;
    int nIter = 2;

    int n[2] = {maxRows, maxCols};
    int cols_padded;
    //if maxCols is even, cols_padded = (maxCols+2). if maxCols is odd, cols_padded = (maxCols+1)
    cols_padded = 2*(maxCols/2 + 1); //allocate this width, but tell FFTW that it's maxCols width
    int inembed[2] = {maxRows, 2*(maxCols/2 + 1)};
    int onembed[2] = {maxRows, (maxCols/2 + 1)}; //default -- equivalent ot onembed=NULL in FFTW

    cufftHandle forwardPlan; 
    float* d_in; cufftComplex* d_freq; 

    CHECK_CUFFT(cufftPlanMany(&forwardPlan,
                  2, //rank
                  n, //dimensions = {maxRows, maxCols}
                  inembed, //inembed
                  depth, //istride
                  1, //idist
                  onembed, //onembed
                  depth, //ostride
                  1, //odist
                  CUFFT_R2C, //cufftType
                  depth /*batch*/));
    
    CHECK_CUDART(cudaMalloc(&d_in, sizeof(float)*maxRows*cols_padded*depth)); //cols_padded varies depending on whether in-place or not
    d_freq = reinterpret_cast<cufftComplex*>(d_in);
    
    double start = read_timer();
    for(int i=0; i<nIter; i++){

        CHECK_CUFFT(cufftExecR2C(forwardPlan, d_in, d_freq));
    }
    CHECK_CUDART(cudaDeviceSynchronize());
    double responseTime = read_timer() - start;
    printf("did %d FFT calls in %f ms \n", nIter, responseTime);

    //TODO: free memory
}

void deviceStuff(){
//    cudaSetDevice(2); 
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
