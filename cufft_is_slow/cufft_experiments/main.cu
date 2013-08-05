//#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cufft.h>
#include "helpers.h"
using namespace std;

#define OUT_OF_PLACE
void cufftForward_experiment(){
    int depth = 32;
    int nRows = 256;
    int nCols = 256;
    int nIter = 8;
    int n[2] = {nRows, nCols};

    #ifdef OUT_OF_PLACE
    //if nCols is even, cols_padded = (nCols+2). if nCols is odd, cols_padded = (nCols+1)
    int cols_padded = 2*(nCols/2 + 1); //allocate this width, but tell FFTW that it's nCols width
    int inembed[2] = {nRows, 2*(nCols/2 + 1)};
    int onembed[2] = {nRows, (nCols/2 + 1)}; //default -- equivalent ot onembed=NULL in FFTW
    #else
    int cols_padded = nCols;
    int inembed[2] = {nRows, nCols};
    int onembed[2] = {nRows, (nCols/2 + 1)}; //default -- equivalent of onembed=NULL
    #endif

    cufftHandle forwardPlan; 
    float* d_in; cufftComplex* d_freq; 

    CHECK_CUFFT(cufftPlanMany(&forwardPlan,
                  2, //rank
                  n, //dimensions = {nRows, nCols}
                  inembed, //inembed
                  depth, //istride
                  1, //idist
                  onembed, //onembed
                  depth, //ostride
                  1, //odist
                  CUFFT_R2C, //cufftType
                  depth /*batch*/));
    
    CHECK_CUDART(cudaMalloc(&d_in, sizeof(float)*nRows*cols_padded*depth)); 
    #ifdef OUT_OF_PLACE
    d_freq = reinterpret_cast<cufftComplex*>(d_in);
    #else
    CHECK_CUDART(cudaMalloc(&d_freq, sizeof(cufftComplex)*nRows*cols_padded*depth)); 
    #endif    

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
    cufftForward_experiment();
    return 0;
}
