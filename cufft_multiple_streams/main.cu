#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cufft.h>
#include "helpers.h"
using namespace std;

void cufftForward_dpmData(){

    //note: in my 'real' app, I'm already using batch mode. 
    // I need to do multiple batch mode operations, each of which operates on a different chunk of data. 
    // hence, the need for doing cuFFT calls in streams

    int nRows = 16; //each FFT is tiny; doesn't saturate the GPU
    int nCols = 16;
    int number_of_FFTs = 3;

    vector<cufftHandle> forwardPlan(number_of_FFTs);
    vector<float*> d_in(number_of_FFTs);
    vector<cufftComplex*> d_freq(number_of_FFTs);

    int nStreams = number_of_FFTs;
    vector<cudaStream_t> streams(nStreams);
    for(int s=0; s<nStreams; s++){
        cudaStreamCreate(&streams[s]);
    }


    for(int i=0; i<number_of_FFTs; i++){
        CHECK_CUFFT(cufftPlan2d(&forwardPlan[i], nCols, nRows, CUFFT_R2C));
        CHECK_CUDART(cudaMalloc(&d_in[i], sizeof(float)*nRows*nCols));
        CHECK_CUDART(cudaMemset(d_in[i], 0, sizeof(float)*nRows*nCols));
        d_freq[i] = reinterpret_cast<cufftComplex *>(d_in[i]);
    }

    double start = read_timer();
    for(int i=0; i<number_of_FFTs; i++){
        CHECK_CUFFT(cufftExecR2C(forwardPlan[i], d_in[i], d_freq[i]));
    }
    double forwardTime = read_timer() - start;
    printf("time for %d forward FFTs in streams = %f \n", number_of_FFTs, forwardTime);
    printf("avg time per FFT = %f \n", forwardTime/number_of_FFTs);
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
