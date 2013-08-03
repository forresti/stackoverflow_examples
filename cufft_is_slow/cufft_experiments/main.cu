//#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cufft.h>
#include "helpers.h"
using namespace std;

void cufftForward_dpmData(){
    int NbFeatures = 32;
    int maxRows = 256;
    int maxCols = 256;
    int nIter = 10;

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
                  NbFeatures, //istride
                  1, //idist
                  onembed, //onembed
                  NbFeatures, //ostride
                  1, //odist
                  CUFFT_R2C, //cufftType
                  NbFeatures /*batch*/));
    //CHECK_CUFFT(cufftSetCompatibilityMode(forwardPlan, CUFFT_COMPATIBILITY_FFTW_ALL));
    
    float* h_in = (float*)malloc(sizeof(float)*maxRows*cols_padded*NbFeatures); //note cols_padded instead of maxCols
    memset(h_in, 0, sizeof(float)*maxRows*cols_padded*NbFeatures);


#if 0
    for(int row=0; row<maxRows; row++){
        for(int col=0; col<maxCols; col++){ //iterate through maxCols, but multiply row by cols_padded.
            for(int depth=0; depth<NbFeatures; depth++){
                int idx = row * cols_padded * NbFeatures +
                          col * NbFeatures +
                          depth;
                h_in[idx] = row * maxCols * NbFeatures + //using maxCols instead of cols_padded here, so that input data is same for in-place and out-of-place versions
                            col * NbFeatures +
                            depth;
//                printf("h_in[row=%d, col=%d, depth=%d] = %f \n", row, col, depth, h_in[idx]);
            }
        }
    }
#endif
    cufftComplex* h_freq = reinterpret_cast<cufftComplex*>(h_in); 
    CHECK_CUDART(cudaMalloc(&d_in, sizeof(float)*maxRows*cols_padded*NbFeatures)); //cols_padded varies depending on whether in-place or not
//    CHECK_CUDART(cudaMemcpy(d_in, h_in, sizeof(float)*maxRows*cols_padded*NbFeatures, cudaMemcpyHostToDevice));
    d_freq = reinterpret_cast<cufftComplex*>(d_in);
    
    double start = read_timer();
    for(int i=0; i<nIter; i++){

        CHECK_CUFFT(cufftExecR2C(forwardPlan, d_in, d_freq));
        CHECK_CUDART(cudaDeviceSynchronize());
    }
    double responseTime = read_timer() - start;
    printf("did %d FFT calls in %f ms \n", nIter, responseTime);

//    CHECK_CUDART(cudaMemcpy(h_freq, d_freq, sizeof(float)*maxRows*cols_padded*NbFeatures, cudaMemcpyDeviceToHost)); //TODO: copy exactly the right amount of space to host

    for(int i=0; i<(maxRows*maxCols*NbFeatures); i++){
//         printf("    cufft h_freq[%d].(x,y) = %0.0f, %0.0f \n", i, h_freq[i].x, h_freq[i].y); //for StackOverflow 
        //printf("cufft h_freq[%d].(x,y) = %0.10f,%0.10f \n", i, h_freq[i].x, h_freq[i].y);
    }

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
