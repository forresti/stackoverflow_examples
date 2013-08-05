#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
extern "C" {
#include <fftw3.h>
}
#include "helpers.h"
using namespace std;

#define OUT_OF_PLACE
void fftwForward_experiment(){
    int depth = 32;
    int nRows = 256;
    int nCols = 256;
    int nIter = 8; 
    int n[2] = {nRows, nCols};

    #ifdef OUT_OF_PLACE
    //if nCols is even, cols_padded = (nCols+2). if nCols is odd, cols_padded = (nCols+1)
    int cols_padded = 2*(nCols/2 + 1); //allocate this width, but tell FFTW that it's nCols width
    int inembed[2] = {nRows, 2*(nCols/2 + 1)};
    int onembed[2] = {nRows, (nCols/2 + 1)}; //default -- equivalent of onembed=NULL
    #else
    int cols_padded = nCols;
    int inembed[2] = {nRows, nCols};
    int onembed[2] = {nRows, (nCols/2 + 1)}; //default -- equivalent of onembed=NULL
    #endif

    float* h_in = (float*)malloc(sizeof(float)*nRows*cols_padded*depth);
    memset(h_in, 0, sizeof(float)*nRows*cols_padded*depth);

    #ifdef OUT_OF_PLACE
    fftwf_complex* h_freq = reinterpret_cast<fftwf_complex*>(h_in); //in-place version
    #else
    fftwf_complex* h_freq = (fftwf_complex*)malloc(sizeof(fftwf_complex)*nRows*cols_padded*depth);
    #endif

    fftwf_plan forwardPlan = fftwf_plan_many_dft_r2c(2, //rank
                                                     n, //dims -- this doesn't include zero-padding
                                                     depth, //howmany
                                                     h_in, //in
                                                     inembed, //inembed
                                                     depth, //istride
                                                     1, //idist
                                                     h_freq, //out
                                                     onembed, //onembed
                                                     depth, //ostride
                                                     1, //odist
                                                     FFTW_PATIENT /*flags*/);
    double start = read_timer();
#pragma omp parallel for
    for(int i=0; i<nIter; i++){
        fftwf_execute_dft_r2c(forwardPlan, h_in, h_freq);
    }
    double responseTime = read_timer() - start;
    printf("did %d FFT calls in %f ms \n", nIter, responseTime);

    free(h_in);
}

int main (int argc, char **argv)
{
    fftwForward_experiment();
    return 0;
}

