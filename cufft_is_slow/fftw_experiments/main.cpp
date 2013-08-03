#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
extern "C" {
#include <fftw3.h>
}
#include "helpers.h"
using namespace std;


void fftwForward_dpmData(){
    int depth = 32;
    int maxRows = 512;
    int maxCols = 512;
    int nIter = 8; //equivalent to 1024x1024x32 for 2 iterations

    int n[2] = {maxRows, maxCols};
    int cols_padded;

    //if maxCols is even, cols_padded = (maxCols+2). if maxCols is odd, cols_padded = (maxCols+1)
    cols_padded = 2*(maxCols/2 + 1); //allocate this width, but tell FFTW that it's maxCols width
    int inembed[2] = {maxRows, 2*(maxCols/2 + 1)};
    int onembed[2] = {maxRows, (maxCols/2 + 1)}; //default -- equivalent ot onembed=NULL

    float* h_in = (float*)malloc(sizeof(float)*maxRows*cols_padded*depth);
    memset(h_in, 0, sizeof(float)*maxRows*cols_padded*depth);

    fftwf_complex* h_freq = reinterpret_cast<fftwf_complex*>(h_in); //in-place version

    fftwf_plan forwardPlan = fftwf_plan_many_dft_r2c(2, //rank
                                                     n, //dims -- this doesn't include zero-padding
                                                     depth, //howmany
                                                     h_in, //in
                                                     inembed, //NULL, //inembed
                                                     depth, //istride
                                                     1, //idist
                                                     h_freq, //out
                                                     onembed, //NULL, //onembed
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
    printf("fftwForward_dpmData() \n");
    fftwForward_dpmData();
    
    return 0;
}

