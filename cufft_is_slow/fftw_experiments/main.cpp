#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
extern "C" {
#include <fftw3.h>
}
#include "helpers.h"
using namespace std;

//#define DPM_DATA
#define INPLACE

void fftwForward_dpmData(){
    //int NbFeatures = 32;
    //int maxRows = 256;
    //int maxCols = 256;
    int NbFeatures = 2;
    int maxRows = 4;
    int maxCols = 4;

    int n[2] = {maxRows, maxCols};
    int cols_padded;

#ifdef INPLACE
    //if maxCols is even, cols_padded = (maxCols+2). if maxCols is odd, cols_padded = (maxCols+1)
    cols_padded = 2*(maxCols/2 + 1); //allocate this width, but tell FFTW that it's maxCols width
    int inembed[2] = {maxRows, 2*(maxCols/2 + 1)};
    int onembed[2] = {maxRows, (maxCols/2 + 1)}; //default -- equivalent ot onembed=NULL
#else
    cols_padded = maxCols;
    int inembed[2] = {maxRows, maxCols}; 
    int onembed[2] = {maxRows, (maxCols/2 + 1)}; //default -- equivalent ot onembed=NULL
#endif

    float* h_in = (float*)malloc(sizeof(float)*maxRows*cols_padded*NbFeatures);
    memset(h_in, 0, sizeof(float)*maxRows*cols_padded*NbFeatures);

#ifdef INPLACE
    fftwf_complex* h_freq = reinterpret_cast<fftwf_complex*>(h_in); //in-place version
#else
    fftwf_complex* h_freq = (fftwf_complex*)malloc(sizeof(fftwf_complex)*maxRows*maxCols*NbFeatures);
#endif

    fftwf_plan forwardPlan = fftwf_plan_many_dft_r2c(2, //rank
                                                     n, //dims -- this doesn't include zero-padding
                                                     NbFeatures, //howmany
                                                     h_in, //in
                                                     inembed, //NULL, //inembed
                                                     NbFeatures, //istride
                                                     1, //idist
                                                     h_freq, //out
                                                     onembed, //NULL, //onembed
                                                     NbFeatures, //ostride
                                                     1, //odist
                                                     FFTW_PATIENT /*flags*/);


#ifdef DPM_DATA
    free(h_in); //memory space used for autotuning setup.
    h_in = readCsv_1dFloat(maxRows*maxCols*NbFeatures, "../plane_filter_0.csv"); //allocates fresh memory space and reads data from file
#else
    for(int row=0; row<maxRows; row++){
        for(int col=0; col<maxCols; col++){ //iterate through maxCols, but multiply row by cols_padded.
            for(int depth=0; depth<NbFeatures; depth++){
                int idx = row * cols_padded * NbFeatures +
                          col * NbFeatures +
                          depth;
                h_in[idx] = row * maxCols * NbFeatures + //using maxCols instead of cols_padded here, so that input data is same for in-place and out-of-place versions
                            col * NbFeatures +
                            depth;
                printf("h_in[row=%d, col=%d, depth=%d] = %f \n", row, col, depth, h_in[idx]);
            }
        }
    }
#endif
    fftwf_execute_dft_r2c(forwardPlan, h_in, h_freq);

    //while the input had zero padding, that zero padding is now filled with useful data:
    for(int i=0; i<(maxRows*maxCols*NbFeatures); i++){
        printf("    result[%d][0,1] = %0.0f, %0.0f \n", i, h_freq[i][0], h_freq[i][1]); //for StackOverflow
        //printf("    fftw h_freq[%d][0,1] = %0.10f, %0.10f \n", i, h_freq[i][0], h_freq[i][1]);
    }
    free(h_in);
#ifndef INPLACE
    free(h_freq);
#endif
}

int main (int argc, char **argv)
{
    printf("fftwForward_dpmData() \n");
    fftwForward_dpmData();
    
    return 0;
}

