#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
extern "C" {
#include <fftw3.h>
}
#include "helpers.h"
using namespace std;

void fftwForward(bool isInPlace){
    int howMany = 1;
    int maxRows = 2;
    int maxCols = 2;
    int n[2] = {maxRows, maxCols};

    float* h_in = (float*)malloc(sizeof(float)*maxRows*maxCols*howMany*2); //*2 is extra space used for in-place version
    fftwf_complex* h_freq;
    if(isInPlace){
        h_freq = reinterpret_cast<fftwf_complex*>(h_in); //in place
    }
    else{
         h_freq = (fftwf_complex*)malloc(sizeof(fftwf_complex)*maxRows*maxCols*howMany); //out of place
    }

    fftwf_plan forwardPlan = fftwf_plan_many_dft_r2c(2, //rank
                                                     n, //dims
                                                     howMany, //howmany
                                                     h_in, //in
                                                     0, //inembed
                                                     howMany, //istride
                                                     1, //idist
                                                     h_freq, //out
                                                     0, //onembed
                                                     howMany, //ostride
                                                     1, //odist
                                                     FFTW_PATIENT /*flags*/);

    for(int i=0; i<(maxRows*maxCols*howMany); i++){
        h_in[i] = (float)i; 
        printf("h_in[%d] = %0.0f \n", i, h_in[i]);
    }

    fftwf_execute_dft_r2c(forwardPlan, h_in, h_freq);

    for(int i=0; i<(maxRows*maxCols*howMany); i++){
        printf("fftw h_freq[%d][0,1] = %0.0f,%0.0f \n", i, h_freq[i][0], h_freq[i][1]);
    }
    free(h_in);
    if(!isInPlace) free(h_freq);
}

int main (int argc, char **argv)
{
    bool isInPlace = false;
    printf("  out-of-place experiment: \n"); 
    fftwForward(isInPlace);

    isInPlace = true;
    printf("\n  in-place experiment: \n");
    fftwForward(isInPlace);

    return 0;
}

