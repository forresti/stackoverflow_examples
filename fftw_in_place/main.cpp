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
    int howMany = 2;
    int maxRows = 4;
    int maxCols = 4;

    int n[2] = {maxRows, maxCols};

    float* h_in = (float*)malloc(sizeof(float)*maxRows*maxCols*howMany);
    fftwf_complex* h_freq = (fftwf_complex*)malloc(sizeof(fftwf_complex)*maxRows*maxCols*howMany);

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
        printf("h_in[%d] = %f \n", i, h_in[i]);
    }

    fftwf_execute_dft_r2c(forwardPlan, h_in, h_freq);

    for(int i=0; i<(maxRows*maxCols*howMany); i++){
        printf("fftw h_freq[%d][0,1] = %f,%f \n", i, h_freq[i][0], h_freq[i][1]);
    }
    free(h_in);
    free(h_freq);
}

int main (int argc, char **argv)
{
    fftwForward_dpmData();
    return 0;
}

