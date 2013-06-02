#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
extern "C" {
#include <fftw3.h>
}
#include "helpers.h"
using namespace std;

#define DPM_DATA

void fftwForward_dpmData(){
    //int NbFeatures = 32;
    //int maxRows = 256;
    //int maxCols = 256;
    int NbFeatures = 1;
    int maxRows = 4;
    int maxCols = 4;

    int n[2] = {maxRows, maxCols};

    float* h_in = (float*)malloc(sizeof(float)*maxRows*maxCols*NbFeatures);
    fftwf_complex* h_freq = (fftwf_complex*)malloc(sizeof(fftwf_complex)*maxRows*maxCols*NbFeatures);

    fftwf_plan forwardPlan = fftwf_plan_many_dft_r2c(2, //rank
                                                     n, //dims
                                                     NbFeatures, //howmany
                                                     h_in, //in
                                                     0, //inembed
                                                     NbFeatures, //istride
                                                     1, //idist
                                                     h_freq, //out
                                                     0, //onembed
                                                     NbFeatures, //ostride
                                                     1, //odist
                                                     FFTW_PATIENT /*flags*/);


#ifdef DPM_DATA
    free(h_in); //memory space used for autotuning setup.
    h_in = readCsv_1dFloat(maxRows*maxCols*NbFeatures, "../plane_filter_0.csv"); //allocates fresh memory space and reads data from file
#else
    for(int i=0; i<(maxRows*maxCols*NbFeatures); i++){
        h_in[i] = (float)i; //* rand();
        printf("h_in[%d] = %f \n", i, h_in[i]);
    }
#endif
    fftwf_execute_dft_r2c(forwardPlan, h_in, h_freq);

    for(int i=0; i<(maxRows*maxCols*NbFeatures); i++){
        printf("fftw h_freq[%d][0,1] = %0.10f,%0.10f \n", i, h_freq[i][0], h_freq[i][1]);
    }
    free(h_in);
    free(h_freq);
}

int main (int argc, char **argv)
{
    fftwForward_dpmData();
    return 0;
}

