#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "helpers.h"
using namespace std;

double read_timer(){
    struct timeval start;
    gettimeofday( &start, NULL );
    return (double)((start.tv_sec) + 1.0e-6 * (start.tv_usec)) * 1000; //in ms
}

//thanks: http://stackoverflow.com/questions/16267149/cufft-error-handling
const char* cufftGetErrorString(cufftResult error)
{
    switch (error){
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
    }
    return "<unknown>";
}

//thanks to this post: www.cplusplus.com/forum/general/17771 (user Duoas)
float* readCsv_1dFloat(int length, char* fname){ //TODO: replace char* with string?
    float* myCsv = (float*)malloc(sizeof(float) * length);
    ifstream infile(fname);
    int idx = 0;
    while(infile){
        string s;
        if (!getline( infile, s )) break;
        istringstream ss( s );
        vector <string> record;
        while (ss)
        {
            string s;
            if (!getline( ss, s, ',' )) break;
            myCsv[idx] = atof(s.c_str());
            idx++;
            if(idx >= length) break;
        }
        if(idx >= length) break;
    }

    printf("in readCsv_1dFloat(). final idx = %d \n", idx);
    return myCsv;
}

