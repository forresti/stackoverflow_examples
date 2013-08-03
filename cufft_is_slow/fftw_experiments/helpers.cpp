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
    return (double)((start.tv_sec) + 1.0e-6 * (start.tv_usec))*1000; //in ms
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

