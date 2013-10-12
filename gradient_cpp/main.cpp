#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "helpers.h"
using namespace std;
using namespace cv;

static inline int clamp(int idx, int min_idx, int max_idx){
    return max(min_idx, min(idx, max_idx));
}

void gradient_reference(Mat img){
    int width = img.cols;
    int height = img.rows;
   
    Mat gradX(height, width, 3, CV_32FC3);
    Mat gradY(height, width, 3, CV_32FC3);

#if 0
    for(int y=0; y<height; y++){
        for(int x=0; x<width; x++)}
            for(int channel=0; channel<3; channel++){
                float gradX = img.at<cv::Vec3b>(y,x+1)[channel] - img.at<cv::Vec3b>(y,x-1)[channel];
                float tmp_gradY = img.at<cv::Vec3b>(y+1,x)[channel] - img.at<cv::Vec3b>(y-1,x)[channel];

            }
        }
    }
#endif
}


int main (int argc, char **argv)
{
    Mat img = imread("../Lena.jpg");


    return 0;
}
