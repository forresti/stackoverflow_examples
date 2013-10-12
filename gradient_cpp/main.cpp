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

    img.convertTo(img, CV_32FC3); //silly to promote to float, but this is just a sanity check impl. 
    Mat gradX(width, height, 3, CV_32FC3);
    Mat gradY(width, height, 3, CV_32FC3);

    for(int y=0; y<height; y++){
        for(int x=0; x<width; x++){
            for(int channel=0; channel<3; channel++){
                gradX.at<cv::Vec3f>(y,x)[channel] = img.at<cv::Vec3f>(y,x)[channel]; //test
                //gradX.at<cv::Vec3f>(y,x)[channel] = fabs(img.at<cv::Vec3f>(y, clamp(x+1, 0, width-1))[channel] - 
                //                                         img.at<cv::Vec3f>(y, clamp(x-1, 0, width-1))[channel]); 
                //float tmp_gradY = img.at<cv::Vec3f>(y+1,x)[channel] - img.at<cv::Vec3f>(y-1,x)[channel];

            }
        }
    }

    imwrite("gradX_gold.jpg", gradX);
}


int main (int argc, char **argv)
{
    Mat img = imread("../Lena.jpg");
    gradient_reference(img);

    return 0;
}
