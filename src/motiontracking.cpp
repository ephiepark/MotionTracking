#include <vector>
#include "opencv2/opencv.hpp"

#include "helper_functions.h"

#define K 3
#define L_A 0.5 // learning rate

using namespace std;
using namespace cv;

int main(int, char**)
{
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    int index = 0;
    char name[1000];

    vector<Mat> frames;
    int width = (int) cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int height = (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    struct gaussian g[height][width][K][3];
    float w[height][width][K];

    for(;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                int p_r = frame.at<cv::Vec3b>(i, j)[0];
                int p_g = frame.at<cv::Vec3b>(i, j)[1];
                int p_b = frame.at<cv::Vec3b>(i, j)[2];
                for (int k=0; k<K; k++) { 
                    if (is_match(p_r, g[i][j][k][0]) && is_match(p_g, g[i][j][k][1]) && 
                            is_match(p_b, g[i][j][k][2])) {
                        w[i][j][k] = update_weight(w[i][j][k], L_A, 1);
                        update_distribution(p_r, g[i][j][k][0]);
                        update_distribution(p_g, g[i][j][k][1]);
                        update_distribution(p_b, g[i][j][k][2]);
        //              if (is_background(g, arrayofg)) {
        //                  // background
        //              }else{
        //                  // foreground
        //              }
                    }else{
                        w[i][j][k] = update_weight(w[i][j][k], L_A, 0);
                    }
                }
            }
        }
    }
    return 0;
}

