#include <vector>
#include <cstdlib>
#include <ctime>
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

    Mat frame0;
    cap >> frame0;
    srand(time(0));
    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            for (int k=0; k<K; k++) {
                g[i][j][k][0].mean = rand() % 256;
                g[i][j][k][1].mean = rand() % 256;
                g[i][j][k][2].mean = rand() % 256;
                g[i][j][k][0].variance = INIT_VARIANCE;
                g[i][j][k][1].variance = INIT_VARIANCE;
                g[i][j][k][2].variance = INIT_VARIANCE;
                w[i][j][k] = 0;
            }
            g[i][j][0][0].mean = frame0.at<cv::Vec3b>(i, j)[0];
            g[i][j][0][1].mean = frame0.at<cv::Vec3b>(i, j)[1];
            g[i][j][0][2].mean = frame0.at<cv::Vec3b>(i, j)[2];    
            w[i][j][0] = 1;
        }
    }
    for(;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                int p_r = frame.at<cv::Vec3b>(i, j)[0];
                int p_g = frame.at<cv::Vec3b>(i, j)[1];
                int p_b = frame.at<cv::Vec3b>(i, j)[2];
                float min = -1;
                int min_ind = -1;
                for (int k=0; k<K; k++) {
                    float z_r = getZ(p_r, g[i][j][k][0]);
                    float z_g = getZ(p_g, g[i][j][k][1]);
                    float z_b = getZ(p_b, g[i][j][k][2]);
                    float z_sum = z_r + z_g + z_b; 
                    if (min == -1 || min > z_sum) {
                        min = z_sum;
                        min_ind = k;
                    }
                }
                if (min > DEVIATION_SQ_THRESH) {
                    min_ind = -1;
                }
                for (int k=0; k<K; k++) {
                    if (k == min_ind) {
                        w[i][j][k] = update_weight(w[i][j][k], L_A, 1);
                        update_distribution(p_r, g[i][j][k][0]);
                        update_distribution(p_g, g[i][j][k][1]);
                        update_distribution(p_b, g[i][j][k][2]);
                    }else{
                        w[i][j][k] = update_weight(w[i][j][k], L_A, 0);
                    }
                }
                if (min_ind == -1) {
                    min = -1;
                    for (int k=0; k<K; k++) {
                        if (min == -1 || min > w[i][j][k]) { // replacement policy can be changed
                            min = w[i][j][k];
                            min_ind = k;
                        }
                    }
                    g[i][j][min_ind][0].mean = p_r;
                    g[i][j][min_ind][0].variance = INIT_VARIANCE;
                    g[i][j][min_ind][1].mean = p_g;
                    g[i][j][min_ind][1].variance = INIT_VARIANCE;
                    g[i][j][min_ind][2].mean = p_b;
                    g[i][j][min_ind][2].variance = INIT_VARIANCE;
                    w[i][j][min_ind] = INIT_MIXPROP;
                }
                // renormalized weight
                float sum = 0;
                for (int k=0; k<K; k++) {
                    sum += w[i][j][k];
                }
                for (int k=0; k<K; k++) {
                    w[i][j][k] = w[i][j][k] / sum;
                }
                
                if (is_background(min_ind, w[i][j], g[i][j])){
                    // background
                }else{
                    // foreground
                    // change to white dot
                    frame.at<cv::Vec3b>(i, j)[0] = 0;
                    frame.at<cv::Vec3b>(i, j)[1] = 0;
                    frame.at<cv::Vec3b>(i, j)[2] = 0;
                }
            }
        }
    }
    return 0;
}

