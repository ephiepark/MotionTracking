#include <vector>
#include <cstdlib>
#include <ctime>
#include "opencv2/opencv.hpp"

#include "helper_functions.h"

using namespace std;
using namespace cv;
const int width = 320, height = 180;

struct gaussian g[height][width][K][3];
float w[height][width][K];
char foreground[height][width];

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
    cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);

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

    puts("Inited Gaussian");

    Mat frame;

    int frameIndex = 0;
    char filename[102];

    for(;;)
    {
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
                    foreground[i][j] = 0;
                    //frame.at<cv::Vec3b>(i, j)[0] = 0;
                    //frame.at<cv::Vec3b>(i, j)[1] = 0;
                    //frame.at<cv::Vec3b>(i, j)[2] = 0;
                }else{
                    // foreground
                    // change to black dot
                    foreground[i][j] = -1;         
                }
            }
        }
        int num_obj = 0;
        int size_obj[256];
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                if (foreground[i][j] == -1) {
                    num_obj++;
                    foreground[i][j] = num_obj;
                    size_obj[num_obj-1] = connected_component(i, j, foreground);
                }
            }
        }
        for (int k=0; k<num_obj; k++) {
            if (size_obj[k] >= size_threshold) {
                for (int i=0; i<height; i++) {
                    for (int j=0; j<width; j++) {
                        if (foreground[i][j] == k+1) {
                            frame.at<cv::Vec3b>(i, j)[0] = 255;
                            frame.at<cv::Vec3b>(i, j)[1] = 255
                            frame.at<cv::Vec3b>(i, j)[2] = 255;
                        }
                    }
                }
            }
        }
        imshow("frame", frame);
        waitKey(1);

        /*
           sprintf(filename, "im%d.png", frameIndex++);
           try {
           imwrite(filename, frame, compression_params);
           }
           catch (runtime_error& ex) {
           fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
           return 1;
           }
           */
    }
    printf("Done\n");
    return 0;
}

