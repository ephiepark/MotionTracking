#include <vector>
#include <cstdlib>
#include <ctime>
#include <pthread.h>
#include "opencv2/opencv.hpp"

#include "helper_functions.h"

#include <sys/resource.h>

using namespace std;
using namespace cv;

struct gaussian g[height][width][K][3];
float w[height][width][K];
int foreground[height][width];

int size_obj_f[width*height];
int x_obj[width*height];
int y_obj[width*height];
int h_i[height][width][3];

KalmanFilter kf_obj[height*width];
int size_obj[height*width];
int num_obj = 0;

Mat frame;

pthread_mutex_t ourmutex;

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
    cudaError_t _m_cudaStat = value;										\
    if (_m_cudaStat != cudaSuccess) {										\
        fprintf(stderr, "Error %s at line %d in file %s\n",					\
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
        exit(1);															\
    } }


//
// __global__ kernel for Gaussian
//
__global__ void foreground_g(int *d_imageArray, struct gaussian *d_g, float *d_w, int *d_f) {
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(!(i < height/2 && j < width)) return; //do nothing if indices are invalid

    unsigned int idx = ((i * width) + j);

    int p_r = d_imageArray[idx * 3]; // frame.at<cv::Vec3b>(i, j)[0];
    int p_g = d_imageArray[idx * 3 + 1]; // frame.at<cv::Vec3b>(i, j)[1];
    int p_b = d_imageArray[idx * 3 + 2]; // frame.at<cv::Vec3b>(i, j)[2];
    float min = -1;
    int min_ind = -1;
    for (int k=0; k<K; k++) {
        float z_r = getZ(p_r, d_g[idx * K * 3 + k * 3] /*g[i][j][k][0]*/);
        float z_g = getZ(p_g, d_g[idx * K * 3 + k * 3 + 1] /*g[i][j][k][1]*/);
        float z_b = getZ(p_b, d_g[idx * K * 3 + k * 3 + 2] /*g[i][j][k][2]*/);
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
            d_w[idx * K + k] /* w[i][j][k] */ = update_weight( d_w[idx * K + k] /* w[i][j][k] */, L_A, 1);
            update_distribution(p_r, &d_g[idx * K * 3 + k * 3] /* g[i][j][k][0] */);
            update_distribution(p_g, &d_g[idx * K * 3 + k * 3 + 1] /* g[i][j][k][1] */);
            update_distribution(p_b, &d_g[idx * K * 3 + k * 3 + 2] /* g[i][j][k][2] */);
        }else{
            d_w[idx * K + k] /* w[i][j][k] */ = update_weight(d_w[idx * K + k] /* w[i][j][k] */, L_A, 0);
        }
    }
    if (min_ind == -1) {
        min = -1;
        for (int k=0; k<K; k++) {
            if (min == -1 || min > d_w[idx * K + k] /* w[i][j][k] */) { // replacement policy can be changed
                min = d_w[idx * K + k]; // w[i][j][k];
                min_ind = k;
            }
        }
        d_g[idx * K * 3 + min_ind * 3].mean /* g[i][j][min_ind][0].mean */ = p_r;
        d_g[idx * K * 3 + min_ind * 3].variance /* g[i][j][min_ind][0].variance */  = INIT_VARIANCE;
        d_g[idx * K * 3 + min_ind * 3 + 1].mean /* g[i][j][min_ind][1].mean */ = p_g;
        d_g[idx * K * 3 + min_ind * 3 + 1].variance /* g[i][j][min_ind][1].variance */ = INIT_VARIANCE;
        d_g[idx * K * 3 + min_ind * 3 + 2].mean /* g[i][j][min_ind][2].mean */ = p_b;
        d_g[idx * K * 3 + min_ind * 3 + 2].variance /* g[i][j][min_ind][2].variance */ = INIT_VARIANCE;
        d_w[idx * K + min_ind] /* w[i][j][min_ind] */ = INIT_MIXPROP;
    }
    // renormalized weight
    float sum = 0;
    for (int k=0; k<K; k++) {
        sum += d_w[idx * K + k]; /* w[i][j][k] */
    }
    for (int k=0; k<K; k++) {
        d_w[idx * K + k] = d_w[idx * K + k] / sum; // w[i][j][k] = w[i][j][k] / sum;
    }

    if (is_background(min_ind, &d_w[idx * K] /*w[i][j]*/, &d_g[idx * K * 3] /*g[i][j]*/)){
        // background

        d_f[idx] = 0; // foreground[i][j] = 0;

        //frame.at<cv::Vec3b>(i, j)[0] = 0;
        //frame.at<cv::Vec3b>(i, j)[1] = 0;
        //frame.at<cv::Vec3b>(i, j)[2] = 0;
    } else {
        // foreground
        // change to black dot

        d_f[idx] = -1; // foreground[i][j] = -1;         
    }
}


void *run(void *args) {

    while(1) {
        // lock 
        pthread_mutex_lock(&ourmutex);


        int num_obj_f = 0;
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                if (foreground[i][j] == -1) {
                    num_obj_f++;
                    foreground[i][j] = num_obj_f;
                    y_obj[num_obj_f-1] = 0;
                    x_obj[num_obj_f-1] = 0;
                    size_obj_f[num_obj_f-1] = connected_component(i, j, foreground, 
                            y_obj[num_obj_f-1], x_obj[num_obj_f-1]);
                    y_obj[num_obj_f-1] = y_obj[num_obj_f-1] / size_obj_f[num_obj_f-1];
                    x_obj[num_obj_f-1] = y_obj[num_obj_f-1] / size_obj_f[num_obj_f-1];
                }
            }
        }

        for (int k=0; k<num_obj_f; k++) {
            if (size_obj_f[k] >= COMPONENT_THRESH) {
                int min_dis = DISTANCE_THRESH;
                int min_i = -1;
                for (int i=0; i<num_obj; i++) {
                    int kf_p_x;
                    int kf_p_y;
                    kalman_predict(kf_obj[i], kf_p_y, kf_p_x);
                    if (get_distance(x_obj[k] - kf_p_x, y_obj[k] - kf_p_y) < min_dis && 
                            (size_obj[i] - size_obj_f[k]) * (size_obj[i] - size_obj_f[k]) < SIZE_THRESH) {
                        min_dis = get_distance(x_obj[k] - kf_p_x, y_obj[k] - kf_p_y);
                        min_i = i;
                    }
                }
                if (min_i == -1) {
                    size_obj[num_obj] = size_obj_f[k];
                    kf_obj[num_obj++] = kalman_init(y_obj[k], x_obj[k]);
                    min_i = num_obj-1;
                }else{
                    size_obj[min_i] = size_obj_f[k];
                    kalman_update(kf_obj[min_i], y_obj[k], x_obj[k]);
                }

                int min_x=width, min_y=height, max_x=0, max_y=0;
                for (int i=0; i<height; i++) {
                    for (int j=0; j<width; j++) {
                        if (foreground[i][j] == k+1) {
                            if (min_x > j) min_x = j;
                            if (min_y > i) min_y = i;
                            if (max_x < j) max_x = j;
                            if (max_y < i) max_y = i;
                            /*
                               int color = 256*256*256/(min_i+1);
                               frame.at<cv::Vec3b>(i, j)[0] = color/(256*256);
                               frame.at<cv::Vec3b>(i, j)[1] = color/256%256;
                               frame.at<cv::Vec3b>(i, j)[2] = color%256;
                             */

                        }
                    }
                }
                for (int i=min_x; i<=max_x; i++) {
                    frame.at<cv::Vec3b>(min_y, i)[0] = 255;
                    frame.at<cv::Vec3b>(min_y, i)[1] = 255;
                    frame.at<cv::Vec3b>(min_y, i)[2] = 255;
                    frame.at<cv::Vec3b>(min_y, i)[min_i % 3] = 0;

                    frame.at<cv::Vec3b>(max_y, i)[0] = 255;
                    frame.at<cv::Vec3b>(max_y, i)[1] = 255;
                    frame.at<cv::Vec3b>(max_y, i)[2] = 255;
                    frame.at<cv::Vec3b>(max_y, i)[min_i % 3] = 0;
                }
                for (int i=min_y; i<=max_y; i++) {
                    frame.at<cv::Vec3b>(i, min_x)[0] = 255;
                    frame.at<cv::Vec3b>(i, min_x)[1] = 255;
                    frame.at<cv::Vec3b>(i, min_x)[2] = 255;
                    frame.at<cv::Vec3b>(i, min_x)[min_i % 3] = 0;

                    frame.at<cv::Vec3b>(i, max_x)[0] = 255;
                    frame.at<cv::Vec3b>(i, max_x)[1] = 255;
                    frame.at<cv::Vec3b>(i, max_x)[2] = 255;
                    frame.at<cv::Vec3b>(i, max_x)[min_i % 3] = 0;
                }
            }
        }


        cv::imshow("something", frame);
        waitKey(1);
        // release lock
    }
    pthread_exit(NULL);
}

int main(int argc, char **argv)
{
    VideoCapture cap; // open the default camera
    if(argc == 1) {
        cap.open(0);
    } else {
        cap.open(argv[1]);
    }
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    pthread_mutex_init(&ourmutex, NULL);

    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

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

    cudaStream_t stream0, stream1;
    CUDA_CHECK_RETURN(cudaStreamCreate( &stream0 ));
    CUDA_CHECK_RETURN(cudaStreamCreate( &stream1 ));

    // data copy of gaussian 
    struct gaussian *d_g;
    size_t size_g_per_stream = sizeof(struct gaussian) * 3 * width * height * K / 2;
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_g, sizeof(struct gaussian) * 3 * width * height * K));
    CUDA_CHECK_RETURN(cudaMemcpy(d_g, g, sizeof(struct gaussian) * 3 * width * height * K, cudaMemcpyHostToDevice));

    // data copy of weight
    float *d_w;
    size_t size_w_per_stream = sizeof(float) * K * width * height / 2;
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_w, sizeof(float) * K * width * height));
    CUDA_CHECK_RETURN(cudaMemcpy(d_w, w, sizeof(float) * K * width * height, cudaMemcpyHostToDevice));

    // data allocation of foreground
    int *d_f;
    size_t size_f_per_stream = sizeof(int) * width * height / 2;
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_f, sizeof(int) * width * height));

    // data allocation of image
    int *d_frame0, *d_frame1;
    size_t size_frame_per_stream = size_f_per_stream * 3;
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_frame0, size_frame_per_stream));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_frame1, size_frame_per_stream));

    pthread_t thread;

    if (pthread_create(&thread, NULL, &run, NULL) != 0) {
        fprintf(stderr, "pthread create failed\n");
    }

    for(;;)
    {
        cap >> frame; // get a new frame from camera

        // data copy of image 
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                h_i[i][j][0] = frame.at<cv::Vec3b>(i, j)[0];
                h_i[i][j][1] = frame.at<cv::Vec3b>(i, j)[1];
                h_i[i][j][2] = frame.at<cv::Vec3b>(i, j)[2];
            }
        }
        CUDA_CHECK_RETURN(cudaMemcpyAsync(d_frame0, h_i, size_frame_per_stream, cudaMemcpyHostToDevice, stream0));
        CUDA_CHECK_RETURN(cudaMemcpyAsync(d_frame1, (char*)(h_i) + size_frame_per_stream, size_frame_per_stream, cudaMemcpyHostToDevice, stream1));

        // define grid and block dimensions
        dim3 dimBlock(32, 32);
        dim3 dimGrid(ceil(width/(double)dimBlock.x), ceil(height/2/(double)dimBlock.y));

        // kernel launch
        foreground_g<<<dimGrid, dimBlock, 0, stream0>>>(d_frame0, d_g, d_w, d_f);
        foreground_g<<<dimGrid, dimBlock, 0, stream1>>>(d_frame1, (gaussian *)((char*)(d_g) + size_g_per_stream), (float*)((char*)(d_w) + size_w_per_stream), (int*)((char*)(d_f) + size_f_per_stream));

        // async copy data back
        CUDA_CHECK_RETURN(cudaMemcpyAsync(foreground, d_f, size_f_per_stream, cudaMemcpyDeviceToHost, stream0));
        CUDA_CHECK_RETURN(cudaMemcpyAsync((char*)(foreground) + size_f_per_stream, (char*)(d_f) + size_f_per_stream, size_f_per_stream, cudaMemcpyDeviceToHost, stream1));

        CUDA_CHECK_RETURN(cudaDeviceSynchronize());	// Wait for the GPU launched work to complete
        CUDA_CHECK_RETURN(cudaGetLastError());

        pthread_mutex_unlock(&ourmutex);
    }
    return 0;
}

