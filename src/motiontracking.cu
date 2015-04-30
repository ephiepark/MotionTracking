#include <vector>
#include <cstdlib>
#include <ctime>
#include "opencv2/opencv.hpp"

#include "helper_functions.h"

#include <sys/resource.h>

using namespace std;
using namespace cv;


Mat frame;

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {      \
    cudaError_t _m_cudaStat = value; \
    if (_m_cudaStat != cudaSuccess) {   \
        fprintf(stderr, "Error %s at line %d in file %s\n",     \
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
        exit(1);        \
    } }


//
// __global__ kernel for Gaussian
//
__global__ void foreground_g(int *d_imageArray, struct gaussian *d_g, float *d_w, int *d_f, int w, int h) {
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(!(i < h/2 && j < w)) return; //do nothing if indices are invalid

    unsigned int idx = ((i * w) + j);

    int p_r = d_imageArray[idx * 3];
    int p_g = d_imageArray[idx * 3 + 1];
    int p_b = d_imageArray[idx * 3 + 2];
    float min = -1;
    int min_ind = -1;
    for (int k=0; k<K; k++) {
        float z_r = getZ(p_r, d_g[idx * K * 3 + k * 3]);
        float z_g = getZ(p_g, d_g[idx * K * 3 + k * 3 + 1]);
        float z_b = getZ(p_b, d_g[idx * K * 3 + k * 3 + 2]);
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
            d_w[idx * K + k] = update_weight( d_w[idx * K + k], L_A, 1);
            update_distribution(p_r, &d_g[idx * K * 3 + k * 3]);
            update_distribution(p_g, &d_g[idx * K * 3 + k * 3 + 1]);
            update_distribution(p_b, &d_g[idx * K * 3 + k * 3 + 2]);
        }else{
            d_w[idx * K + k] = update_weight(d_w[idx * K + k], L_A, 0);
        }
    }
    if (min_ind == -1) {
        min = -1;
        for (int k=0; k<K; k++) {
            if (min == -1 || min > d_w[idx * K + k]) {
                min = d_w[idx * K + k];
                min_ind = k;
            }
        }
        d_g[idx * K * 3 + min_ind * 3].mean = p_r;
        d_g[idx * K * 3 + min_ind * 3].variance  = INIT_VARIANCE;
        d_g[idx * K * 3 + min_ind * 3 + 1].mean = p_g;
        d_g[idx * K * 3 + min_ind * 3 + 1].variance = INIT_VARIANCE;
        d_g[idx * K * 3 + min_ind * 3 + 2].mean = p_b;
        d_g[idx * K * 3 + min_ind * 3 + 2].variance = INIT_VARIANCE;
        d_w[idx * K + min_ind] = INIT_MIXPROP;
    }
    // renormalized weight
    float sum = 0;
    for (int k=0; k<K; k++) {
        sum += d_w[idx * K + k];
    }
    for (int k=0; k<K; k++) {
        d_w[idx * K + k] = d_w[idx * K + k] / sum;
    }

    if (is_background(min_ind, &d_w[idx * K], &d_g[idx * K * 3])){
        // background
        d_f[idx] = 0;
    } else {
        // foreground
        d_f[idx] = -1;
    }
}

int main(int argc, char **argv) {

    VideoCapture cap;
    char *output = NULL;
    if(argc == 1) {
        cap.open(0);
    } else if(argc == 2) {
        cap.open(argv[1]);
    } else if(argc == 3) {
        if(strcmp(argv[1], "cam") == 0) {
            cap.open(0);
        } else {
            cap.open(argv[1]);
        }
        output = argv[2];
    }

    if(!cap.isOpened()) {
        fprintf(stderr, "Failed to open video source.\n");
        return -1;
    }

    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

    int num_obj = 0;
    int id = 0;

    size_t size_w = height * width * K;
    size_t size_g = size_w * 3;
    size_t size_f = height * width * 2;
    h_w = new float[size_w];
    h_g = new gaussian[size_g];
    h_f = new int[size_f];

    size_t size_frame = height * width;
    int *size_obj_f = new int[size_frame];
    int *x_obj = new int[size_frame];
    int *y_obj = new int[size_frame];
    int *x_obj_prev = new int[size_frame];
    int *y_obj_prev = new int[size_frame];
    int *size_obj = new int[size_frame];
    int *counter_obj = new int[size_frame];
    int *id_obj = new int[size_frame];
    int *flag_obj = new int[size_frame];
    KalmanFilter *kf_obj = new KalmanFilter[size_frame];

    int *h_i = new int[size_frame * 3];
    qx = new int[size_frame];
    qy = new int[size_frame];

    VideoWriter out;

    if(output) {
        out.open(output, CV_FOURCC('8', 'B', 'P', 'S'), cap.get(CV_CAP_PROP_FPS), Size(width, height), true);

        if(!out.isOpened()) {
            fprintf(stderr, "Failed to open output video file.\n");
            return -1;
        }
    }

    Mat frame0;
    cap >> frame0;
    srand(time(0));
    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            for (int k=0; k<K; k++) {
                g(i, j, k, 0).mean = rand() % 256;
                g(i, j, k, 1).mean = rand() % 256;
                g(i, j, k, 2).mean = rand() % 256;
                g(i, j, k, 0).variance = INIT_VARIANCE;
                g(i, j, k, 1).variance = INIT_VARIANCE;
                g(i, j, k, 2).variance = INIT_VARIANCE;
                w(i, j, k) = 0;
            }
            g(i, j, 0, 0).mean = frame0.at<cv::Vec3b>(i, j)[0];
            g(i, j, 0, 1).mean = frame0.at<cv::Vec3b>(i, j)[1];
            g(i, j, 0, 2).mean = frame0.at<cv::Vec3b>(i, j)[2];    
            w(i, j, 0) = 1;
        }
    }

    cudaStream_t stream0, stream1;
    CUDA_CHECK_RETURN(cudaStreamCreate( &stream0 ));
    CUDA_CHECK_RETURN(cudaStreamCreate( &stream1 ));

    // data copy of gaussian 
    struct gaussian *d_g;
    size_t size_g_per_stream = sizeof(struct gaussian) * 3 * width * height * K / 2;
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_g, sizeof(struct gaussian) * 3 * width * height * K));
    CUDA_CHECK_RETURN(cudaMemcpy(d_g, h_g, sizeof(struct gaussian) * 3 * width * height * K, cudaMemcpyHostToDevice));

    // data copy of weight
    float *d_w;
    size_t size_w_per_stream = sizeof(float) * K * width * height / 2;
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_w, sizeof(float) * K * width * height));
    CUDA_CHECK_RETURN(cudaMemcpy(d_w, h_w, sizeof(float) * K * width * height, cudaMemcpyHostToDevice));

    // data allocation of foreground
    int *d_f;
    size_t size_f_per_stream = sizeof(int) * width * height / 2;
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_f, sizeof(int) * width * height));

    // data allocation of image
    int *d_frame0, *d_frame1;
    size_t size_frame_per_stream = size_f_per_stream * 3;
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_frame0, size_frame_per_stream));
    CUDA_CHECK_RETURN(cudaMalloc((void **) &d_frame1, size_frame_per_stream));

    int turn = 0;

    cap >> frame; // get a new frame from camera

    // data copy of image 
    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            h_i[((i * width) + j) * 3 + 0] = frame.at<cv::Vec3b>(i, j)[0];
            h_i[((i * width) + j) * 3 + 1] = frame.at<cv::Vec3b>(i, j)[1];
            h_i[((i * width) + j) * 3 + 2] = frame.at<cv::Vec3b>(i, j)[2];
        }
    }
    CUDA_CHECK_RETURN(cudaMemcpyAsync(d_frame0, h_i, size_frame_per_stream, cudaMemcpyHostToDevice, stream0));
    CUDA_CHECK_RETURN(cudaMemcpyAsync(d_frame1, (char*)(h_i) + size_frame_per_stream, size_frame_per_stream, cudaMemcpyHostToDevice, stream1));

    // define grid and block dimensions
    dim3 dimBlock(32, 32);
    dim3 dimGrid(ceil(width/(double)dimBlock.x), ceil(height/2/(double)dimBlock.y));

    // kernel launch
    foreground_g<<<dimGrid, dimBlock, 0, stream0>>>(d_frame0, d_g, d_w, d_f, width, height);
    foreground_g<<<dimGrid, dimBlock, 0, stream1>>>(d_frame1, (gaussian *)((char*)(d_g) + size_g_per_stream), (float*)((char*)(d_w) + size_w_per_stream), (int*)((char*)(d_f) + size_f_per_stream), width, height);

    // async copy data back
    CUDA_CHECK_RETURN(cudaMemcpyAsync(h_f + turn * height * width, d_f, size_f_per_stream, cudaMemcpyDeviceToHost, stream0));
    CUDA_CHECK_RETURN(cudaMemcpyAsync((char*)(h_f + turn * height * width) + size_f_per_stream, (char*)(d_f) + size_f_per_stream, size_f_per_stream, cudaMemcpyDeviceToHost, stream1));

    CUDA_CHECK_RETURN(cudaDeviceSynchronize()); // Wait for the GPU launched work to complete
    CUDA_CHECK_RETURN(cudaGetLastError());

    turn = 1 - turn;

    for(;;)
    {
        cap >> frame; // get a new frame from camera
        if(frame.empty()) break;

        // data copy of image 
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                h_i[((i * width) + j) * 3 + 0] = frame.at<cv::Vec3b>(i, j)[0];
                h_i[((i * width) + j) * 3 + 1] = frame.at<cv::Vec3b>(i, j)[1];
                h_i[((i * width) + j) * 3 + 2] = frame.at<cv::Vec3b>(i, j)[2];
            }
        }
        CUDA_CHECK_RETURN(cudaMemcpyAsync(d_frame0, h_i, size_frame_per_stream, cudaMemcpyHostToDevice, stream0));
        CUDA_CHECK_RETURN(cudaMemcpyAsync(d_frame1, (char*)(h_i) + size_frame_per_stream, size_frame_per_stream, cudaMemcpyHostToDevice, stream1));

        // define grid and block dimensions
        dim3 dimBlock(32, 32);
        dim3 dimGrid(ceil(width/(double)dimBlock.x), ceil(height/2/(double)dimBlock.y));

        // kernel launch
        foreground_g<<<dimGrid, dimBlock, 0, stream0>>>(d_frame0, d_g, d_w, d_f, width, height);
        foreground_g<<<dimGrid, dimBlock, 0, stream1>>>(d_frame1, (gaussian *)((char*)(d_g) + size_g_per_stream), (float*)((char*)(d_w) + size_w_per_stream), (int*)((char*)(d_f) + size_f_per_stream), width, height);

        // async copy data back
        CUDA_CHECK_RETURN(cudaMemcpyAsync(h_f + turn * height * width, d_f, size_f_per_stream, cudaMemcpyDeviceToHost, stream0));
        CUDA_CHECK_RETURN(cudaMemcpyAsync((char*)(h_f + turn * height * width) + size_f_per_stream, (char*)(d_f) + size_f_per_stream, size_f_per_stream, cudaMemcpyDeviceToHost, stream1));

        int num_obj_f = 0;
        int label = 0;
        for (int i=0; i<height; i++) {
            for (int j=0; j<width; j++) {
                if (foreground(1 - turn, i, j) == -1) {
                    label++;
                    num_obj_f++;
                    foreground(1 - turn, i, j) = label;
                    y_obj[num_obj_f-1] = 0;
                    x_obj[num_obj_f-1] = 0;
                    size_obj_f[num_obj_f-1] = connected_component(i, j, h_f + (1 - turn) * height * width, 
                            y_obj[num_obj_f-1], x_obj[num_obj_f-1]);
                    if (size_obj_f[num_obj_f-1] < COMPONENT_THRESH) {
                        num_obj_f--;
                        continue;
                    }

                    y_obj[num_obj_f-1] = i;
                    x_obj[num_obj_f-1] = j;
                }
            }
        }

        for (int i=0; i<num_obj_f + num_obj; i++) {
            flag_obj[i] = 0;
        }
        for (int k=0; k<num_obj; k++) {
            int min_dis;
            int min_i = -1;
            int kf_p_x, kf_p_y;
            kalman_predict(kf_obj[k], kf_p_y, kf_p_x); 
            min_dis = get_distance(x_obj_prev[k] - kf_p_x, y_obj_prev[k] - kf_p_y) * 2;
            if (min_dis < DISTANCE_THRESH) min_dis = DISTANCE_THRESH;
            for (int i=0; i<num_obj_f; i++) {
                // if the new obj is smaller than the COMPONENT_THRESH throw it away
                if (size_obj_f[i] < COMPONENT_THRESH || flag_obj[i] == 1) continue;
                if (get_distance(x_obj[i] - kf_p_x, y_obj[i] - kf_p_y) < min_dis &&
                        (size_obj[k] - size_obj_f[i]) * (size_obj[k] - size_obj_f[i]) < SIZE_THRESH) {
                    min_dis = get_distance(x_obj[i] - kf_p_x, y_obj[i] - kf_p_y);
                    min_i = i;
                }
            }
            if (min_i == -1) {
                counter_obj[k] -- ;
            }else{
                size_obj[k] = size_obj_f[min_i];
                counter_obj[k] ++ ;
                x_obj_prev[k] = x_obj[min_i];
                y_obj_prev[k] = y_obj[min_i];
                kalman_update(kf_obj[k], y_obj[min_i], x_obj[min_i]);
                flag_obj[min_i] = 1;    
                if (counter_obj[k] > 5) {
                    int min_x=width, min_y=height, max_x=0, max_y=0;
                    for (int i=0; i<height; i++) {
                        for (int j=0; j<width; j++) {
                            if (foreground(1 - turn, i, j) == foreground(1 - turn, y_obj[min_i], x_obj[min_i])) {
                                if (min_x > j) min_x = j;
                                if (min_y > i) min_y = i;
                                if (max_x < j) max_x = j;
                                if (max_y < i) max_y = i;
                            }
                        }
                    }
                    for (int i=min_x; i<=max_x; i++) {
                        frame.at<cv::Vec3b>(min_y, i)[0] = 255;
                        frame.at<cv::Vec3b>(min_y, i)[1] = 255;
                        frame.at<cv::Vec3b>(min_y, i)[2] = 255;

                        frame.at<cv::Vec3b>(max_y, i)[0] = 255;
                        frame.at<cv::Vec3b>(max_y, i)[1] = 255;
                        frame.at<cv::Vec3b>(max_y, i)[2] = 255;
                    }
                    for (int i=min_y; i<=max_y; i++) {
                        frame.at<cv::Vec3b>(i, min_x)[0] = 255;
                        frame.at<cv::Vec3b>(i, min_x)[1] = 255;
                        frame.at<cv::Vec3b>(i, min_x)[2] = 255;

                        frame.at<cv::Vec3b>(i, max_x)[0] = 255;
                        frame.at<cv::Vec3b>(i, max_x)[1] = 255;
                        frame.at<cv::Vec3b>(i, max_x)[2] = 255;
                    }
                }
            }
        }
        for (int k=0; k<num_obj_f; k++) {
            // check size of the object
            if (flag_obj[k] == 0 && size_obj_f[k] >= COMPONENT_THRESH) {
                id_obj[num_obj] = id ++;
                size_obj[num_obj] = size_obj_f[k];
                counter_obj[num_obj] = 1;
                x_obj_prev[num_obj] = x_obj[k];
                y_obj_prev[num_obj] = y_obj[k];
                kf_obj[num_obj++] = kalman_init(y_obj[k], x_obj[k]);
                kalman_update(kf_obj[num_obj-1], y_obj[k], x_obj[k]);
            }
        }
        int counter_tmp = 0;
        for (int i=0; i<num_obj; i++) {
            while (i+counter_tmp < num_obj && counter_obj[i + counter_tmp] < -2) {
                counter_tmp++;
            }
            if (i+counter_tmp >= num_obj) break;
            size_obj[i] = size_obj[i + counter_tmp];
            kf_obj[i] = kf_obj[i + counter_tmp];
            id_obj[i] = id_obj[i + counter_tmp];
            x_obj_prev[i] = x_obj_prev[i+counter_tmp];
            y_obj_prev[i] = y_obj_prev[i+counter_tmp];
            counter_obj[i] = counter_obj[i + counter_tmp];
        }
        num_obj -= counter_tmp;

        if(out.isOpened()) {
            out << frame;
        } else {
            cv::imshow("output", frame);
            waitKey(1);
        }

        CUDA_CHECK_RETURN(cudaDeviceSynchronize());     // Wait for the GPU launched work to complete
        CUDA_CHECK_RETURN(cudaGetLastError());

        turn = 1 - turn;
    }
    return 0;
}

