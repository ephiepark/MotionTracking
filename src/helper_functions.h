#ifndef __HELPER_FUNCTIONS_H__
#define __HELPER_FUNCTIONS_H__

#include <cstdio>
#include <cmath>

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

#define RHO 0.01

using std::swap;
using cv::KalmanFilter;
using cv::Mat_;
using cv::Mat;
using cv::Scalar;

const int K = 3;
const float L_A = 0.02;
const float DEVIATION_SQ_THRESH =  49;
const float INIT_VARIANCE = 3;
const float INIT_MIXPROP = 0.00001;
const float BACKGROUND_THRESH = 0.9;
const float COMPONENT_THRESH = 100;
const int DISTANCE_THRESH = 100000;
const int SIZE_THRESH = 1000000;
const int width = 352, height = 288;

int qx[width*height], qy[width*height];

struct gaussian {
        float mean;
	    float variance;
};

// when given a gaussian distribution g, a current pixel x_t, returns the Z-score 
__device__ float getZ(int x_t, const gaussian &g) {
    return (x_t - g.mean) * (x_t - g.mean) / g.variance;
}

// w is current weight
// l_a is alpha value for learning rate
// m is 1 for the model which matched the current pixel and 0 for the remaining models
// TODO after this approximation, the weights are renormalized.
__device__ float update_weight(float w, float l_a, int m) {
    return (1 - l_a) * w + l_a * m;
}

// The mean and variance parameters for unmatched distributions remain the same
// Only the distribution which matches the new observation would call this function to update its parameters 
__device__ void update_distribution(int x_t, gaussian *gau) {
    gau->mean = (1.0 - RHO) * gau->mean + RHO * x_t;
    // matlab code uses same variance for 3 colors... 
    gau->variance = (1.0 - RHO) * gau->variance + RHO * (x_t - gau->mean) * (x_t - gau->mean);
}

// TODO B = argmin_b(sum(1, b) w > T)
int num_background(float t, float *weights) {
    return 1;
}

__device__ inline float weight_over_sigma(float weight, float v1, float v2, float v3) {
    return weight / pow(v1*v2*v3, (float)(1.0/6.0));
}

__device__ void d_swap(int &a, int &b) {
    int c = a;
    a = b;
    b = c;
}

__device__ void d_swap(float &a, float &b) {
    float c = a;
    a = b;
    b = c;
}

__device__ void d_swap(struct gaussian &a, struct gaussian &b) {
    struct gaussian c = a;
    a = b;
    b = c;
}

__device__ int is_background(int selected_gaussian, float w[K], struct gaussian *g) {
    // sort by weight / sqrt(variance) 
    int sorted_index[K];
    for (int i=0; i<K; i++) {
        sorted_index[i] = i;
    }
    // bubble sort...
    for (int i=0; i<K; i++) {
        for (int j=0; j<K-1; j++) {
            if (weight_over_sigma(w[j], g[j*3+0].variance, g[j*3+1].variance, g[j*3+2].variance) < 
                    weight_over_sigma(w[j+1], g[(j+1)*3+0].variance, g[(j+1)*3+1].variance, g[(j+1)*3+2].variance)) {
		d_swap(sorted_index[j], sorted_index[j+1]);
		d_swap(w[j], w[j+1]);
		d_swap(g[j*3+0], g[(j+1)*3+0]);
		d_swap(g[j*3+1], g[(j+1)*3+1]);
		d_swap(g[j*3+2], g[(j+1)*3+2]);
            }
        }
    }
    
    // accumulated_weight[i] = sum ( sorted_weight[0] + ...sorted_weight[i]);
    float accumulated_weight[K];
    accumulated_weight[0] = w[0];
    if (selected_gaussian == sorted_index[0]) return 1;
    for (int i=1; i<K; i++) {
        accumulated_weight[i] = accumulated_weight[i-1] + w[i];
        if (accumulated_weight[i-1] < BACKGROUND_THRESH) {
	    if (selected_gaussian == sorted_index[i]) return 1;
        }
    }
    return 0;
    // background[i] = 1 if accumulated_weight[i] < BACKGROUND_THRESH;
    // background[0] = 1; always
    // if selected_gaussian is background 
    //  return true
    // if not
    //  return false // foreground
}

int connected_component(int sy, int sx, int foreground[height][width], int &y_sum, int &x_sum) {
    int counter = 1;
    y_sum += sy;
    x_sum += sx;

    int tail = 1;
    qx[0] = sx;
    qy[0] = sy;

    for(int head = 0; head < tail; ++head) {
	int x = qx[head], y = qy[head];

	for (int i=-3; i<=3; i++) {
	    for (int j=-3; j<=3; j++) {
		if (i == 0 && j == 0) continue;
		if (y + i >= 0 && y + i < height && x + j >= 0 && x + j < width && 
			foreground[y+i][x+j] == -1) {
		    foreground[y+i][x+j] = foreground[y][x];
		    ++counter;

		    y_sum += y+i;
		    x_sum += x+j;

		    qy[tail] = y+i;
		    qx[tail] = x+j;
		    ++tail;
		}
	    }
	}
    }
    return counter;
}

KalmanFilter kalman_init(int y, int x) {
    KalmanFilter KF(4, 2, 0);
    KF.transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
    KF.statePre.at<float>(0) = (float)x;
    KF.statePre.at<float>(1) = (float)y;
    KF.statePre.at<float>(2) = 0;
    KF.statePre.at<float>(3) = 0;

    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
    setIdentity(KF.measurementNoiseCov, Scalar::all(10));
    setIdentity(KF.errorCovPost, Scalar::all(.1));

    return KF;
}

void kalman_predict(KalmanFilter &KF, int &y, int &x) {
    Mat prediction = KF.predict();
    x = (int)(prediction.at<float>(0));
    y = (int)(prediction.at<float>(1));
}

void kalman_update(KalmanFilter &KF, int y, int x) {
    Mat_<float> actual(2,1);
    actual.setTo(Scalar(0));
    actual(0) = x;
    actual(1) = y;
    KF.correct(actual); //returns an estimation
}

int get_distance(int a, int b) {
    return a*a + b*b;
}

/*
// TODO check the equation....
// covariance_matrix = variance ^ 2 * I for simplicity
float gaussian_prob_density_function(pixel x_t, float mean, float variance) {
    covariance_matrix = variance ^ 2 * I;
    return (E ^ ((-1/2) * (x_t - mean)^T * (covariance) ^ (-1) * (x_t - mean))) / ((2 * pi) ^ (n / 2) * sqrt(covariance_matrix));
}
*/

#endif
