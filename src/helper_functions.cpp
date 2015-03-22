#include <cstdio>
#include <cmath>
#include "helper_functions.h"

using std::swap;

// when given a gaussian distribution g, a current pixel x_t, returns the Z-score 
float getZ(int x_t, const gaussian &g) {
    return (x_t - g.mean) * (x_t - g.mean) / g.variance;
}

// w is current weight
// l_a is alpha value for learning rate
// m is 1 for the model which matched the current pixel and 0 for the remaining models
// TODO after this approximation, the weights are renormalized.
float update_weight(float w, float l_a, int m) {
    return (1 - l_a) * w + l_a * m;
}

// The mean and variance parameters for unmatched distributions remain the same
// Only the distribution which matches the new observation would call this function to update its parameters 
void update_distribution(int x_t, gaussian &g) {
    g.mean = (1.0 - RHO) * g.mean + RHO * x_t;
    // matlab code uses same variance for 3 colors... 
    g.variance = (1.0 - RHO) * g.variance + RHO * (x_t - g.mean) * (x_t - g.mean);
}

// TODO B = argmin_b(sum(1, b) w > T)
int num_background(float t, float *weights) {
    return 1;
}

inline float weight_over_sigma(float weight, float v1, float v2, float v3) {
    return weight / pow(v1*v2*v3, 1.0/6.0);
}

int is_background(int selected_gaussian, float w[K], struct gaussian g[K][3]) {
    // sort by weight / sqrt(variance) 
    int sorted_index[K];
    for (int i=0; i<K; i++) {
        sorted_index[i] = i;
    }
    // bubble sort...
    for (int i=0; i<K; i++) {
        for (int j=0; j<K-1; j++) {
            if (weight_over_sigma(w[j], g[j][0].variance, g[j][1].variance, g[j][2].variance) < 
                    weight_over_sigma(w[j+1], g[j+1][0].variance, g[j+1][1].variance, g[j+1][2].variance)) {
		swap(sorted_index[j], sorted_index[j+1]);
		swap(w[j], w[j+1]);
		swap(g[j][0], g[j+1][0]);
		swap(g[j][1], g[j+1][1]);
		swap(g[j][2], g[j+1][2]);
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

int connected_component(int y, int x, char foreground[height][width]) {
    int counter = 1;
    for (int i=-1; i<=1; i++) {
        for (int j=-1; j<=1; j++) {
            if (i == 0 && j == 0) continue;
            if (y + i >= 0 && y + i < height && x + j >= 0 && x + j < width && 
                    foreground[y+i][x+i] == -1) {
                foreground[y+i][x+i] = foreground[y][x];
                counter += connected_component(y+i, x+i, foreground);
            }
        }
    }
    return counter;
}

/*
// TODO check the equation....
// covariance_matrix = variance ^ 2 * I for simplicity
float gaussian_prob_density_function(pixel x_t, float mean, float variance) {
    covariance_matrix = variance ^ 2 * I;
    return (E ^ ((-1/2) * (x_t - mean)^T * (covariance) ^ (-1) * (x_t - mean))) / ((2 * pi) ^ (n / 2) * sqrt(covariance_matrix));
}
*/

