#include <cmath>
#include "helper_functions.h"

// when given a gaussian distribution g, a current pixel x_t, returns the Z-score 
float getZ(int x_t, const gaussian &g) {
    return (x_t - g.mean) / sqrt(g.variance);
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
    g.variance = (1.0 - RHO) * g.variance + RHO * (x_t - g.mean) ^ T * (x_t - g.mean);
}

// TODO B = argmin_b(sum(1, b) w > T)
int num_background(float t, float *weights) {
    return 1;
}

/*
// TODO check the equation....
// covariance_matrix = variance ^ 2 * I for simplicity
float gaussian_prob_density_function(pixel x_t, float mean, float variance) {
    covariance_matrix = variance ^ 2 * I;
    return (E ^ ((-1/2) * (x_t - mean)^T * (covariance) ^ (-1) * (x_t - mean))) / ((2 * pi) ^ (n / 2) * sqrt(covariance_matrix));
}
*/

