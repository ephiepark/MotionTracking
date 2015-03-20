#ifndef __HELPER_FUNCTIONS_H__
#define __HELPER_FUNCTIONS_H__

const float RHO = 0.02;

struct gaussian {
    float mean;
    float variance;
};

float getZ(int x_t, const gaussian &g);
float update_weight(float w, float l_a, int m);
void update_distribution(int x_t, gaussian &g);
float gaussian_prob_density_function(int x_t, float mean, float variance);
int num_background(float t, float *weights);
#endif

