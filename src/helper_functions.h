#ifndef __HELPER_FUNCTIONS_H__
#define __HELPER_FUNCTIONS_H__

const int K = 3;
const float RHO = 0.01;
const float L_A = 0.02;
const float DEVIATION_SQ_THRESH = 49;
const float INIT_VARIANCE = 3;
const float INIT_MIXPROP = 0.00001;
const float BACKGROUND_THRESH = 0.0;
const float COMPONENT_THRESH = 10;

struct gaussian {
    float mean;
    float variance;
};

float getZ(int x_t, const gaussian &g);
float update_weight(float w, float l_a, int m);
void update_distribution(int x_t, gaussian &g);
float gaussian_prob_density_function(int x_t, float mean, float variance);
int num_background(float t, float *weights);
int is_background(int selected_gaussian, float w[K], struct gaussian g[K][3]);
#endif

