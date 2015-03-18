#ifndef __HELPER_FUNCTIONS_H__
#define __HELPER_FUNCTIONS_H__
int is_match(pixel x_t, gaussian g, float t);
float update_weight(float w, float l_a, int m);
void update_distribution(pixel x_t, float mean, float variance);
float gaussian_prob_density_function(pixel x_t, float mean, float variance);
int num_background(float t, float *weights);
#endif
