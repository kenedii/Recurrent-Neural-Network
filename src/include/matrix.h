#ifndef MATRIX_H
#define MATRIX_H

void mat_vec_mul(float *mat, float *vec, float *result, int rows, int cols);
void vec_add_scaled(float *result, float *vec, int size, float alpha);
void tanh_vec(float *vec, int size);
void softmax_vec(float *vec, int size);
float cross_entropy(float *output, int target, int size);
void add_outer_product(float *result, float *vec1, float *vec2, int rows, int cols, float alpha);
void trans_mat_vec_mul(float *mat, float *vec, float *result, int rows, int cols);

#endif