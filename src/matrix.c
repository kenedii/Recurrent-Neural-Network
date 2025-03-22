#include <math.h>
#include <stdlib.h>

/* Matrix-vector multiplication: result = mat * vec
 * mat is a rows x cols matrix, vec is cols x 1, result is rows x 1 */
void mat_vec_mul(float *mat, float *vec, float *result, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++)
        {
            result[i] += mat[i * cols + j] * vec[j];
        }
    }
}

/* Scaled vector addition: result += alpha * vec
 * result and vec are size x 1, alpha is a scalar */
void vec_add_scaled(float *result, float *vec, int size, float alpha)
{
    for (int i = 0; i < size; i++)
    {
        result[i] += alpha * vec[i];
    }
}

/* Element-wise tanh: vec[i] = tanh(vec[i])
 * vec is size x 1 */
void tanh_vec(float *vec, int size)
{
    for (int i = 0; i < size; i++)
    {
        vec[i] = tanhf(vec[i]);
    }
}

/* Softmax activation: applies softmax to vec in place
 * vec is size x 1 */
void softmax_vec(float *vec, int size)
{
    float max_val = vec[0];
    for (int i = 1; i < size; i++)
    {
        if (vec[i] > max_val)
            max_val = vec[i];
    }
    float sum = 0.0;
    for (int i = 0; i < size; i++)
    {
        vec[i] = expf(vec[i] - max_val);
        sum += vec[i];
    }
    for (int i = 0; i < size; i++)
    {
        vec[i] /= sum;
    }
}

/* Cross-entropy loss: returns -log(output[target])
 * output is size x 1, target is the index of the correct class */
float cross_entropy(float *output, int target, int size)
{
    return -logf(output[target]);
}

/* Add scaled outer product: result += alpha * vec1 * vec2^T
 * result is rows x cols, vec1 is rows x 1, vec2 is cols x 1 */
void add_outer_product(float *result, float *vec1, float *vec2, int rows, int cols, float alpha)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result[i * cols + j] += alpha * vec1[i] * vec2[j];
        }
    }
}

/* Transpose matrix-vector multiplication: result = mat^T * vec
 * mat is rows x cols, mat^T is cols x rows, vec is rows x 1, result is cols x 1 */
void trans_mat_vec_mul(float *mat, float *vec, float *result, int rows, int cols)
{
    for (int k = 0; k < cols; k++)
    {
        result[k] = 0.0;
        for (int i = 0; i < rows; i++)
        {
            result[k] += mat[i * cols + k] * vec[i];
        }
    }
}