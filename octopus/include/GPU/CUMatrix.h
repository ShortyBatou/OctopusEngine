#pragma once
#include "Core/Base.h"

// reduction
/**
 * tid: current thread
 * block_size: nb values in shared buffer size(s_data) / v_size
 * offset : where is the beginning
 * v_size : size of final data vector
 * s_data : shared data buffer
 */
template<typename T>
__device__ void all_reduction(const int tid, const int block_size, const int offset, const int v_size, T* s_data) {
    __syncthreads();
    for(int i = block_size/2, b=(block_size+1)/2; i > 0; b=(b+1)/2, i/=2) {
        if(tid < i) {
            for(int j = 0; j < v_size; ++j) {
                s_data[(offset + tid)*v_size+j] += s_data[(offset + tid+b)*v_size+j];
            }
        }
        __syncthreads();
        i = (b>i) ? b : i;
    }
}


__device__ void vec_reduction(int tid, int block_size, int offset, int v_size, scalar* s_data);

// global device function
__device__ scalar mat3x3_trace(const Matrix3x3 &m);
__device__ Matrix3x3 mat3x3_com(const Matrix3x3 &m);

__device__ scalar squared_trace(const Matrix3x3 &m);
__device__ scalar squared_norm(const Matrix3x3& m);

__device__ Matrix3x3 vec_hat(const Vector3 &v);

__device__ void mat3x3_polardecomposition(const Matrix3x3 &F, Matrix3x3 &R, Matrix3x3 &S, int max_it = 10, scalar tol = 1e-8);
__device__ bool mat_3x3_is_symmetric(const glm::mat3& A, scalar eps = 1e-6f);
__device__ bool mat_3x3_is_positive_definite(const glm::mat3& A);
__device__ void print_vec(const Vector3 &v);
__device__ void print_mat(const Matrix3x3 &m);
__device__ void print_mat(const Matrix2x2 &m);
