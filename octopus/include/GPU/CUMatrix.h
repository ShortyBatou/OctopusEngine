#pragma once
#include "Core/Base.h"



// global device function
__device__ scalar mat3x3_trace(const Matrix3x3 &m);

__device__ scalar squared_trace(const Matrix3x3 &m);
__device__ scalar squared_norm(const Matrix3x3& m);

__device__ Matrix3x3 vec_hat(const Vector3 &v);

__device__ void print_vec(const Vector3 &v);
__device__ void print_mat(const Matrix3x3 &m);
__device__ void print_mat(const Matrix2x2 &m);
