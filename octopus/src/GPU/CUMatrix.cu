#include "GPU/GPU_PBD.h"

#include <Manager/Debug.h>
#include <Manager/Key.h>

// global device function
__device__ scalar mat3x3_trace(const Matrix3x3 &m) {
    return m[0][0] + m[1][1] + m[2][2];
}

__device__ scalar squared_trace(const Matrix3x3 &m)
{
    return m[0][0] * m[0][0] + m[1][1] * m[1][1] + m[2][2] * m[2][2] + (m[0][1] * m[1][0] + m[1][2] * m[2][1] + m[2][0] * m[0][2]) * 2.f;
}

__device__ scalar squared_norm(const Matrix3x3& m)
{
    scalar st = 0.f;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            st += m[i][j] * m[i][j];
    return st;
}

__device__ void print_vec(const Vector3 &v) {
    printf("x:%f y:%f z:%f", v.x, v.y, v.z);
}

__device__ void print_mat(const Matrix3x3 &m) {
    printf("%f %f %f %f %f %f %f %f %f", m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1], m[0][2], m[1][2],
           m[2][2]);
}

__device__ void print_mat(const Matrix2x2 &m) {
    printf("%f %f %f %f", m[0][0], m[1][0],m[0][1], m[1][1]);
}