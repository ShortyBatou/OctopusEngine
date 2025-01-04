#pragma once
#include <Dynamic/FEM/ContinuousMaterial.h>
#include "Core/Base.h"

__device__ __forceinline__ Matrix3x3 get_strain(const Matrix3x3 &F);

__device__ __forceinline__ Matrix3x3 get_strain_linear(const Matrix3x3 &F);

__device__ __forceinline__ Matrix3x3 hooke_stress(const Matrix3x3 &F, scalar lambda, scalar mu);
__device__ __forceinline__ Matrix3x3 stvk_stress(const Matrix3x3 &F, scalar lambda, scalar mu);
__device__ __forceinline__ Matrix3x3 snh_stress(const Matrix3x3 &F, scalar lambda, scalar mu);

__device__ void hooke_hessian(const Matrix3x3 &F, scalar lambda, scalar mu, Matrix3x3 d2W_dF2[6]);
__device__ void stvk_hessian(const Matrix3x3 &F, scalar lambda, scalar mu, Matrix3x3 d2W_dF2[6]);
__device__ void snh_hessian(const Matrix3x3 &F, scalar lambda, scalar mu, Matrix3x3 d2W_dF2[6]);

__device__ Matrix3x3 eval_pk1_stress(Material material, scalar lambda, scalar mu, const Matrix3x3 &F);
__device__ void eval_hessian(Material material, scalar lambda, scalar mu, const Matrix3x3 &F, Matrix3x3 d2W_dF2[6]);

__device__ Matrix3x3 pk1_to_cauchy_stress(const Matrix3x3 &F, const Matrix3x3 &P);
__device__ scalar von_mises_stress(const Matrix3x3 &C);

__device__ Matrix3x3 assemble_sub_hessian(const Vector3& dFdx, const scalar& V, Matrix3x3 d2W_dF2[6]);