#pragma once
#include <Dynamic/FEM/ContinuousMaterial.h>
#include "Core/Base.h"
#include "cuda_runtime.h"

__device__ Matrix3x3 get_strain(const Matrix3x3 &F);

__device__ Matrix3x3 get_strain_linear(const Matrix3x3 &F);

__device__ void hooke_stress(const Matrix3x3 &F, scalar lambda, scalar mu,Matrix3x3 &P);
__device__ void stvk_stress(const Matrix3x3 &F, scalar lambda, scalar mu,Matrix3x3 &P);
__device__ void snh_stress(const Matrix3x3 &F, scalar lambda, scalar mu,Matrix3x3 &P);

__device__ void hooke_hessian(const Matrix3x3 &F, scalar lambda, scalar mu, Matrix3x3 d2W_dF2[9]);
__device__ void stvk_hessian(const Matrix3x3 &F, scalar lambda, scalar mu, Matrix3x3 d2W_dF2[9]);
__device__ void snh_hessian(const Matrix3x3 &F, scalar lambda, scalar mu, Matrix3x3 d2W_dF2[9]);

__device__ void eval_stress(Material material, scalar lambda, scalar mu, const Matrix3x3 &F, Matrix3x3 &P);
__device__ void eval_hessian(Material material, scalar lambda, scalar mu, const Matrix3x3 &F, Matrix3x3 d2W_dF2[9]);