#pragma once
#include <Dynamic/FEM/ContinuousMaterial.h>
#include "Core/Base.h"
#include "cuda_runtime.h"


__device__ void hooke_first(const Matrix3x3 &F, Matrix3x3 &P, scalar &C);
__device__ void hooke_second(const Matrix3x3 &F, Matrix3x3 &P, scalar &C);

__device__ void stvk_first(const Matrix3x3 &F, Matrix3x3 &P, scalar &C);
__device__ void stvk_second(const Matrix3x3 &F, Matrix3x3 &P, scalar &C);
__device__ void dsnh_first(const Matrix3x3 &F, Matrix3x3 &P, scalar &C);
__device__ void dsnh_second(const Matrix3x3 &F, Matrix3x3 &P, scalar &C);

__device__ void snh_first(const Matrix3x3 &F, Matrix3x3 &P, scalar &C, scalar alpha);
__device__ void snh_second(const Matrix3x3 &F, Matrix3x3 &P, scalar &C);

__device__ void corot_first(const Matrix3x3 &F, Matrix3x3 &P, scalar &C);
__device__ void corot_second(const Matrix3x3 &F, Matrix3x3 &P, scalar &C);

__device__ void fixed_corot_first(const Matrix3x3 &F, Matrix3x3 &P, scalar &C);
__device__ void fixed_corot_second(const Matrix3x3 &F, Matrix3x3 &P, scalar &C);

__device__ void eval_material(Material material, int m, scalar lambda, scalar mu, const Matrix3x3 &F, Matrix3x3 &P, scalar &energy);