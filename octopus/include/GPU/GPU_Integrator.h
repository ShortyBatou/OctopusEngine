#pragma once
#include "GPU_Dynamic.h"
#include "GPU_ParticleSystem.h"
#include "Core/Base.h"

enum IntegratorType
{
    Explicit, RK2, RK4
};

struct GPU_Integrator : GPU_Dynamic {
    GPU_Integrator(bool _apply_constraints = true) : apply_constraints(_apply_constraints) { }

    void step(GPU_ParticleSystem* ps, scalar dt) override = 0;

    virtual void add_dynamics(GPU_Dynamic* dynamic) { _dynamics.push_back(dynamic); }
    virtual void add_constraint(GPU_Dynamic* constraint) { _constraints.push_back(constraint); }

    virtual void init_dynamics(GPU_ParticleSystem *ps, scalar dt);
    virtual void eval_dynamics(GPU_ParticleSystem *ps, scalar dt);
    virtual void eval_constraints(GPU_ParticleSystem *ps, scalar dt);

    ~GPU_Integrator() override
    {
        for(const GPU_Dynamic* dynamic: _dynamics) delete dynamic;
        for(const GPU_Dynamic* dynamic: _constraints) delete dynamic;
    }

    bool apply_constraints;
    std::vector<GPU_Dynamic*> _dynamics;
    std::vector<GPU_Dynamic*> _constraints;
};

struct GPU_SemiExplicit final : GPU_Integrator {
    void step(GPU_ParticleSystem* ps, scalar dt) override;
};

struct RK_Data
{
    Vector3* x;
    Vector3* v;
    int step;
    scalar w1;
    scalar w2;
    Vector3* dv;
    Vector3* dx;
};

struct GPU_Explicit_RK : GPU_Integrator
{
    GPU_Explicit_RK(const int nb, const std::vector<scalar>& _w1, const std::vector<scalar>& _w2) : w1(_w1), w2(_w2)
    {
        order = w1.size();
        const std::vector<Vector3> values(nb, Vector3(0, 0, 0));
        cb_kx = new Cuda_Buffer<Vector3>(values);
        cb_kv = new Cuda_Buffer<Vector3>(values);
        cb_x = new Cuda_Buffer<Vector3>(values);
        cb_v = new Cuda_Buffer<Vector3>(values);
    }
    void step(GPU_ParticleSystem* ps, scalar dt) override;

    ~GPU_Explicit_RK() override
    {
        delete cb_kx;
        delete cb_kv;
        delete cb_x;
        delete cb_v;
    }
private:
    int order;
    std::vector<scalar> w1,w2;
    Cuda_Buffer<Vector3>* cb_x, *cb_v;
    Cuda_Buffer<Vector3>* cb_kx, *cb_kv;
};

struct GPU_Explicit_RK2 final : GPU_Explicit_RK {
    GPU_Explicit_RK2(int nb) : GPU_Explicit_RK(nb, {1,0.5f}, {1,1})
    { }
};

struct GPU_Explicit_RK4 final : GPU_Explicit_RK {
    GPU_Explicit_RK4(int nb) : GPU_Explicit_RK(nb, {0.5f,0.5,1.f,1.f/6.f}, {1,0.5,0.5,1})
    { }
};

__global__ void kenerl_semi_exicit_integration(int n, scalar dt, Vector3 g, GPU_ParticleSystem_Parameters ps);
