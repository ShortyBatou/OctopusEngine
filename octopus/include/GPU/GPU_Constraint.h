#pragma once
#include "Core/Base.h"
#include <Mesh/Mesh.h>
#include <Mesh/MeshTools.h>
#include <Tools/Area.h>
#include <Tools/Random.h>

#include "GPU/Cuda_Buffer.h"
#include "GPU/GPU_Dynamic.h"

struct GPU_Fix_Constraint final : GPU_Dynamic {
    GPU_Fix_Constraint(const std::vector<Vector3>& positions, Area* arean);
    void step(GPU_ParticleSystem *ps, scalar dt) override;
    ~GPU_Fix_Constraint() override { delete cb_ids;}
    Axis axis;
    Vector3 com;
    Cuda_Buffer<int>* cb_ids;
};

struct GPU_Box_Limit final : GPU_Dynamic {
    GPU_Box_Limit(Vector3 _pmin, Vector3 _pmax) : pmin(_pmin), pmax(_pmax) {}
    void step(GPU_ParticleSystem *ps, scalar dt) override;
    Vector3 pmin, pmax;
};


struct GPU_Crush final : GPU_Dynamic {
    void step(GPU_ParticleSystem *ps, scalar dt) override;
};


struct GPU_RandomSphere final : GPU_Dynamic {
    GPU_RandomSphere(Mesh::Geometry& geo, const scalar r) : radius(r) {
        center = std::accumulate(std::next(geo.begin()), geo.end(), glm::vec3(0.0f)) * (1.f / geo.size());
    }
    void step(GPU_ParticleSystem *ps, scalar dt) override;
    scalar radius;
    Vector3 center;
};


struct GPU_HighStretch final : GPU_Dynamic {
    GPU_HighStretch(Mesh* mesh, const int nb_constraint, const scalar max_d) : t(0), max_dist(max_d) {
        Mesh::Geometry& geo = mesh->geometry();
        center = std::accumulate(std::next(geo.begin()), geo.end(), glm::vec3(0.0f)) * (1.f / geo.size());
        std::vector<int> ids = MeshTools::Get_Surface_Geometry_Ids(mesh);
        std::set<int> chosed;
        std::vector<int> final_ids;
        for(int i = 0; i < nb_constraint; i++) {
            scalar max_dist = 0;
            int current = ids[0];
            for(int j = 0; j < ids.size(); ++j) {
                if(chosed.find(ids[j]) != chosed.end()) continue;
                scalar d =  std::numeric_limits<scalar>::max();
                for(int k = 0; k < final_ids.size(); ++k) {
                    d = std::min(d, glm::length2(geo[ids[j]] - geo[final_ids[k]]));
                }
                if(d > max_dist) {
                    current = ids[j];
                    max_dist = d;
                }
            }
            final_ids.push_back(current);
            chosed.insert(current);
        }
        cb_ids = new Cuda_Buffer<int>(final_ids);
    }


    void step(GPU_ParticleSystem *ps, scalar dt) override;
    Vector3 center;
    Cuda_Buffer<int>* cb_ids;
    scalar t;
    scalar max_dist;

    ~GPU_HighStretch() override { delete cb_ids; }
};

