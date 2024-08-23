#include "Dynamic/PBD/XPBD_FEM_Generic.h"

void XPBD_FEM_Generic::init(const std::vector<Particle*>& particles) {
    build(get_particles(particles));
}

bool XPBD_FEM_Generic::project(const std::vector<Particle*>& x, std::vector<Vector3>& grads, scalar& C) {
    Matrix3x3 Jx, F, P;
    scalar energy;
    int nb_quadrature = static_cast<int>(_shape->weights.size());
    for (int i = 0; i < nb_quadrature; ++i) {
        // Compute transform (reference => scene)
        Jx = get_jacobian(x, _shape->dN[i]);

        // Deformation gradient (material => scene   =   material => reference => scene)
        F = Jx * _JX_inv[i];

        // Get piola kirchoff stress tensor + energy
        _pbd_material->get_pk1_and_energy(F, P, energy);

        // add forces
        P = P * glm::transpose(_JX_inv[i]) * _V[i];
        for (int j = 0; j < this->nb(); ++j)
            grads[j] += P * _shape->dN[i][j];

        // add energy
        C += energy * _V[i];
    }

    // convert energy to constraint
    if (std::abs(C) <= eps) return false;
    scalar s = (C > 0) ? 1 : -1; // don't know if it's useful
    C = sqrt(abs(C)) * s;

    // convert force to constraint gradient
    scalar C_inv = 1.f / 2.f * C;
    for (int j = 0; j < this->nb(); ++j) {
        grads[j] *= C_inv;
    }
    return true;
}
