#include "Dynamic/PBD/XPBD_FEM_Generic.h"

#include <Manager/Input.h>

void XPBD_FEM_Generic::init(const std::vector<Particle*>& particles) {
    build(get_particles(particles));
}

bool XPBD_FEM_Generic::project(const std::vector<Particle*>& x, std::vector<Vector3>& grads, scalar& C) {

    const int nb_quadrature = static_cast<int>(_shape->weights.size());
    for (int i = 0; i < nb_quadrature; ++i) {
        // Compute transform (reference => scene)
        Matrix3x3 Jx = get_jacobian(x, _shape->dN[i]);

        // Deformation gradient (material => scene   =   material => reference => scene)
        Matrix3x3 F = Jx * _JX_inv[i];

        // Get piola kirchoff stress tensor + energy
        Matrix3x3 P;
        scalar energy;
        _pbd_material->get_pk1_and_energy(F, P, energy);

        // add forces
        P = P * glm::transpose(_JX_inv[i]) * _V[i];

        for (int j = 0; j < nb(); ++j)
            grads[j] += P * _shape->dN[i][j];

        // add energy
        C += energy * _V[i];
    }

    // convert energy to constraint
    if (std::abs(C) <= eps) return false;
    const scalar s = (C > 0) ? 1 : -1; // don't know if it's useful
    C = std::sqrt(abs(C)) * s;

    // convert force to constraint gradient
    const scalar C_inv = 0.5f / C;
    for (int j = 0; j < this->nb(); ++j) {
        grads[j] *= C_inv;
    }
    return true;
}


void XPBD_FEM_Generic_Coupled::init(const std::vector<Particle*>& particles) {
        build(get_particles(particles));
    }

// overide the XPBD solve function to work with coupled FEM constraint
void XPBD_FEM_Generic_Coupled::apply(const std::vector<Particle*>& particles, const scalar dt) {
    if (_stiffness <= 0) return;
    std::vector<Particle*> x(nb());
    for (int i = 0; i < nb(); ++i) {
        x[i] = particles[ids[i]];
    }

    // Ugly : assume there is only 2 materials (and not optimised)
    // Eval both constraint at the same time C = {C1, C2}, gradC = {gradC1, gradC2}
    scalar C1 = 0, C2 = 0;
    std::vector<Vector3> gradsC1(nb(), Unit3D::Zero());
    std::vector<Vector3> gradsC2(nb(), Unit3D::Zero());
    _pbd_material = _pbd_materials[0];
    if (!project(x, gradsC1, C1)) return;

    _pbd_material = _pbd_materials[1];
    if (!project(x, gradsC2, C2)) return;

    // solve the problem for both constraint
    const scalar a1 = 1.f / (_pbd_materials[0]->get_stiffness() * dt * dt);
    const scalar a2 = 1.f / (_pbd_materials[1]->get_stiffness() * dt * dt);
    Matrix2x2 A(a1,0,0,a2);
    for (int i = 0; i < nb(); ++i) {
        A[0][0] += glm::dot(gradsC1[i], gradsC1[i]) * x[i]->inv_mass;
        A[1][0] += glm::dot(gradsC2[i], gradsC1[i]) * x[i]->inv_mass;
        A[1][1] += glm::dot(gradsC2[i], gradsC2[i]) * x[i]->inv_mass;
    }
    A[0][1] = A[1][0]; // 2x2 symmetric matrix

    // needs a solve of 2x2 matrix : A dt_lambda = -C ==>  dt_lambda = - A^-1 C
    if(abs(glm::determinant(A)) < 1e-12) return;
    Vector2 dt_lambda = -glm::inverse(A) * Vector2(C1, C2);
    for (int i = 0; i < nb(); ++i) {
        x[i]->force += (dt_lambda[0] * gradsC1[i] + dt_lambda[1] * gradsC2[i]) * x[i]->inv_mass;
    }
}

scalar XPBD_FEM_Generic_Coupled::compute_stress(const std::vector<Vector3> &p) const {
    const size_t nb_w = _shape->weights.size();
    const size_t nb_s = _shape->nb;
    assert(p.size() >= nb_s);
    scalar stress = 0.;
    for (size_t i = 0; i < nb_w; ++i) {
        Matrix3x3 F = get_jacobian(p, _shape->dN[i]) * _JX_inv[i];
        for(int m = 0; m < 2; ++m) {
            Matrix3x3 P = _pbd_materials[m]->get_pk1(F);
            P = ContinuousMaterial::pk1_to_chauchy_stress(F, P);
            stress += ContinuousMaterial::von_mises_stress(P) * _V[i];
        }
    }
    return stress;
}