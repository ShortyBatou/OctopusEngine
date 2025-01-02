#pragma once
#include "Core/Base.h"
#include "Core/Component.h"
#include "Mesh/Mesh.h"
#include "Core/Entity.h"
#include "Rendering/GL_Graphic.h"
#include "Script/Dynamic/FEM_Dynamic.h"

struct FEM_DataDisplay : public Component {
    enum Type {
        BaseColor, Displacement, Stress, V_Stress, Volume, Volume_Diff, Mass, Inv_Mass, Velocity, None
    };

    explicit FEM_DataDisplay(const Type mode = BaseColor, const ColorMap::Type type = ColorMap::Type::Default)
        : color_map(type), _mode(mode), _mesh(nullptr), _graphic(nullptr), _ps_dynamic(nullptr), _fem_dynamic(nullptr) {}

    static std::string Type_To_Str(Type mode);

    void init() override;

    static std::vector<Color> convert_to_color(const std::vector<scalar>& data);

    void update() override;


    ColorMap::Type color_map;
    Type _mode;
private:
    Mesh* _mesh;
    GL_Graphic* _graphic;
    ParticleSystemDynamics_Getters* _ps_dynamic;
    FEM_Dynamic_Getters* _fem_dynamic;
};