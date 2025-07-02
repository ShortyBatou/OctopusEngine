#include "Script/VTK/VTK_Attribute.h"

#include <Core/Entity.h>
#include <Mesh/Converter/MeshLoader.h>
#include <UI/AppInfo.h>


void VTK_Attribute::late_update() {
    Mesh* mesh = _entity->get_component<Mesh>();
    if(applied) return;
    applied = true;
    VTK_Loader loader(AppInfo::PathToAssets() + _file);
    if(loader.check_vector_data(_att)) {
        const std::vector<Vector3> u_v3 = loader.get_point_data_v3(_att);
        for(int i = 0; i < mesh->geometry().size(); ++i) {
            mesh->geometry()[i] += u_v3[i];
        }
    }
}
