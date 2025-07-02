#pragma once
#include "Core/Component.h"
#include "Mesh/Converter/VTK_Formater.h"


class VTK_Attribute : public Component {
public:
    explicit VTK_Attribute(const std::string& file, const std::string& att) : _file(file), _att(att) { applied = false; }

    void late_update() override;

protected:
    bool applied;
    std::string _file;
    std::string _att;
};