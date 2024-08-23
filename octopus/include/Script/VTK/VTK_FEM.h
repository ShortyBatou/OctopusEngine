#pragma once
#include "Core/Component.h"
#include "Mesh/Converter/VTK_Formater.h"


class VTK_Graphic : public Component {
public:
    explicit VTK_Graphic(const std::string& name) : _name(name) { }

    void late_update() override;

    void save() const;

    std::string file_name() {
        return _name;
    }
protected:
    std::string _name;
};