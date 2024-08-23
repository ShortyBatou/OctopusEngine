#pragma once
#include "Rendering/GL_Graphic.h"
#include "Mesh/Converter/MeshConverter.h"

class GL_GraphicElement : public GL_Graphic
{
public:
    explicit GL_GraphicElement(const Color& color, scalar scale = 0.9);

    void update_gl_geometry() override;
    void update_gl_topology() override;
    void update_gl_vcolors() override;

    scalar& scale() {
        return _scale;
    }

protected:
    scalar _scale;
    std::map<Element, MeshConverter*> _converters;
};