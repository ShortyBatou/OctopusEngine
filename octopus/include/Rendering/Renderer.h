#pragma once
#include "Core/Pattern.h"
#include "Core/Component.h"
class Renderer : public Component {
public:
	virtual void draw() = 0;
	virtual void after_draw() { };
};