#pragma once
#include "Core/Entity.h"
class UI_Display {
public:
	virtual void init() { }
	virtual void draw() = 0;
	virtual char* name() = 0;
};

class UI_Component {
public:
	virtual void init() { }
	virtual bool can_draw(Entity* entity) = 0;
	virtual void draw(Entity* entity) = 0;
	virtual char* name() = 0;
};
