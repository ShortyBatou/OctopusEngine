#pragma once
#include "Core/Entity.h"
class UI_Display {
public:
	virtual ~UI_Display() = default;

	virtual void init() { }
	virtual void draw() = 0;
	[[nodiscard]] virtual std::string name() const = 0;
};

class UI_Component {
public:
	virtual ~UI_Component() = default;

	virtual void init() { }
	virtual bool can_draw(Entity* entity) = 0;
	virtual void draw(Entity* entity) = 0;
	[[nodiscard]] virtual std::string name() const = 0;
};
