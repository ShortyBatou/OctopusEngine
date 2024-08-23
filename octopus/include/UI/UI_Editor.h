#pragma once
#include <Core/Entity.h>
#include "UI/UI_Primitive.h"

class UI_Editor final : public UI_Display {
public:
    UI_Editor() : _step(true) {
    }

    [[nodiscard]] std::string name() const override {
        return "Editor";
    }

    void init() override;

    void draw() override;

    void add_manager_ui(UI_Component *manager_ui) {
        _managers_ui.push_back(manager_ui);
    }

    void add_component_ui(UI_Component *component_ui) {
        _components_ui.push_back(component_ui);
    }

    void add_core_ui(UI_Display *core_ui) {
        _core_ui.push_back(core_ui);
    }

    void example();

    void clear();

    ~UI_Editor() override {
        clear();
    }

protected:
    void stop();

    void play();

    std::vector<UI_Display *> _core_ui;
    std::vector<UI_Component *> _managers_ui;
    std::vector<UI_Component *> _components_ui;
    bool _step;
    int _nb_step{};
    int _count_step{};
};
