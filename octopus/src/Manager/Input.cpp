#pragma once
#include "Core/Base.h"
#include "manager/Input.h"


void Input::reset() {
    delete _keyboard;
    delete _mouse;
    _keyboard = new InputDevice<Key>(keys, 120);
    _mouse = new InputDevice<Mouse>(m_keys, 12);
    _first_mouse_update = true;
    _mouse_offset = Unit3D::Zero();
    _mouse_position = Unit3D::Zero();
    _mouse_prev_position = Unit3D::Zero();
    _scroll = Unit3D::Zero();
    _current_scroll = Unit3D::Zero();
    _scrolled = false;
}

void Input::update_devices() {
    _keyboard->update_inputs();
    _mouse->update_inputs();

    if (_first_mouse_update) {
        _mouse_prev_position = _mouse_position;
        _first_mouse_update = false;
    } else {
        _mouse_offset = _mouse_position - _mouse_prev_position;
        _mouse_prev_position = _mouse_position;
    }

    _current_scroll = _scroll;
    if (_scrolled) {
        _scroll = Unit3D::Zero();
        _scrolled = false;
    }
}

Input::~Input() {
    delete _keyboard;
    delete _mouse;
}
