#pragma once
#include "Core/Base.h"
#include "Core/Pattern.h"
#include "Manager/InputDevice.h"
#include "Key.h"
#include <map>


class InputManager;

struct Input final : Singleton<Input> {
    static bool Exist(const Key &key) { return Instance().keyboard().exist(key); }
    static bool Down(const Key &key) { return Instance().keyboard().get_down(key); }
    static bool Up(const Key &key) { return Instance().keyboard().get_up(key); }
    static bool Loop(const Key &key) { return Instance().keyboard().get(key); }

    static bool StateDown(const Key &key) { return Instance().keyboard().state_down(key); }
    static bool StateUp(const Key &key) { return Instance().keyboard().state_up(key); }
    static bool StateLoop(const Key &key) { return Instance().keyboard().state_loop(key); }

    static bool Exist(const Mouse &button) { return Instance().mouse().exist(button); }
    static bool Down(const Mouse &button) { return Instance().mouse().get_down(button); }
    static bool Up(const Mouse &button) { return Instance().mouse().get_up(button); }
    static bool Loop(const Mouse &button) { return Instance().mouse().get(button); }

    static bool StateDown(const Mouse &button) { return Instance().mouse().state_down(button); }
    static bool StateUp(const Mouse &button) { return Instance().mouse().state_up(button); }
    static bool StateLoop(const Mouse &button) { return Instance().mouse().state_loop(button); }

    static scalar MouseScroll() { return Instance()._current_scroll.y; }
    static Vector2 MousePosition() { return Instance()._mouse_position; }
    static Vector2 MouseOffset() { return Instance()._mouse_offset; }

    void set_mouse_position(const Vector2 &position) { _mouse_position = position; }

    void set_scroll(const Vector2 &scroll) {
        _scrolled = true;
        _scroll = scroll;
    }

    [[nodiscard]] InputDevice<Key> &keyboard() const { return *_keyboard; }
    [[nodiscard]] InputDevice<Mouse> &mouse() const { return *_mouse; }

    void reset();

    void update_devices();

    ~Input();

protected:
    friend InputManager;
    friend Singleton;

    InputDevice<Key> *_keyboard;
    InputDevice<Mouse> *_mouse;

    bool _first_mouse_update;
    Vector2 _mouse_offset;
    Vector2 _mouse_prev_position;
    Vector2 _mouse_position;

    bool _scrolled;
    Vector2 _scroll;
    Vector2 _current_scroll;

    Input() : _keyboard(nullptr), _mouse(nullptr) {
        reset();
    }
};
