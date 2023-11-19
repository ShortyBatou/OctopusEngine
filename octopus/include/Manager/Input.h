#pragma once
#include "Core/Pattern.h"
#include "Manager/InputDevice.h"
#include "Key.h"
#include <map>


class InputManager;
struct Input : public Singleton<Input>
{
    static bool Exist(Key key) { return Instance().keyboard().exist(key); }
    static bool Down(Key key) { return Instance().keyboard().get_down(key); }
    static bool Up(Key key) { return Instance().keyboard().get_up(key); }
    static bool Loop(Key key) { return Instance().keyboard().get(key); }

    static bool StateDown(Key key) { return Instance().keyboard().state_down(key); }
    static bool StateUp(Key key) { return Instance().keyboard().state_up(key); }
    static bool StateLoop(Key key) { return Instance().keyboard().state_loop(key); }

    static bool Exist(Mouse button) { return Instance().mouse().exist(button); }
    static bool Down(Mouse button) { return Instance().mouse().get_down(button); }
    static bool Up(Mouse button) { return Instance().mouse().get_up(button); }
    static bool Loop(Mouse button) { return Instance().mouse().get(button); }

    static bool StateDown(Mouse button) { return Instance().mouse().state_down(button); }
    static bool StateUp(Mouse button) { return Instance().mouse().state_up(button); }
    static bool StateLoop(Mouse button) { return Instance().mouse().state_loop(button); }

    static scalar MouseScroll() { return Instance()._current_scroll.y; } 
    static Vector2 MousePosition() { return Instance()._mouse_position; }
    static Vector2 MouseOffset() { return Instance()._mouse_offset; }

    void set_mouse_position(const Vector2& position) { _mouse_position = position; }

    void set_scroll(const Vector2& scroll) { 
        _scrolled = true;
        _scroll = scroll; 
    }

    InputDevice<Key>& keyboard() { return *_keyboard; }
    InputDevice<Mouse>& mouse() { return *_mouse; }

    void reset() {
        delete _keyboard;
        delete _mouse;
        _keyboard            = new InputDevice<Key>(keys, 120);
        _mouse               = new InputDevice<Mouse>(m_keys, 12);
        _first_mouse_update = true;
        _mouse_offset        = Unit3D::Zero();
        _mouse_position      = Unit3D::Zero();
        _mouse_prev_position = Unit3D::Zero();
        _scroll              = Unit3D::Zero();
        _current_scroll      = Unit3D::Zero();
        _scrolled            = false;
    }

    void update_devices() { 
        _keyboard->update_inputs();
        _mouse->update_inputs();

        if (_first_mouse_update)
        {
            _mouse_prev_position = _mouse_position;
            _first_mouse_update  = false;
        }
        else
        {
            _mouse_offset        = _mouse_position - _mouse_prev_position;
            _mouse_prev_position = _mouse_position;
        }

        _current_scroll = _scroll;
        if (_scrolled)
        {
            _scroll   = Unit3D::Zero();
            _scrolled = false;
        }
    }

    virtual ~Input() {
        delete _keyboard;
        delete _mouse;
    }

protected: 
    friend InputManager;
    friend Singleton<Input>;
    
    InputDevice<Key>* _keyboard;
    InputDevice<Mouse>* _mouse;

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