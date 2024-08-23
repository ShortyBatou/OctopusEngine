#pragma once
#include <map>
#include <vector>

enum key_state {
    False, // unactive, free to use
    True, // active this frame
    Wait, // wait next frame
    Lock // lock until event
};

template<class K>
struct InputDevice {
    std::map<K, key_state> up;
    std::map<K, key_state> pressed;
    std::map<K, key_state> down;
    std::vector<K> inputs;

    InputDevice::InputDevice(const K INPUTS[], unsigned int nb_inputs);

    bool exist(K input) {
        return down.find(input) != down.end();
    }

    bool get_down(K input) { return down[input] == True; }
    bool get_up(K input) { return up[input] == True; }
    bool get(K input) { return pressed[input] == True; }

    key_state state_down(K input) { return down[input]; }
    key_state state_up(K input) { return up[input]; }
    key_state state_loop(K input) { return pressed[input]; }

    void set_pressed(K input) {
        down[input] = (down[input] == False) ? Wait : down[input];
        pressed[input] = True;
    }

    void set_released(K input) {
        up[input] = (up[input] == False) ? Wait : up[input];
        pressed[input] = False;
    }

    void reset_inputs();

    void update_inputs();
};
