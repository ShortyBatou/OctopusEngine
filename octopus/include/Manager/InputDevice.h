#pragma once
#include "iostream"
#include <vector>
enum key_state
{
    False,  // unactive, free to use
    True,   // active this frame
    Wait,   // wait next frame
    Lock    // lock until event
};

template<typename K>
struct InputDevice
{
    std::map<K, key_state> up;
    std::map<K, key_state> pressed;
    std::map<K, key_state> down;
    std::vector<K> inputs;

    InputDevice(const K INPUTS[], unsigned int nb_inputs)
    {
        for (unsigned int i = 0; i < nb_inputs; ++i)
            inputs.push_back(INPUTS[i]);
        reset_inputs();
    }

    bool exist(K input) {
        return down.find(input) != down.end();
    }

    bool get_down(K input) { return down[input] == True;}
    bool get_up(K input) { return up[input] == True; }
    bool get(K input) { return pressed[input] == True; }

    key_state state_down(K input) { return down[input]; }
    key_state state_up(K input) { return up[input]; }
    key_state state_loop(K input) { return pressed[input]; }

    void set_pressed(K input)
    {
        down[input]    = (down[input] == False) ? Wait : down[input];
        pressed[input] = True;
    }

    void set_released(K input)
    {
        up[input]      = (up[input] == False) ? Wait : up[input];
        pressed[input] = False;
    }

    void reset_inputs()
    {
        for (K input : inputs)
        {
            down[input]    = False;
            up[input]      = False;
            pressed[input] = False;
        }
    }

    void update_inputs()
    {
        for (K input : inputs)
        {
            if (down[input] == Wait)
            {
                down[input] = True;
                up[input]   = False;
            }
            else if (down[input] == True)
            {
                down[input] = Lock;  // wait until key up event = true
            }

            if (up[input] == Wait)
            {
                up[input]   = True;
                down[input] = False;
            }
            else if (up[input] == True) 
            {
                up[input] = Lock;  // wait until key down event = true
            }
        }
    }
};