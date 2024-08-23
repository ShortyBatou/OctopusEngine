#pragma once
#include "Manager/InputDevice.h"

template<class K>
InputDevice<K>::InputDevice(const K INPUTS[], unsigned int nb_inputs) {
    for (unsigned int i = 0; i < nb_inputs; ++i)
        inputs.push_back(INPUTS[i]);
    reset_inputs();
}

template<class K>
void InputDevice<K>::reset_inputs() {
    for (K input: inputs) {
        down[input] = False;
        up[input] = False;
        pressed[input] = False;
    }
}

template<class K>
void InputDevice<K>::update_inputs() {
    for (K input: inputs) {
        if (down[input] == Wait) {
            down[input] = True;
            up[input] = False;
        } else if (down[input] == True) {
            down[input] = Lock; // wait until key up event = true
        }

        if (up[input] == Wait) {
            up[input] = True;
            down[input] = False;
        } else if (up[input] == True) {
            up[input] = Lock; // wait until key down event = true
        }
    }
}
