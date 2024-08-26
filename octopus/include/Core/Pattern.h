#pragma once
#include <cassert>
#include <iostream>

class Behaviour {
protected:
    bool _active;
public:
    Behaviour() : _active(true) {}
    virtual void init() {}
    virtual void late_init() {}
    virtual void update() {}
    virtual void late_update() { }
    [[nodiscard]] bool active() const {return _active;}
    void setActive(const bool state) { _active = state; std::cout << "set active " << state << std::endl; }
    virtual void enable() { _active = true; }
    virtual void disable() { _active = false;}
    virtual ~Behaviour() = default;
};


template<typename T>
class Singleton {
    static T* _instance;
    static T* Init()
    {
        _instance = new T();
        return _instance;
    };

protected:
    Singleton() { }
    ~Singleton() = default;
public:
    
    static T& Instance() {
        if (_instance == nullptr) Init();
        return *_instance;
    }

    static T* Instance_ptr()
    {
        if (_instance == nullptr) Init();
        return _instance;
    }

    static void Delete() {
        delete _instance;
        _instance = nullptr;
    }
};

template <typename T>
T* Singleton<T>::_instance = nullptr;

struct UniqueBinder
{
    virtual ~UniqueBinder() = default;

    UniqueBinder() : _binded(false) { }
    void bind() {
        assert(!_binded);
        bind_action();
        _binded = true;
    }
    
    void unbind() { 
        assert(_binded);
        unbind_action();
        _binded = false;
    }

    virtual bool binded() { return _binded; }

protected:
    virtual void bind_action()   = 0;
    virtual void unbind_action() = 0;
    bool _binded;
};