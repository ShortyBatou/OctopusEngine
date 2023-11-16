#pragma once
#include <cassert>
class Behaviour {
protected:
    bool _active;
public:
    Behaviour() : _active(true) {}
    virtual void init() {}
    virtual void late_init() {}
    virtual void update() {}
    virtual void late_update() { }
    inline bool active() {return _active;}
    inline void setActive(bool state) { _active = state; std::cout << "set active " << state << std::endl; }
    virtual inline void enable() { _active = true; }
    virtual inline void disable() { _active = false;}
    virtual ~Behaviour() { }
};

template<typename T>
class ID_Creator {
    static unsigned int COUNT;
    unsigned int _id; 

public:
    ID_Creator()
    {
        _id = COUNT; COUNT++;
    }
    virtual ~ID_Creator() { }
    inline unsigned int id() { return _id;}
};

template<typename T>
unsigned int ID_Creator<T>::COUNT = 0;


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
    ~Singleton() { }
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

    bool binded() { return _binded; }

protected:
    virtual void bind_action()   = 0;
    virtual void unbind_action() = 0;
    bool _binded;
};