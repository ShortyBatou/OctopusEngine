#pragma once
#include <chrono>

#include "Core/Base.h"
#include "Core/Pattern.h"

class TimeManager;

struct Time : Singleton<Time> {
protected:
    using Chrono = typename std::chrono::steady_clock::time_point;
    friend Singleton<Time>;
    Time() : _dt(0.01), _fixed_dt(_dt), _time(0.), _fixed_time(0.), _frame(0) { }

public:
    static scalar DeltaTime() {return Time::Instance()._dt;}
    static scalar Fixed_DeltaTime() {return Time::Instance()._fixed_dt;};
    static scalar Timer() { return Time::Instance()._time; };
    static int Frame() { return Time::Instance()._frame; };
    static scalar Fixed_Timer() { return Time::Instance()._fixed_time; };

    static void Tic() { Time::Instance()._tic = std::chrono::steady_clock::now();};
    static scalar Tac() { 
        const std::chrono::duration<scalar> elapsed_seconds{ std::chrono::steady_clock::now() - Time::Instance()._tic };
        return scalar(elapsed_seconds.count());
    };

    void set_delta(scalar dt) { _dt = dt;}
    void set_fixed_deltaTime(scalar fixed_dt) {_fixed_dt = fixed_dt;}
    void set_time(scalar time) { _time = time; }
    void set_fixed_time(scalar time) { _fixed_time = time; }
    void set_frame(unsigned int nb) { _frame = nb; }
private:
    scalar _dt;
    scalar _fixed_dt;
    scalar  _time;
    scalar _fixed_time;
    int _frame;
    Chrono _tic;
};

struct TimeManager : Behaviour {
    using Chrono = typename std::chrono::steady_clock::time_point;
    Chrono previous;
    Chrono start;
    
    TimeManager(scalar fixed_dt) {
        
        Time::Instance().set_fixed_deltaTime(fixed_dt);
    }
    
    void init() {
        Time::Instance().set_delta(Time::Fixed_DeltaTime());
        previous = std::chrono::steady_clock::now();
        start    = std::chrono::steady_clock::now();
    }

    void update() {
        Chrono current = std::chrono::steady_clock::now();
        const std::chrono::duration<scalar> elapsed_seconds{current - previous};
        Time::Instance().set_delta( elapsed_seconds.count() );
        Time::Instance().set_time(Time::Timer() + scalar(elapsed_seconds.count()));
        Time::Instance().set_fixed_time(Time::Fixed_Timer() + Time::Fixed_DeltaTime());
        Time::Instance().set_frame(Time::Frame()+1);
        previous = current;
    }

    virtual inline void enable() { 
        this->_active = true; 
        previous = std::chrono::steady_clock::now();
    }

    virtual ~TimeManager() {
        Time::Delete();
    }
};