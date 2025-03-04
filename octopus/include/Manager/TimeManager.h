#pragma once
#include <chrono>

#include "Core/Base.h"
#include "Core/Pattern.h"

struct TimeManager;

struct Time : Singleton<Time> {
protected:
    using Chrono = std::chrono::steady_clock::time_point;
    friend Singleton;

    Time() : _dt(0.01f), _fixed_dt(_dt), _time(0.f), _fixed_time(0.f), _frame(0) {
    }

public:
    static scalar DeltaTime() { return Time::Instance()._dt; }
    static scalar Fixed_DeltaTime() { return Time::Instance()._fixed_dt; };
    static scalar Timer() { return Time::Instance()._time; };
    static int Frame() { return Time::Instance()._frame; };
    static scalar Fixed_Timer() { return Time::Instance()._fixed_time; };

    static void Tic() { Time::Instance()._tic = std::chrono::steady_clock::now(); };

    static scalar Tac() {
        const std::chrono::duration<scalar> elapsed_seconds{std::chrono::steady_clock::now() - Time::Instance()._tic};
        return elapsed_seconds.count();
    };

    void set_delta(const scalar dt) { _dt = dt; }
    void set_fixed_deltaTime(const scalar fixed_dt) { _fixed_dt = fixed_dt; }
    void set_time(const scalar time) { _time = time; }
    void set_fixed_time(const scalar time) { _fixed_time = time; }
    void set_frame(const int nb) { _frame = nb; }

private:
    scalar _dt;
    scalar _fixed_dt;
    scalar _time;
    scalar _fixed_time;
    int _frame;
    Chrono _tic;
};

struct TimeManager : Behaviour {
    using Chrono = std::chrono::steady_clock::time_point;
    Chrono previous;
    Chrono start;

    explicit TimeManager(const scalar &fixed_dt) {
        Time::Instance().set_fixed_deltaTime(fixed_dt);
    }

    void init() override;

    void update() override;

    void enable() override {
        this->_active = true;
        previous = std::chrono::steady_clock::now();
    }

    ~TimeManager() override;
};
