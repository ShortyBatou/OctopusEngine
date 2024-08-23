#pragma once
#include <chrono>
#include "Manager/TimeManager.h"

void TimeManager::init() {
    Time::Instance().set_delta(Time::Fixed_DeltaTime());
    previous = std::chrono::steady_clock::now();
    start = std::chrono::steady_clock::now();
}

void TimeManager::update() {
    const Chrono current = std::chrono::steady_clock::now();
    const std::chrono::duration<scalar> elapsed_seconds{current - previous};
    Time::Instance().set_delta(elapsed_seconds.count());
    Time::Instance().set_time(Time::Timer() + elapsed_seconds.count());
    Time::Instance().set_fixed_time(Time::Fixed_Timer() + Time::Fixed_DeltaTime());
    Time::Instance().set_frame(Time::Frame() + 1);
    previous = current;
}

TimeManager::~TimeManager() {
    Time::Delete();
}
