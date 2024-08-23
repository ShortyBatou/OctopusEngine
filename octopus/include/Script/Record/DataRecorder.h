#pragma once

#include "Core/Component.h"
#include "Script/Record/Recorder.h"
#include "UI/AppInfo.h"


class DataRecorder : public Component {
public:
    explicit DataRecorder(const std::string &experiment, bool loop = false)
        : _experiment(experiment), save_loop(loop), nb_save(0) {
    }

    void late_init() override;

    void late_update() override;

    virtual void save();

    void add(Recorder *recorder) {
        _recorders.push_back(recorder);
    }

    std::string json_path() {
        return AppInfo::PathToAssets() + _experiment + ".json";
    };

    bool save_loop;

protected:
    unsigned int nb_save;
    std::string _experiment;
    std::vector<Recorder *> _recorders;
};
