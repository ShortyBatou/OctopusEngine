#pragma once
#include "Core/Base.h"
#include "Core/Component.h"
#include "Script/Record/Recorder.h"

class DataRecorder : public Component {
public:
	DataRecorder(std::string experiment) : _experiment(experiment), save_loop(true)
	{}

	virtual void late_init() {
		for (Recorder* recorder : _recorders) {
			recorder->init(this->entity());
		}
	}

	virtual void late_update() {
		// check if data need to be saved
		if (Input::Down(Key::S) && Input::Loop(Key::LEFT_SHIFT) || save_loop) {
			save();
		}
	}

	virtual void save() {
		for (Recorder* recorder : _recorders) {
			recorder->save();
		}

		for (Recorder* recorder : _recorders) {
			recorder->print();
		}
		std::cout << std::endl;

		std::ofstream file;
		file.open(json_path());
		file << "{";
		// save all data in file
		for (unsigned int i = 0; i < _recorders.size(); ++i) {
			file << "\"" << _recorders[i]->get_name() << "\" : ";
			_recorders[i]->add_data_json(file);
			if (i < _recorders.size() - 1) file << ",";
		}
		file << "}";
		file.close();
	}


	void add(Recorder* recorder) {
		_recorders.push_back(recorder);
	}

	std::string json_path() {
		return AppInfo::PathToAssets() + _experiment + ".json";
	};

	bool save_loop;
protected:
	unsigned int nb_save;
	std::string _experiment;
	std::vector<Recorder*> _recorders;
};

