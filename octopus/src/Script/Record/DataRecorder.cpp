#include "Script/Record/DataRecorder.h"

void DataRecorder::late_init() {
	for (Recorder* recorder : _recorders) {
		recorder->init(this->entity());
	}
}

void DataRecorder::late_update() {
	// check if data need to be saved
	if (Input::Down(Key::S) && Input::Loop(Key::LEFT_SHIFT) || save_loop) {
		save();
	}
}

void DataRecorder::save() {
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


