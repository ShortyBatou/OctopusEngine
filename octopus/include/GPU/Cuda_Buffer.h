#pragma once
#include <vector>
#include <iostream>

template<typename T>
struct Cuda_Buffer {
	T* buffer;
	int nb;

	explicit Cuda_Buffer(const std::vector<T>& data);
	~Cuda_Buffer();
	void get_data(std::vector<T>& data);
	void malloc();
	void load_data(const std::vector<T>& data);
	void free() const;
};