#pragma once
#include <vector>
#include <iostream>
#include <ostream>
#include "cuda_runtime.h"

template<typename T>
struct Cuda_Buffer final
{
	T* buffer;
	int nb;

	explicit Cuda_Buffer(const std::vector<T>& data);
	explicit Cuda_Buffer(int size, const T& d_val);
	explicit Cuda_Buffer(int size);
	~Cuda_Buffer();
	void get_data(std::vector<T>& data);
	void malloc();
	void load_data(const std::vector<T>& data);
	void free() const;
};

template<typename T>
Cuda_Buffer<T>::Cuda_Buffer(const std::vector<T> &data) {
	nb = data.size();
	malloc();
	load_data(data);
}

template<typename T>
Cuda_Buffer<T>::Cuda_Buffer(const int size, const T& d_val) {
	nb = size;
	malloc();
	load_data(std::vector<T>(nb, d_val));
}

template<typename T>
Cuda_Buffer<T>::Cuda_Buffer(const int size) {
	nb = size;
	malloc();
	load_data(std::vector<T>(nb));
}

template<typename T>
Cuda_Buffer<T>::~Cuda_Buffer() {
	free();
	std::cout << "Free buffer : " << nb << std::endl;
}

template<typename T>
void Cuda_Buffer<T>::get_data(std::vector<T> &data) {
	data.resize(nb);
	cudaMemcpy(data.data(), buffer, nb * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
void Cuda_Buffer<T>::malloc() { cudaMalloc(&buffer, nb * sizeof(T)); }

template<typename T>
void Cuda_Buffer<T>::load_data(const std::vector<T> &data) {
	cudaMemcpy(buffer, data.data(), nb * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void Cuda_Buffer<T>::free() const { cudaFree(buffer); }
