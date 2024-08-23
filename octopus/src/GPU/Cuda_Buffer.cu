#pragma once
#include "GPU/Cuda_Buffer.h"


template<typename T>
Cuda_Buffer<T>::Cuda_Buffer(const std::vector<T> &data) {
    nb = data.size();
    malloc();
    load_data(data);
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
