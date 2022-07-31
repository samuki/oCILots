#pragma once

#include <cstddef>
#include <iostream>

template<typename T>
class NDArray {
private:
    std::byte* m_data;
    unsigned m_dims;
    const unsigned* m_shape;
    const unsigned* m_strides;

public:
    NDArray(std::byte* data, unsigned dims, const unsigned* shapes, const unsigned* strides)
        : m_data(data), m_dims(dims), m_shape(shapes), m_strides(strides) {}
    NDArray& operator=(const NDArray& a) = default;

    // element access functions for 1, 2, and 3 dimensions
    inline const T& operator()(unsigned i) const {
        return *reinterpret_cast<T*>(m_data + i*m_strides[0]);
    }
    inline T& operator()(unsigned i) {
        return *reinterpret_cast<T*>(m_data + i*m_strides[0]);
    }
    inline const T& operator()(unsigned i, unsigned j) const {
        return *reinterpret_cast<T*>(m_data + i*m_strides[0] + j*m_strides[1]);
    }
    inline T& operator()(unsigned i, unsigned j) {
        return *reinterpret_cast<T*>(m_data + i*m_strides[0] + j*m_strides[1]);
    }
    inline const T& operator()(unsigned i, unsigned j, unsigned k) const {
        return *reinterpret_cast<T*>(m_data + i*m_strides[0]+ j*m_strides[1] + k*m_strides[2]);
    }
    inline T& operator()(unsigned i, unsigned j, unsigned k) {
        return *reinterpret_cast<T*>(m_data + i*m_strides[0]+ j*m_strides[1] + k*m_strides[2]);
    }

    // getters for dimension, shape, and stride
    inline unsigned dims() const { return m_dims; }
    inline unsigned shape(unsigned d) const { return m_shape[d]; }
    inline unsigned stride(unsigned d) const { return m_strides[d]; }

    // obtain an n-1-dimensional slice of the array
    inline const NDArray<T> slice(unsigned i) const {
        return NDArray<T>(m_data + i*m_strides[0], m_dims-1, m_shape + 1, m_strides + 1);
    }
    inline NDArray<T> slice(unsigned i) {
        return NDArray<T>(m_data + i*m_strides[0], m_dims-1, m_shape + 1, m_strides + 1);
    }
};
