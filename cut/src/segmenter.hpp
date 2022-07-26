#pragma once

#include "util.hpp"
#include "ndarray.hpp"

class Segmenter {
protected:
    const unsigned m_resolution;
    NDArray<double> m_image;

    Segmenter(unsigned resolution)
        : m_resolution(resolution), m_image(nullptr, 0, nullptr, nullptr) {}

public:
    virtual std::string name() const = 0;
    void segment(const NDArray<double>& in_image, NDArray<long>& out_image);
    virtual ~Segmenter() = default;

protected:
    constexpr Capacity discretize(double x) const {
        return round(m_resolution * x);
    }

    virtual Capacity edge_weight(unsigned i1, unsigned j1, unsigned i2, unsigned j2) const = 0;
    virtual Capacity edge_weight_s(unsigned i, unsigned j) const = 0;
    virtual Capacity edge_weight_t(unsigned i, unsigned j) const = 0;
};


class RBFLogSegmenter : public Segmenter {
private:
    const double m_lambda;
    const double m_sigma;

public:
    RBFLogSegmenter(double sigma, double lambda, unsigned resolution)
        : Segmenter(resolution), m_lambda(lambda), m_sigma(sigma) {}

    std::string name() const {
        std::stringstream s;
        s << "RBFLogSegmenter(s" << m_sigma << "_l" << m_lambda << "_r" << m_resolution << ')';
        return s.str();
    }

protected:
    Capacity edge_weight(unsigned i1, unsigned j1, unsigned i2, unsigned j2) const {
        const double diff = m_image(i1, j1) - m_image(i2, j2);
        const double cap = exp(-0.5 * diff * diff / m_sigma);
        std::clog << "interior capacity " << cap << " => " << discretize(cap) << '\n';
        return discretize(cap);
    }

    Capacity edge_weight_s(unsigned i, unsigned j) const {
        const double cap = - m_lambda * log(m_image(i, j));
        std::clog << "source capacity " << cap << " => " << discretize(cap) << '\n';
        return discretize(cap);
    }

    Capacity edge_weight_t(unsigned i, unsigned j) const {
        const double cap = - m_lambda * log(1 - m_image(i, j));
        std::clog << "sink capacity " << cap << " => " << discretize(cap) << '\n';
        return discretize(cap);
    }
};
