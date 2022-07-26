#pragma once

#include "util.hpp"


class Segmenter {
protected:
    const unsigned m_resolution;
    const InImage* m_image;

    Segmenter(unsigned resolution) : m_resolution(resolution), m_image(nullptr) {}

public:
    virtual std::string name() const = 0;
    void segment(const InImage& in_image, OutImage& out_image);
    virtual ~Segmenter() = default;

protected:
    constexpr Capacity discretize(double x) const {
        return round(m_resolution * x);
    }

    virtual Capacity edge_weight(int i, int j) const = 0;
    virtual Capacity edge_weight_s(int i) const = 0;
    virtual Capacity edge_weight_t(int i) const = 0;
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
    Capacity edge_weight(int i, int j) const {
        const double diff = (*m_image)[i] - (*m_image)[j];
        const double cap = exp(-0.5 * diff * diff / m_sigma);
        return discretize(cap);
    }

    Capacity edge_weight_s(int i) const {
        const double cap = - m_lambda * log((*m_image)[i]);
        return discretize(cap);
    }

    Capacity edge_weight_t(int i) const {
        const double cap = - m_lambda * log(1 - (*m_image)[i]);
        return discretize(cap);
    }
};
