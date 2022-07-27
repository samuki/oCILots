#pragma once

#include "ndarray.hpp"
#include <cmath>
#include <vector>
#include <stack>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/push_relabel_max_flow.hpp>


template<typename InPixel, typename Capacity, typename OutPixel>
class Segmenter {
protected:
    const unsigned m_resolution;
    NDArray<InPixel> m_image;

    Segmenter(unsigned resolution)
        : m_resolution(resolution), m_image(nullptr, 0, nullptr, nullptr) {}

public:
    virtual std::string name() const = 0;
    void segment(const NDArray<InPixel>& in_image, NDArray<OutPixel>& out_image);
    virtual ~Segmenter() = default;

protected:
    constexpr Capacity discretize(InPixel x) const {
        return round(m_resolution * x);
    }

    virtual Capacity edge_weight(unsigned i1, unsigned j1, unsigned i2, unsigned j2) const = 0;
    virtual Capacity edge_weight_s(unsigned i, unsigned j) const = 0;
    virtual Capacity edge_weight_t(unsigned i, unsigned j) const = 0;

private:
    using traits = boost::adjacency_list_traits<boost::vecS, boost::vecS, boost::directedS>;
    using graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, boost::no_property,
          boost::property<boost::edge_capacity_t, Capacity,
          boost::property<boost::edge_residual_capacity_t, Capacity,
          boost::property<boost::edge_reverse_t, traits::edge_descriptor>>>>;

    class edge_adder {
        graph &G;
    public:
        explicit edge_adder(graph &G) : G(G) {}

        void add_edge(int from, int to, Capacity capacity) {
            auto c_map = boost::get(boost::edge_capacity, G);
            auto r_map = boost::get(boost::edge_reverse, G);
            const auto e = boost::add_edge(from, to, G).first;
            const auto rev_e = boost::add_edge(to, from, G).first;
            c_map[e] = capacity;
            c_map[rev_e] = 0; // reverse edge has no capacity!
            r_map[e] = rev_e;
            r_map[rev_e] = e;
        }
    };
};


template<typename InPixel, typename Capacity, typename OutPixel>
class RBFLogSegmenter : public Segmenter<InPixel, Capacity, OutPixel> {
private:
    const InPixel m_lambda;
    const InPixel m_sigma;

public:
    RBFLogSegmenter(InPixel sigma, InPixel lambda, unsigned resolution)
        : Segmenter<InPixel, Capacity, OutPixel>(resolution), m_lambda(lambda), m_sigma(sigma) {}

    std::string name() const {
        std::stringstream s;
        s << "RBFLogSegmenter(s" << m_sigma << "_l" << m_lambda
            << "_r" << this->m_resolution << ')';
        return s.str();
    }

protected:
    Capacity edge_weight(unsigned i1, unsigned j1, unsigned i2, unsigned j2) const {
        const InPixel diff = this->m_image(i1, j1) - this->m_image(i2, j2);
        const InPixel cap = exp(-0.5 * diff * diff / m_sigma);
        return this->discretize(cap);
    }

    Capacity edge_weight_s(unsigned i, unsigned j) const {
        const InPixel cap = - m_lambda * log(this->m_image(i, j));
        return this->discretize(cap);
    }

    Capacity edge_weight_t(unsigned i, unsigned j) const {
        const InPixel cap = - m_lambda * log(1 - this->m_image(i, j));
        return this->discretize(cap);
    }
};


template class Segmenter<float, unsigned long long, unsigned>;
