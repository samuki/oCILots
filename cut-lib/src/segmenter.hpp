#pragma once

#include "ndarray.hpp"
#include <concepts>
#include <cmath>
#include <vector>
#include <stack>
#include <functional>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/push_relabel_max_flow.hpp>


template<typename InPixel, typename Capacity, std::integral OutPixel>
class Segmenter {
protected:
    using traits = boost::adjacency_list_traits<boost::vecS, boost::vecS, boost::directedS>;
    using graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS,
          boost::no_property,
          boost::property<boost::edge_capacity_t, Capacity,
          boost::property<boost::edge_residual_capacity_t, Capacity,
          boost::property<boost::edge_reverse_t, traits::edge_descriptor>>>>;

    const unsigned m_resolution;
    NDArray<InPixel> m_image;
    unsigned m_W, m_H;

    graph m_G;
    unsigned m_src, m_snk;
    std::function<unsigned(unsigned, unsigned)> m_index;

    Segmenter(unsigned resolution)
        : m_resolution(resolution), m_image(nullptr, 0, nullptr, nullptr) {}

public:
    virtual std::string name() const = 0;
    virtual void segment(const NDArray<InPixel>& in_image, NDArray<OutPixel>& out_image);
    virtual ~Segmenter() = default;

protected:
    constexpr Capacity discretize(InPixel x) const {
        return round(m_resolution * x);
    }

    virtual void add_st_edges();
    virtual void add_neighbor_edges();

    virtual void build_graph();

    virtual Capacity edge_weight(unsigned i1, unsigned j1, unsigned i2, unsigned j2) const = 0;
    virtual Capacity edge_weight_s(unsigned i, unsigned j) const = 0;
    virtual Capacity edge_weight_t(unsigned i, unsigned j) const = 0;

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


template<typename InPixel, typename Capacity, std::integral OutPixel>
class RBFLogSegmenter : public Segmenter<InPixel, Capacity, OutPixel> {
private:
    const InPixel m_lambda;
    const InPixel m_sigma;

public:
    RBFLogSegmenter(InPixel sigma, InPixel lambda, unsigned resolution)
        : Segmenter<InPixel, Capacity, OutPixel>(resolution), m_lambda(lambda), m_sigma(sigma) {}

    virtual std::string name() const {
        std::stringstream s;
        s << "RBFLogSegmenter(s" << m_sigma << "_l" << m_lambda
            << "_r" << this->m_resolution << ')';
        return s.str();
    }

protected:
    virtual Capacity edge_weight(unsigned i1, unsigned j1, unsigned i2, unsigned j2) const {
        const double dx = i1 - i2, dy = j1 - j2;
        const double dist = std::sqrt(dx*dx + dy*dy);
        const InPixel diff = this->m_image(i1, j1) - this->m_image(i2, j2);
        const InPixel cap = exp(-0.5 * diff * diff / m_sigma);
        return this->discretize(cap / dist);
    }

    virtual Capacity edge_weight_s(unsigned i, unsigned j) const {
        InPixel val = this->m_image(i, j);
        val = std::min(val, static_cast<InPixel>(0.999));
        const InPixel cap = - m_lambda * log(val);
        return this->discretize(cap);
    }

    virtual Capacity edge_weight_t(unsigned i, unsigned j) const {
        InPixel val = this->m_image(i, j);
        val = std::max(val, static_cast<InPixel>(0.001));
        const InPixel cap = - m_lambda * log(1 - val);
        return this->discretize(cap);
    }
};


template<std::integral InPixel, typename Capacity, std::integral OutPixel>
class DirectionSegmenter : public Segmenter<InPixel, Capacity, OutPixel> {
private:
    double m_lambda_pred;
    double m_lambda_dir;
    int m_radius;
    double m_delta_theta;

public:
    DirectionSegmenter(
            InPixel lambda_pred,
            InPixel lambda_dir,
            unsigned radius,
            double delta_theta,
            unsigned resolution)
        : Segmenter<InPixel, Capacity, OutPixel>(resolution),
          m_lambda_pred(lambda_pred),
          m_lambda_dir(lambda_dir),
          m_radius(radius),
          m_delta_theta(delta_theta) {}

    virtual std::string name() const {
        std::stringstream s;
        s << "DirectionSegmenter(pred=" << m_lambda_pred
            << ", dir=" << m_lambda_dir << ", r=" << m_radius << ", dt=" << m_delta_theta << ')';
        return s.str();
    }

    virtual Capacity edge_weight(unsigned i1, unsigned j1, unsigned i2, unsigned j2) const {
        // to silence unused warning, which is intentional here
        (void) i1; (void) j1; (void) i2; (void) j2;
        return 0;
    }

    virtual Capacity edge_weight_s(unsigned i, unsigned j) const {
        return m_lambda_pred * this->m_image(i, j);
    }

    virtual Capacity edge_weight_t(unsigned i, unsigned j) const {
        return m_lambda_pred * (1 - this->m_image(i, j));
    }

    virtual void add_direction_edges();
    virtual void build_graph();
};


template class RBFLogSegmenter<float, unsigned long long, unsigned>;
template class RBFLogSegmenter<double, unsigned long long, unsigned>;

template class DirectionSegmenter<unsigned, unsigned long long, unsigned>;
