#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <stack>
#include <cmath>
#include <chrono>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/push_relabel_max_flow.hpp>

template<typename Pixel> using Image = std::vector<Pixel>;
using InImage = Image<double>;
using OutImage = Image<unsigned short>;
using Capacity = unsigned long long;

using traits = boost::adjacency_list_traits<boost::vecS, boost::vecS, boost::directedS>;
using graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, boost::no_property,
      boost::property<boost::edge_capacity_t, Capacity,
      boost::property<boost::edge_residual_capacity_t, Capacity,
      boost::property<boost::edge_reverse_t, traits::edge_descriptor>>>>;

constexpr unsigned W = 400;
constexpr unsigned H = 400;
constexpr unsigned N = W * H;
constexpr unsigned N_IMAGES = 144;
const std::string IN_FILENAME = "../../data/predictions_flattened.csv";
const std::string OUT_BASE_DIR = "../../data/segmentations";

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

constexpr unsigned index(unsigned i, unsigned j) {
    return j*W + i;
}

inline std::string segmentation_filename(std::string name) {
    std::stringstream s;
    s << OUT_BASE_DIR << "/" << name;
    return s.str();
}

std::vector<Image<double>> load_images(const std::string& filename);
void save_images(const std::vector<OutImage>& images, const std::string& filename);
