#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <stack>
#include <cmath>
#include <chrono>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/push_relabel_max_flow.hpp>

using Capacity = unsigned long long;

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
