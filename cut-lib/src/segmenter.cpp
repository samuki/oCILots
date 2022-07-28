#include "segmenter.hpp"
#include <fstream>


template<typename InPixel, typename Capacity, typename OutPixel>
void Segmenter<InPixel, Capacity, OutPixel>::add_st_edges(edge_adder& adder) {
    for (unsigned i = 0; i < m_W; ++i) {
        for (unsigned j = 0; j < m_H; ++j) {
            const unsigned curr = m_index(i, j);
            adder.add_edge(m_src, curr, edge_weight_s(i, j));
            adder.add_edge(curr, m_snk, edge_weight_t(i, j));
        }
    }
}

template<typename InPixel, typename Capacity, typename OutPixel>
void Segmenter<InPixel, Capacity, OutPixel>::add_neighbor_edges(edge_adder& adder) {
    // interior edges
    for (unsigned i = 0; i < m_W-1; ++i) {
        for (unsigned j = 0; j < m_H-1; ++j) {
            const unsigned curr = m_index(i, j);
            const unsigned right = m_index(i+1, j);
            const unsigned bottom = m_index(i, j+1);
            adder.add_edge(curr, right, edge_weight(i, j, i+1, j));
            adder.add_edge(curr, bottom, edge_weight(i, j, i, j+1));
        }
    }
    // bottom border edges
    for (unsigned i = 0; i < m_W-1; ++i) {
        const unsigned curr = m_index(i, m_H-1), right = m_index(i+1, m_H-1);
        adder.add_edge(curr, right, edge_weight(i, m_H-1, i+1, m_H-1));
    }
    // right border edges
    for (unsigned j = 0; j < m_H-1; ++j) {
        const unsigned curr = m_index(m_W-1, j), bottom = m_index(m_W-1, j+1);
        adder.add_edge(curr, bottom, edge_weight(m_W-1, j, m_W-1, j+1));
    }
}

template<typename InPixel, typename Capacity, typename OutPixel>
void Segmenter<InPixel, Capacity, OutPixel>::build_graph() {
    m_W = m_image.shape(0);
    m_H = m_image.shape(1);
    const unsigned N = m_W * m_H;
    m_G = graph{N+2};
    edge_adder adder{m_G};

    m_src = N;
    m_snk = N + 1;
    m_index = [this](unsigned i, unsigned j) -> int {
        return i*this->m_W + j;
    };

    add_st_edges(adder);
    add_neighbor_edges(adder);
}


template<typename InPixel, typename Capacity, typename OutPixel>
void Segmenter<InPixel, Capacity, OutPixel>::segment(
        const NDArray<InPixel>& in_image,
        NDArray<OutPixel>& out_image) {
    m_image = in_image;
    const unsigned W = m_image.shape(0);
    const unsigned H = m_image.shape(1);
    const unsigned N = W * H;

    build_graph();
    boost::push_relabel_max_flow(m_G, m_src, m_snk);

    // find cut using DFS on residual graph
    auto res_cap = boost::get(boost::edge_residual_capacity, m_G);
    std::vector<bool> visited(N+2, false);
    std::stack<unsigned> Q;
    Q.push(m_src);
    visited[m_src] = true;
    while (!Q.empty()) {
        const unsigned u = Q.top(); Q.pop();
        typename boost::graph_traits<graph>::out_edge_iterator e, eend;
        for (std::tie(e, eend) = boost::out_edges(u, m_G); e != eend; ++e) {
            const unsigned v = boost::target(*e, m_G);
            if (visited[v] || res_cap[*e] == 0) continue;
            Q.push(v);
            visited[v] = true;
        }
    }

    // set result image
    for (unsigned i = 0; i < W; ++i) {
        for (unsigned j = 0; j < H; ++j) {
            out_image(i, j) = visited[m_index(i, j)] ? 0l : 1l;
        }
    }
}
