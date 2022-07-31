#include "segmenter.hpp"
#include <fstream>


template<typename InPixel, typename Capacity, std::integral OutPixel>
void Segmenter<InPixel, Capacity, OutPixel>::add_st_edges() {
    edge_adder adder{m_G};
    for (unsigned i = 0; i < m_W; ++i) {
        for (unsigned j = 0; j < m_H; ++j) {
            const unsigned curr = m_index(i, j);
            adder.add_edge(m_src, curr, edge_weight_s(i, j));
            adder.add_edge(curr, m_snk, edge_weight_t(i, j));
        }
    }
}

template<typename InPixel, typename Capacity, std::integral OutPixel>
void Segmenter<InPixel, Capacity, OutPixel>::add_neighbor_edges() {
    edge_adder adder{m_G};
    // edges to bottom
    for (unsigned i = 0; i < m_W; ++i) {
        for (unsigned j = 0; j < m_H-1; ++j) {
            const unsigned curr = m_index(i, j);
            const unsigned bottom = m_index(i, j+1);
            adder.add_edge(curr, bottom, edge_weight(i, j, i, j+1));
        }
    }
    // edges to to right
    for (unsigned i = 0; i < m_W-1; ++i) {
        for (unsigned j = 0; j < m_H; ++j) {
            const unsigned curr = m_index(i, j);
            const unsigned right = m_index(i+1, j);
            adder.add_edge(curr, right, edge_weight(i, j, i+1, j));
        }
    }
    // edges to diagonal bottom right
    for (unsigned i = 0; i < m_W-1; ++i) {
        for (unsigned j = 0; j < m_H-1; ++j) {
            const unsigned curr = m_index(i, j);
            const unsigned diag = m_index(i+1, j+1);
            adder.add_edge(curr, diag, edge_weight(i, j, i+1, j+1));
        }
    }
    // edges to diagonal bottom left
    for (unsigned i = 1; i < m_W; ++i) {
        for (unsigned j = 0; j < m_H-1; ++j) {
            const unsigned curr = m_index(i, j);
            const unsigned diag = m_index(i-1, j+1);
            adder.add_edge(curr, diag, edge_weight(i, j, i-1, j+1));
        }
    }
}

template<typename InPixel, typename Capacity, std::integral OutPixel>
void Segmenter<InPixel, Capacity, OutPixel>::build_graph() {
    const unsigned N = m_W * m_H;
    m_G = graph{N+2};
    edge_adder adder{m_G};

    m_src = N;
    m_snk = N + 1;
    m_index = [this](unsigned i, unsigned j) -> int {
        return i*this->m_W + j;
    };

    add_st_edges();
    add_neighbor_edges();
}


template<typename InPixel, typename Capacity, std::integral OutPixel>
void Segmenter<InPixel, Capacity, OutPixel>::segment(
        const NDArray<InPixel>& in_image,
        NDArray<OutPixel>& out_image) {
    m_image = in_image;
    m_W = m_image.shape(0);
    m_H = m_image.shape(1);
    const unsigned N = m_W * m_H;

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
    for (unsigned i = 0; i < m_W; ++i) {
        for (unsigned j = 0; j < m_H; ++j) {
            out_image(i, j) = visited[m_index(i, j)] ? 0l : 1l;
        }
    }
}



// --------------------------------------------------------------------------------------------
//                                         Directional Edges
// --------------------------------------------------------------------------------------------

template<std::integral InPixel, typename Capacity, std::integral OutPixel>
void DirectionSegmenter<InPixel, Capacity, OutPixel>::add_direction_edges() {
    typename Segmenter<InPixel, Capacity, OutPixel>::edge_adder adder{this->m_G};
    for (unsigned i = 0; i < this->m_W; ++i) {
        for (unsigned j = 0; j < this->m_H; ++j) {
            // for each direction, count the number of white pixels in that direction
            InPixel max_n_white_pixels = 0;
            for (double theta = 0; theta < M_PI; theta += m_delta_theta) {
                InPixel n_white_pixels = 0;
                for (double d = -m_radius; d <= m_radius; ++d) {
                    const int x = i + std::round(d * std::sin(theta));
                    const int y = j + std::round(d * std::cos(theta));
                    if (x >= 0 && static_cast<unsigned>(x) < this->m_W
                            && y >= 0 && static_cast<unsigned>(y) < this->m_H)
                        n_white_pixels += this->m_image(x, y);
                }
                max_n_white_pixels = std::max(max_n_white_pixels, n_white_pixels);
            }
            // add an edge with the white sum weighted with the pixel's probability to the source
            const Capacity strength = this->discretize(m_lambda_dir * max_n_white_pixels);
            adder.add_edge(this->m_src, this->m_index(i, j), strength);
        }
    }
}

template<std::integral InPixel, typename Capacity, std::integral OutPixel>
void DirectionSegmenter<InPixel, Capacity, OutPixel>::build_graph() {
    const unsigned N = this->m_W * this->m_H;
    this->m_G = typename Segmenter<InPixel, Capacity, OutPixel>::graph{N+2};
    typename Segmenter<InPixel, Capacity, OutPixel>::edge_adder adder{this->m_G};

    this->m_src = N;
    this->m_snk = N + 1;
    this->m_index = [this](unsigned i, unsigned j) -> int {
        return i*this->m_W + j;
    };

    this->add_st_edges();
    add_direction_edges();
}
