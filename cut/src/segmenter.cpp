#include "segmenter.hpp"

void Segmenter::segment(const InImage& in_image, OutImage& out_image) {
        m_image = &in_image;
        graph G{N+2};
        edge_adder adder{G};

        constexpr unsigned src = N, snk = N + 1;

        // edges to source & sink
        for (unsigned i = 0; i < W; ++i) {
            for (unsigned j = 0; j < H; ++j) {
                const unsigned curr = index(i, j);
                adder.add_edge(src, curr, edge_weight_s(curr));
                adder.add_edge(curr, snk, edge_weight_t(curr));
            }
        }
        // interior edges
        for (unsigned i = 0; i < W-1; ++i) {
            for (unsigned j = 0; j < H-1; ++j) {
                const unsigned curr = index(i, j);
                const unsigned right = index(i+1, j);
                const unsigned bottom = index(i, j+1);
                adder.add_edge(curr, right, edge_weight(curr, right));
                adder.add_edge(curr, bottom, edge_weight(curr, bottom));
            }
        }
        // bottom border edges
        for (unsigned i = 0; i < W-1; ++i) {
            const unsigned curr = index(i, H-1), right = index(i+1, H-1);
            adder.add_edge(curr, right, edge_weight(curr, right));
        }
        // right border edges
        for (unsigned j = 0; j < H-1; ++j) {
            const unsigned curr = index(W-1, j), bottom = index(W-1, j+1);
            adder.add_edge(curr, bottom, edge_weight(curr, bottom));
        }

        // solve max-flow
        boost::push_relabel_max_flow(G, src, snk);

        // find cut using DFS on residual graph
        auto res_cap = boost::get(boost::edge_residual_capacity, G);
        std::vector<bool> visited(N+2, false);
        std::stack<unsigned> Q;
        Q.push(src);
        visited[src] = true;
        while (!Q.empty()) {
            const unsigned u = Q.top(); Q.pop();
            boost::graph_traits<graph>::out_edge_iterator e, eend;
            for (std::tie(e, eend) = boost::out_edges(u, G); e != eend; ++e) {
                const unsigned v = boost::target(*e, G);
                if (visited[v] || res_cap[*e] != 0) continue;
                Q.push(v);
                visited[v] = true;
            }
        }

        // set result image
        for (unsigned i = 0; i < W; ++i) {
            for (unsigned j = 0; j < H; ++j) {
                const unsigned curr = index(i, j);
                out_image[curr] = visited[curr] ? 0. : 1.;
            }
        }
    }
