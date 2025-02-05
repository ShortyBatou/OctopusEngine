#pragma once
#include "Core/Base.h"
#include <vector>
#include <set>
#include <algorithm>
#include <numeric>
#include <queue>

struct Graph {
    Graph(const int nb_vert_elem, const Mesh::Topology& topology, const bool primal = true) {
        if(primal) build_primal(nb_vert_elem, topology);
        else build_dual(nb_vert_elem, topology);
    }

    explicit Graph(const std::vector<std::set<int>>& adjacence) : n(adjacence.size()), adj(adjacence) {
        degree.resize(n);
        for(int i = 0; i < adj.size(); ++i) {
            degree[i] = adj[i].size();
        }
    }

    void build_primal(const int nb_vert_elem, const Mesh::Topology& topology) {
        n = *std::max_element(topology.begin(), topology.end())+1;
        degree.resize(n);
        adj.resize(n);
        for (int i = 0; i < topology.size(); i += nb_vert_elem) {
            for (int j = 0; j < nb_vert_elem; ++j) {
                for (int k = 0; k < nb_vert_elem; ++k) {
                    if (k == j) continue;
                    const int id = topology[i + j];
                    adj[id].insert(topology[i + k]);
                }
            }
        }

        for(int i = 0; i < adj.size(); ++i) {
            degree[i] = adj[i].size();
        }
    }

    void build_dual(const int nb_vert_elem, const Mesh::Topology& topology) {
        n = topology.size() / nb_vert_elem; // nb_element
        degree.resize(n);
        adj.resize(n);
        const Element elem = get_elem_by_size(nb_vert_elem);
        Element lin_elem = get_linear_element(elem);
        int lin_nb_vert_elem = elem_nb_vertices(elem);

        Mesh::Topology r_tri = ref_triangles(lin_elem);
        Mesh::Topology r_quads = ref_quads(lin_elem);

        // we need to find pairs of faces (triangle or quads)
        std::map<Face<4>, int> quads;
        std::map<Face<3>, int> triangles;

        for(int eid = 0; eid < n; ++eid) {
            int i = eid * nb_vert_elem;
            for(int j = 0; j < r_tri.size(); j+= 3) {
                Face<3> tri({topology[i + r_tri[j]], topology[i + r_tri[j+1]], topology[i + r_tri[j+2]]});
                if(auto it = triangles.find(tri); it != triangles.end()) {
                    int eid2 = triangles[tri];
                    adj[eid].insert(eid2);
                    adj[eid2].insert(eid);
                    triangles.erase(it);
                }
                else {
                    triangles[tri] = eid;
                }
            }

            for(int j = 0; j < r_quads.size(); j+= 4) {
                Face<4> quad({topology[i + r_quads[j]], topology[i + r_quads[j+1]], topology[i + r_quads[j+2]], topology[i + r_quads[j+3]]});
                if(auto it = quads.find(quad); it != quads.end()) {
                    int eid2 = quads[quad];
                    adj[eid].insert(eid2);
                    adj[eid2].insert(eid);
                    quads.erase(it);
                }
                else {
                    quads[quad] = eid;
                }
            }
        }

        for(int i = 0; i < adj.size(); ++i) {
            degree[i] = adj[i].size();
        }
    }

    int n;
    std::vector<int> degree;
    std::vector<std::set<int>> adj;
};

struct Coloration {
    int nb_color;
    std::vector<int> color;
};

class GraphColoration {
public:
    static Coloration Greedy(const Graph& graph) {
        int nb_color = 0;
        std::vector<int> colors(graph.n, -1);
        for(int i = 0; i < graph.n; ++i) {
            find_color_greedy(graph, i, nb_color, colors);
        }
        return {nb_color, colors};
    }

    static Coloration Greedy_LF(const Graph& graph) {
        int nb_color = 0;
        std::vector<int> colors(graph.n, -1);
        std::vector<int> ids(graph.n, 0);
        std::iota(ids.begin(), ids.end(), 0);
        std::sort(ids.begin(), ids.end(),
         [&](const int& a, const int& b) {
                return graph.degree[a] >= graph.degree[b];
            }
        );

        for(const int i : ids) {
            find_color_greedy(graph, i, nb_color, colors);
        }

        return {nb_color, colors};
    }

    static Coloration Greedy_SLF(const Graph& graph) {
        int nb_color = 0;
        int n = graph.n;

        // sort vertices by decreasing degree
        using P_Vertex = std::pair<int, int>; // degree / id
        std::priority_queue<P_Vertex, std::vector<P_Vertex>, std::greater<>> min_degree;
        for(int i = 0; i < graph.n; ++i) {
            min_degree.push({graph.degree[i], i});
        }

        std::vector<int> ids(graph.n, 0);
        std::vector<bool> removed(n, false);
        std::vector<int> degree = graph.degree;
        for (int i = n - 1; i >= 0; i--) {
            // remove all vertices that have been visited
            while (!min_degree.empty() && removed[min_degree.top().second]) {
                min_degree.pop();
            }

            const int v = min_degree.top().second;
            min_degree.pop();
            removed[v] = true;
            ids[i] = v;

            //
            for (int neighbor : graph.adj[v]) {
                if (!removed[neighbor]) {
                    degree[neighbor]--;
                    min_degree.push({degree[neighbor], neighbor});
                }
            }
        }

        std::vector<int> colors(graph.n, -1);
        for(const int i : ids) {
            find_color_greedy(graph, i, nb_color, colors);
        }

        return {nb_color, colors};
    }

    static Coloration Greedy_RLF(const Graph& graph) {
        std::vector<bool> uncolored(graph.n, true);
        std::vector<int> colors(graph.n, -1);
        int nb_color = 0;

        while (count(uncolored.begin(), uncolored.end(), true) > 0) {
            int maxDegree = -1, v = -1;
            for (int i = 0; i < graph.n; i++) {
                if (uncolored[i]) {
                    if (graph.degree[i] > maxDegree) {
                        maxDegree = graph.degree[i];
                        v = i;
                    }
                }
            }

            if (v == -1) break;

            std::vector<int> independentSet;
            independentSet.push_back(v);
            uncolored[v] = false;

            std::set<int> neighbors(graph.adj[v].begin(), graph.adj[v].end());

            for (int i = 0; i < graph.n; i++) {
                if (uncolored[i] && neighbors.find(i) == neighbors.end()) {
                    independentSet.push_back(i);
                    uncolored[i] = false;
                    for (int neighbor : graph.adj[i]) {
                        neighbors.insert(neighbor);
                    }
                }
            }

            for (const int u : independentSet) {
                colors[u] = nb_color;
            }
            nb_color++;
        }
        return {nb_color, colors};
    }

    static Coloration DSAT(const Graph& graph) {
        int vid, i;
        std::vector<bool> used(graph.n, false);
        std::vector<int> colors(graph.n, -1);
        std::vector<int> degree(graph.degree);
        std::vector<std::set<int> > adjCols(graph.n);
        std::set<Node, maxSat> max_queue; // sort vertices depending on their saturation > degree > id

        // generate nodes
        for (vid = 0; vid < graph.n; vid++) {
            max_queue.emplace(Node{ 0, degree[vid], vid });
        }

        // while there is vertices to color
        while (!max_queue.empty()) {
            // get and pop the max saturation vertex
            const auto maxPtr = max_queue.begin();
            vid = maxPtr->vertex;
            max_queue.erase(maxPtr);

            // get used color around vertex
            for (const int v : graph.adj[vid])
                if (colors[v] != -1)
                    used[colors[v]] = true;

            // get the first color i not used
            for (i = 0; i < used.size(); i++)
                if (!used[i]) break;

            //
            for (const int v : graph.adj[vid])
                if (colors[v] != -1)
                    used[colors[v]] = false;

            // color vertex
            colors[vid] = i;

            // change saturation of vertices around current vertex
            for (const int v : graph.adj[vid]) {
                if (colors[v] == -1) {
                    max_queue.erase(
                        { static_cast<int>(adjCols[v].size()), degree[v], v });
                    adjCols[v].insert(i);
                    degree[v]--;
                    max_queue.emplace(Node{ static_cast<int>(adjCols[v].size()),degree[v], v });
                }
            }
        }

        return { *std::max_element(colors.begin(), colors.end())+1, colors};
    }
private:
    struct Node {
        int sat;
        int deg;
        int vertex;
    };

    struct maxSat {
        bool operator()(const Node& lhs,
                        const Node& rhs) const
        {
            return std::tie(lhs.sat, lhs.deg, lhs.vertex)
                   > std::tie(rhs.sat, rhs.deg, rhs.vertex);
        }
    };

    static void find_color_greedy(const Graph& graph, const int i, int& nb_color, std::vector<int>& colors) {
        std::set<int> used_colors;
        for(const int id : graph.adj[i]) {
            if(colors[id] != -1)
                used_colors.insert(colors[id]);
        }

        std::vector<int> available_colors;
        for(int c = 0; c < nb_color; ++c) {
            if(used_colors.find(c) == used_colors.end())
                available_colors.push_back(c);
        }

        if(available_colors.empty()) {
            colors[i] = nb_color;
            nb_color++;
        }
        else {
            colors[i] = available_colors.front();
        }
    }

};