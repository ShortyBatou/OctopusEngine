#pragma once
#include "Core/Base.h"
#include <vector>
#include <set>
#include <algorithm>
#include <fstream>
#include <iso646.h>
#include <numeric>
#include <queue>
#include <UI/AppInfo.h>

#include "Random.h"

struct Graph {
    Graph() : n(0) {}

    Graph(const Element element, const Mesh::Topology& topology, const bool primal = true) {
        if(primal) build_primal(element, topology);
        else build_dual(element, topology);
    }

    explicit Graph(const std::vector<std::set<int>>& adjacence) : n(adjacence.size()), adj(adjacence) {
        degree.resize(n);
        for(int i = 0; i < adj.size(); ++i) {
            degree[i] = adj[i].size();
        }
    }

    void build_primal(const Element element, const Mesh::Topology& topology) {
        const int nb_vert_elem = elem_nb_vertices(element);
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

    void build_dual(const Element element, const Mesh::Topology& topology) {
        const int nb_vert_elem = elem_nb_vertices(element);
        n = topology.size() / nb_vert_elem; // nb_element
        degree.resize(n);
        adj.resize(n);
        const Element lin_elem = get_linear_element(element);
        const Mesh::Topology r_tri = ref_triangles(lin_elem);
        const Mesh::Topology r_quads = ref_quads(lin_elem);

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

    void save_as_file(const std::string &filename) const {
        const std::string f_path = AppInfo::PathToAssets() + filename + ".txt";
        std::ofstream file;
        file.open(f_path);

        for(int i = 0; i < n; ++i) {
            for(const int id : adj[i]) {
                file << i << ":" << id << ", ";
            }
            file << std::endl;
        }
    }

    void add_edge(const int a, const int b)
    {
        adj[a].insert(b);
        adj[b].insert(a);
        degree[a]++;
        degree[b]++;
    }

    int new_vertice()
    {
        n++;
        degree.push_back(0);
        adj.push_back({});
        return n - 1;
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

    static Coloration Primal_Dual_DSAT(const Element& elem, const Mesh::Topology& topo, const Graph& p_graph, const Graph& d_graph) {
        const int nb_vert_elem = elem_nb_vertices(elem);
        std::vector<int> colors(p_graph.n,-1);
        std::queue<int> q; // element to visit
        std::vector<bool> visited(d_graph.n, false);

        {
            // color the first element
            for(int i = 0; i < nb_vert_elem; i++) {
                colors[topo[i]] = i;
            }

            // adds its neighbors to queue
            for(int neighbor : d_graph.adj[0]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }

        // while there is element to color
        while(!q.empty()) {
            // get element id and offset in topo array
            const int eid = q.front();
            const int offset = eid * nb_vert_elem;
            q.pop();
            // get vertices that are not colored and get their saturation
            std::vector<std::pair<int,int>> sat_and_id;
            for(int i = 0; i < nb_vert_elem; i++) {
                const int vid = topo[offset + i];
                int c = colors[vid];
                if (c != -1) continue;
                int s = 0;
                for(const int neighbor : p_graph.adj[vid]) {
                    if(colors[neighbor] != -1) s++;
                }
                sat_and_id.push_back({s,vid});
            }

            std::sort(sat_and_id.begin(), sat_and_id.end(),
                [&](const std::pair<int,int>& a, const std::pair<int,int>& b) {
                return a.first > b.first;
            });

            for(auto [sat, vid] : sat_and_id) {
                std::set<int> used_color;
                for(const int neighbor : p_graph.adj[vid]) {
                    used_color.insert(colors[neighbor]);
                }
                int c = 0;
                while(used_color.find(c) != used_color.end()) {
                    c++;
                }
                colors[vid] = c;
            }

            // add neighbors elements
            for(int e_neighbor : d_graph.adj[eid]) {
                // if not visited => add to queue for coloring
                if(!visited[e_neighbor]) {
                    q.push(e_neighbor);
                    visited[e_neighbor] = true;
                }
            }
        }

        return { *std::max_element(colors.begin(), colors.end())+1, colors };
    }

    static Coloration Primal_Dual_Element(const Element& elem, const Mesh::Topology& topo, const Graph& p_graph, const Graph& d_graph) {
        const int nb_vert_elem = elem_nb_vertices(elem);
        int n = p_graph.n;
        std::set<int> v_conflict; // [vid, nb_conflict]

        std::vector<int> colors(n,-1);
        std::vector<int> elem_color(nb_vert_elem,0);
        std::iota(elem_color.begin(), elem_color.end(), 0);

        // get the graph of the reference element
        Mesh::Topology edges = ref_edges(elem);
        {
            // edges go in both direction
            Mesh::Topology edges2 = ref_edges(elem); // A => B
            std::reverse(edges2.begin(), edges2.end()); // B => A
            edges.insert(edges.end(), edges2.begin(), edges2.end());
        }
        const Graph e_graph(Line, edges);

        std::queue<int> q; // element to visit
        std::vector<bool> visited(d_graph.n, false);

        // color the first element
        {
            srand(time(NULL));
            //const int first_id = d_graph.n * Random::Eval();
            const int first_id = 15803;
            std::cout << "ID " << first_id << std::endl;
            for(int i = 0; i < nb_vert_elem; i++) {
                colors[topo[first_id * nb_vert_elem + i]] = i;
            }

            // adds its neighbors to queue
            for(int neighbor : d_graph.adj[first_id]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }

        // while there is element to color
        while(!q.empty()) {
            // get element id and offset in topo array
            const int eid = q.front();
            const int offset = eid * nb_vert_elem;
            q.pop();

            // get all colored vertices in element
            std::vector<int> sub_coloration(nb_vert_elem);
            std::vector<int> sub_saturation(nb_vert_elem, 0);
            std::vector<int> not_saturate; // colored vertices that doesn't have all their neighbors colored
            int nb_color = 0;
            // get coloration and saturation in element graph
            for(int i = 0; i < nb_vert_elem; ++i) {
                sub_coloration[i] = colors[topo[offset + i]];
                if(sub_coloration[i] != -1) nb_color++;

                for(const int j : e_graph.adj[i]) {
                    if(colors[topo[offset+j]] == -1) continue;
                    sub_saturation[i]++;
                }
                int diff = e_graph.degree[i] - sub_saturation[i];
                if(diff == 1 && sub_coloration[i] != -1) not_saturate.push_back(i);
            }

            if(nb_color == 0) {
                for(int i = 0; i < nb_vert_elem; ++i) {
                    sub_coloration[i] = i;
                }
            }

            while(!not_saturate.empty()) {
                const int& rid = not_saturate.back();
                not_saturate.pop_back();
                if(sub_saturation[rid] == e_graph.degree[rid]) continue;

                const int c_i = sub_coloration[rid];

                std::set<int> available_color;
                for(int c_j : e_graph.adj[c_i]) {
                    available_color.insert(c_j);
                }

                int last_j = -1;
                for(int j : e_graph.adj[rid]) {
                    int c_j = sub_coloration[j];
                    if(c_j == -1) last_j = j;
                    available_color.erase(c_j);
                }

                int c_j = *available_color.begin();
                sub_coloration[last_j] = c_j;

                for(const int k : e_graph.adj[last_j]) {
                    sub_saturation[k]++;
                    int diff = e_graph.degree[k] - sub_saturation[k];
                    if(diff == 1 && sub_coloration[k] != -1) not_saturate.push_back(k);
                }
            }

            bool conflict = false;
            for(int i = 0; i < nb_vert_elem; i++) {
                const int ci = sub_coloration[i];
                const int vid = topo[offset+i];
                if(colors[vid] != -1) continue; // don't touch if already colored
                for(int j : p_graph.adj[vid]) {
                    if(colors[j] != ci) continue;
                    conflict = true;
                    break;
                }
                if(conflict) sub_coloration[i] = -1;
            }

            // if there is a conflict ignore this coloration
            for(int i = 0; i < nb_vert_elem; i++) {
                const int ci = sub_coloration[i];
                const int vid = topo[offset+i];
                if(colors[vid] != -1) continue; // don't touch if already colored
                //if(!conflict) colors[vid] = ci; // only apply coloration when possible
                //else v_conflict.insert(vid); // else add vertices in conflict
                if(ci != -1) colors[vid] = ci; // only apply coloration when possible
                else v_conflict.insert(vid); // else add vertices in conflict
            }

            if(!conflict) {
                // add neighbors elements
                for(int e_neighbor : d_graph.adj[eid]) {
                    // if not visited => add to queue for coloring
                    if(!visited[e_neighbor]) {
                        q.push(e_neighbor);
                        visited[e_neighbor] = true;
                    }
                }
            }
        }
        /*
        //use DSAT to finish coloration
        std::vector<int> adjCols(p_graph.n,0);
        std::set<Node, maxSat> max_queue; // sort vertices depending on their saturation > degree > id
        std::vector<int> degree(p_graph.degree);
        // generate nodes
        for (int vid = 0; vid < p_graph.n; vid++) {
            for(int j : p_graph.adj[vid]) {
                if(colors[j] == -1) continue;
                adjCols[vid]++;
            }
        }

        for (int c : v_conflict) {
            max_queue.insert({adjCols[c], degree[c], c});
        }

        // while there is vertices to color
        while (!max_queue.empty()) {
            // get and pop the max saturation vertex
            const auto maxPtr = max_queue.begin();
            int vid = maxPtr->vertex;
            max_queue.erase(maxPtr);

            // get used color around vertex
            std::set<int> used;
            for (const int v : p_graph.adj[vid]) {
                if (colors[v] == -1) continue;
                used.insert(colors[v]);
            }

            // get the first color i not used
            int i;
            for (i = 0; i < used.size(); i++)
                if (used.find(i) == used.end()) break;

            // color vertex
            colors[vid] = i;

            // add neighbors and change saturation of vertices around current vertex
            for (const int v : p_graph.adj[vid]) {
                if (colors[v] == -1) {
                    max_queue.erase({ adjCols[v], degree[v], v });
                    adjCols[v]++;
                    degree[v]--;
                    max_queue.emplace(Node{ adjCols[v],degree[v], v });
                }
            }
        }*/

        return { *std::max_element(colors.begin(), colors.end())+1, colors };
    }

    static Coloration BFS(const Graph& graph) {
        std::vector<int> colors(graph.n,-1);
        std::queue<int> q;
        std::vector<bool> visited(graph.n, false);
        visited[0] = true;
        q.push(0);
        while (!q.empty()) {
            const int vid = q.front();
            q.pop();
            std::set<int> used;
            for (const int j : graph.adj[vid]) {
                if(colors[j] == -1) continue;
                used.insert(colors[j]);
            }

            int c = 0;
            while(used.find(c) != used.end()) {
                c++;
            }

            colors[vid] = c;
            for (const int j : graph.adj[vid]) {
                if(visited[j] || colors[j] != -1) continue;
                visited[j] = true;
                q.push(j);
            }
        }
        return { *std::max_element(colors.begin(), colors.end())+1, colors };
    }


    static int Nb_Conflict(const Graph& graph, const Coloration& coloration) {
        return static_cast<int>(Get_Conflict(graph, coloration).size());
    }

    static std::map<int, int> Get_Conflict(const Graph& graph, const Coloration& coloration) {
        std::map<int, int> conflict;
        for(int i = 0; i < graph.n; i++) {
            int c = coloration.color[i];
            for(int neighbor : graph.adj[i]) {
                int n_c = coloration.color[neighbor];
                if(n_c != c) continue;
                if(conflict.find(i) == conflict.end()) {
                    conflict[i] = 0;
                }
                if(conflict.find(neighbor) == conflict.end()) {
                    conflict[neighbor] = 0;
                }
                conflict[i]++;
                conflict[neighbor]++;
            }
        }
        return conflict;
    }

private:
    struct Node {
        int sat;
        int deg;
        int vertex;
    };

    struct Node2 {
        int sat;
        int conflict;
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

    struct maxSat2 {
        bool operator()(const Node2& lhs,
                        const Node2& rhs) const
        {
            return std::tie(lhs.sat, lhs.conflict, lhs.deg, lhs.vertex)
                   > std::tie(rhs.sat, lhs.conflict, rhs.deg, rhs.vertex);
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

struct GraphBalance  {
    static void Greedy(const Graph& graph, Coloration& coloration, int nb_iteration = 1000000) {
        std::map<int, std::set<int>> c_verts;
        for (int i = 0; i < graph.n; ++i) {
            int c = coloration.color[i];
            c_verts[c].insert(i);
        }

        const int mean_size = graph.n / coloration.nb_color;

        // get the color that have more or less vertices that the mean size
        std::vector<int> over_represented;
        std::vector<int> under_represented;
        for (auto &[color, verts] : c_verts) {
            if (verts.size() >= mean_size) over_represented.push_back(color);
            else under_represented.push_back(color);
        }
        bool color_change = true;
        while(color_change) {
            color_change = false;
            for(int i = 0; i < over_represented.size(); ++i) {
                int over_color = over_represented[i];
                // get all vertices of this color and try to assign to an under represented color
                for (int vid : c_verts[over_color]) {
                    std::set<int> used_colors;
                    for (const int neighbor : graph.adj[vid]) {
                        used_colors.insert(coloration.color[neighbor]);
                    }

                    for(int j = 0; j < under_represented.size(); ++j) {
                        int under_color = under_represented[j];
                        if (used_colors.count(under_color)) continue;
                        coloration.color[vid] = under_color;
                        c_verts[over_color].erase(vid);
                        c_verts[under_color].insert(vid);
                        color_change = true;
                        if (c_verts[under_color].size() == mean_size) {
                            under_represented.erase(under_represented.begin() + j);
                            over_represented.push_back(under_color);
                        }
                        break;
                    }

                    if(c_verts[over_color].size() <= mean_size) {
                        over_represented.erase(over_represented.begin() + i);
                        i--;
                        under_represented.push_back(over_color);
                    }
                    if(color_change) break;
                }
            }
        }
    }
};