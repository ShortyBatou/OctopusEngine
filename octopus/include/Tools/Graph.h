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

    void remove_edge(const int a, const int b) {
        adj[a].erase(b);
        adj[b].erase(a);
        degree[a]--;
        degree[b]--;
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
    std::vector<int> colors;
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

    static Coloration Primal_Dual_Element(const Element& elem, const Mesh::Topology& topo, const Graph& p_graph, const Graph& d_graph) {
        const int nb_vert_elem = elem_nb_vertices(elem);
        int n = p_graph.n;
        std::set<int> v_conflict; // [vid, nb_conflict]

        std::vector<int> colors(n,-1);
        std::vector<int> elem_color(nb_vert_elem,0);
        std::iota(elem_color.begin(), elem_color.end(), 0);

        // get the graph of the reference element
        const Graph e_graph(Line, get_ref_element_edges(elem));

        std::queue<int> q; // element to visit
        std::vector<bool> visited(d_graph.n, false);
        const int eid = Random::Range(0, d_graph.n);
        for(int i = 0; i < nb_vert_elem; i++) {
            colors[topo[eid * nb_vert_elem + i]] = i;
        }
        add_neighbors(eid, d_graph, visited, q);

        // while there is element to color
        while(!q.empty() ) {
            // get element id and offset in topo array
            const int eid = q.front();
            const int offset = eid * nb_vert_elem;
            const int* e_topo = topo.data() + offset;
            q.pop();

            // get coloration and saturation in element graph
            // get all vertices  to color
            std::vector<int> sub_coloration = get_local_coloration(nb_vert_elem, e_topo, colors);
            std::vector<int> sub_saturation = get_local_staturation(nb_vert_elem, e_topo, e_graph, colors);
            complete_ref_element_coloration(nb_vert_elem, e_graph, sub_saturation, sub_coloration);

            const std::set<int> conflicts = get_local_conflicts(nb_vert_elem, e_topo, p_graph, sub_coloration, colors);
            for(const int rid : conflicts) {
                sub_coloration[rid] = -1;
            }

            // if there is a conflict ignore this coloration
            for(int i = 0; i < nb_vert_elem; i++) {
                const int ci = sub_coloration[i];
                const int vid = topo[offset+i];
                if(colors[vid] != -1) continue; // don't touch if already colored
                if(ci != -1) colors[vid] = ci; // only apply coloration when possible
                else v_conflict.insert(vid); // else add vertices in conflict
            }

            if(conflicts.empty()) {
                // add neighbors elements
                add_neighbors(eid, d_graph, visited, q);
            }
        }

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
        }

        return { *std::max_element(colors.begin(), colors.end())+1, colors };
    }



    static Coloration Primal_Dual_Element_2(
        const Element& elem, const Mesh::Topology& topo,
        const Graph& p_graph, const Graph& d_graph, const std::vector<std::vector<int>>& owners)
    {
        const int nb_vert_elem = elem_nb_vertices(elem);
        const int n = p_graph.n;

        std::vector<int> colors(n,-1);
        std::set<int> v_conflict;
        std::set<int> e_conflict;

        // get the graph of the reference element and coloration
        const Graph e_graph(Line, get_ref_element_edges(elem));
        std::vector<int> elem_color(nb_vert_elem,0);
        std::iota(elem_color.begin(), elem_color.end(), 0);

        // element visit information
        std::queue<int> q_tile; // element to visit
        std::queue<int> q_conflict;
        q_conflict.push(Random::Range(0, d_graph.n - 1));
        std::vector<bool> visited(d_graph.n, false); // visited elements
        // while there is element to color
        while(!q_tile.empty() || !q_conflict.empty()) {
            // color a random element
            int eid;
            if(q_tile.empty()) {
                eid = q_conflict.front();
                q_conflict.pop();
            }
            else {
                eid = q_tile.front();
                q_tile.pop();
            }

            const int* e_topo = topo.data() + eid * nb_vert_elem;

            // get coloration in element graph
            std::vector<int> local_coloration;
            local_coloration = get_local_coloration(nb_vert_elem, e_topo, colors);
            color_ref_element(nb_vert_elem, e_graph, local_coloration);
            //std::vector<int> local_saturation = get_local_staturation(nb_vert_elem, e_topo, e_graph, colors);
            //complete_ref_element_coloration(nb_vert_elem, e_graph, local_saturation, local_coloration);

            // check if this coloration create conflicts (return rid of vertices in conflinct)
            const std::set<int> conflicts = get_local_conflicts(nb_vert_elem, e_topo, p_graph, local_coloration, colors);

            // handle conflicts and try to find the lower color
            for(const int rid : conflicts) {
                int vid = e_topo[rid];
                v_conflict.insert(vid);
                for(int own : owners[vid]) {
                    e_conflict.insert(own);
                }
            }

            for(int i = 0; i < nb_vert_elem; i++) {
                colors[e_topo[i]] = local_coloration[i];
            }

            if(conflicts.empty()) {
                // only add neighbors that does not have conflict vertices
                for(int neighbor : d_graph.adj[eid]) {
                    if(visited[neighbor]) continue;

                    if(e_conflict.find(neighbor) != e_conflict.end()) q_conflict.push(neighbor);
                    else q_tile.push(neighbor);
                    visited[neighbor] = true;
                }
            }
        }

        // handle conflict at the end by coloring them by "importance"

        std::cout << "NB Conflict = " << v_conflict.size() << std::endl;
        return { *std::max_element(colors.begin(), colors.end())+1, colors };
    }

    static Mesh::Topology get_ref_element_edges(Element e) {
        Mesh::Topology edges = ref_edges(e);
        // edges go in both direction
        Mesh::Topology edges2 = ref_edges(e); // A => B
        std::reverse(edges2.begin(), edges2.end()); // B => A
        edges.insert(edges.end(), edges2.begin(), edges2.end());
        return edges;
    }

    static int get_random_element(std::set<int> elements)
    {
        srand(time(NULL));
        auto it = std::begin(elements);
        // 'advance' the iterator n times
        const int first_id = elements.size() * Random::Eval();
        std::advance(it,first_id);
        return *it;
    }

    static void add_neighbors(const int id, const Graph& graph, std::vector<bool>& visited, std::queue<int>& q) {
        for(int neighbor : graph.adj[id]) {
            // if not visited => add to queue for coloring
            if(!visited[neighbor]) {
                q.push(neighbor);
                visited[neighbor] = true;
            }
        }
    }

    static void color_ref_element(int nb, const Graph& e_graph, std::vector<int>& local_coloration) {
        std::set<int> to_color; // colored vertices that doesn't have all their neighbors colored
        std::vector<int> c = local_coloration;
        std::vector<std::set<int>> availables(nb);
        for(int i = 0; i < nb; ++i) {
            if(local_coloration[i] == -1) to_color.insert(i);

            for(int j = 0; j < nb; ++j)
                availables[i].insert(j);
        }

        for(int i = 0; i < nb; ++i) {
            const int ci = local_coloration[i];
            if(ci == -1) continue;
            std::set<int> n_color; // all color that can appear in neighbors
            for(int j : e_graph.adj[ci]) {
                n_color.insert(j);
            }

            for(const int j : e_graph.adj[i]) {
                std::set<int> intersect;
                std::set_intersection(
                    availables[j].begin(), availables[j].end(), n_color.begin(), n_color.end(),
                    std::inserter(intersect, intersect.begin())
                );
                availables[j] = intersect;
            }

            for(int j = 0; j < nb; ++j) availables[j].erase(ci);
        }

        while(!to_color.empty()) {
            int id = *to_color.begin();
            int nb_color = nb;
            // get the vertex with the minimum coloration possibility
            for(const int i : to_color) {
                if(availables[i].size() == 0 || availables[i].size() >= nb_color) continue;
                id = i;
                nb_color = availables[i].size();
            }
            to_color.erase(id);
            int ci = -1;
            if(availables[id].size() > 0) {
                auto it = std::begin(availables[id]);
                const int r = Random::Range(0, static_cast<int>(availables[id].size()-1));
                std::advance(it, r);
                ci = *it;
            }

            local_coloration[id] = ci;

            std::set<int> n_color; // all color that can appear in neighbors
            for(int j : e_graph.adj[ci]) {
                n_color.insert(j);
            }
            availables[id].clear();
            for(const int j : e_graph.adj[id]) {
                std::set<int> intersect;
                std::set_intersection(
                    availables[j].begin(), availables[j].end(), n_color.begin(), n_color.end(),
                    std::inserter(intersect, intersect.begin())
                );
                availables[j] = intersect;
            }

            for(int j = 0; j < nb; ++j) availables[j].erase(ci);
        }

        for(int i = 0; i < nb; ++i) {
            std::set<int> possible;
            int ci = local_coloration[i];
            // all color we are supposed to find around i
            for(int j : e_graph.adj[ci]) {
                possible.insert(j);
            }

            // check all color around i
            for(int j : e_graph.adj[i]) {
                int cj = local_coloration[j];
                if(possible.find(cj) == possible.end()) {
                    std::cout << "err" << std::endl;
                }
            }
        }
    }

    static void complete_ref_element_coloration(int nb, const Graph& e_graph, std::vector<int>& saturation, std::vector<int>& coloration) {
        std::vector<int> not_saturated; // colored vertices that doesn't have all their neighbors colored
        for(int i = 0; i < nb; ++i) {
            // if there is only one color missing, we add the vertex ref id to color
            const int diff = e_graph.degree[i] - saturation[i];
            if(diff == 1 && coloration[i] != -1) not_saturated.push_back(i);
        }

        // while there is that are not fully saturated
        // do an indirect coloration. We are on a vertex that have one neighbor that is not colored.
        // With the reference element coloration, we can deduce the color of the vertex.
        while(!not_saturated.empty()) {
            // get the vertex id (in the reference element) and color
            const int& rid = not_saturated.back();
            const int c_i = coloration[rid];
            not_saturated.pop_back();

            // happens often (many vertex can be saturated at the same time when one vertex is colored)
            if(saturation[rid] == e_graph.degree[rid]) continue;

            // get the color which is not used
            std::set<int> available_color;
            for(int c_j : e_graph.adj[c_i]) {
                available_color.insert(c_j);
            }

            int last_j = -1;
            for(const int j : e_graph.adj[rid]) {
                int c_j = coloration[j];
                if(c_j == -1) last_j = j;
                available_color.erase(c_j);
            }
            // only one color remains
            const int c_j = *available_color.begin();
            coloration[last_j] = c_j;
            // update staturation, and add new vertices that can be saturated
            for(const int k : e_graph.adj[last_j]) {
                saturation[k]++;
                const int diff = e_graph.degree[k] - saturation[k];
                if(diff == 1 && coloration[k] != -1) not_saturated.push_back(k);
            }
        }
    }

    static std::vector<int> get_local_coloration(const int nb, const int* e_topo, const std::vector<int>& colors) {
        std::vector<int> local_coloration(nb, 0);
        for(int i = 0; i < nb; ++i) {
            local_coloration[i] = colors[e_topo[i]];
        }
        return local_coloration;
    }

    static std::vector<int> get_local_staturation(const int nb, const int* e_topo, const Graph& e_graph, const std::vector<int>& colors) {
        std::vector<int> local_saturation(nb, 0);
        for(int i = 0; i < nb; ++i) {
            for(const int j : e_graph.adj[i]) {
                if(colors[e_topo[j]] == -1) continue;
                local_saturation[i]++;
            }
        }
        return local_saturation;
    }

    static std::set<int> get_local_conflicts(const int nb, const int* e_topo, const Graph& p_graph, const std::vector<int>& sub_coloration, const std::vector<int> &colors) {
        std::set<int> conflicts;

        for(int i = 0; i < nb; i++) {
            const int ci = sub_coloration[i];
            const int vid = e_topo[i];
            if(colors[vid] != -1) continue; // don't touch if already colored
            for(const int j : p_graph.adj[vid]) {
                if(colors[j] != ci) continue;
                conflicts.insert(i);
                break;
            }
        }
        return conflicts;
    }


    static int Nb_Conflict(const Graph& graph, const Coloration& coloration) {
        return static_cast<int>(Get_Conflict(graph, coloration).size());
    }

    static std::map<int, int> Get_Conflict(const Graph& graph, const Coloration& coloration) {
        std::map<int, int> conflict;
        for(int i = 0; i < graph.n; i++) {
            int c = coloration.colors[i];
            for(int neighbor : graph.adj[i]) {
                int n_c = coloration.colors[neighbor];
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

struct GraphReduction {
    struct Map {
        Map(const int nb_vert, const Mesh::Topology& topo) {
            weights.resize(nb_vert);
            true_id.resize(nb_vert);
            nb_split.resize(nb_vert);
            topology = topo;
        }
        std::vector<int> topology;
        std::vector<int> true_id;

        std::vector<int> nb_split;
        std::vector<int> id_split;
        std::vector<int> off_split;
        std::vector<scalar> weights;
    };

    static Map Color_Split_Primal_Dual(
        const Element elem,
        const Mesh::Topology& topo,
        const Graph& graph,
        Coloration& coloration,
        std::vector<std::vector<int>>& owners,
        std::vector<std::vector<int>>& r_ids
        )
    {
        const int e_nb_vert = elem_nb_vertices(elem);
        std::vector<int>& colors = coloration.colors;
        coloration.nb_color = e_nb_vert;
        Map map(colors.size(), topo);

        // get all vertices with a color >= nb
        std::set<int> to_color;
        for(int i = 0; i < graph.n; ++i) {
            if(colors[i] >= e_nb_vert) {
                to_color.insert(i);
            }
        }

        // for each vertex
        for(const int vid : to_color) {
            const int nb_owners = owners[vid].size();
            std::vector<std::set<int>> color_available(nb_owners);
            std::vector<int> nb_occurence(e_nb_vert, 0); // count the number of occurrence to choose the most common color
            // get all possible color for each element for this vertex
            for(int j = 0; j < nb_owners; ++j) {
                const int* e_topo = topo.data() + owners[vid][j] * e_nb_vert;
                int rid = r_ids[vid][j]; // id in the current element
                for(int c = 0; c < e_nb_vert; ++c) color_available[j].insert(c);
                for(int k = 0; k < e_nb_vert; ++k) color_available[j].erase(colors[e_topo[k]]);
                for(const int c : color_available[j]) nb_occurence[c]++;
            }

            // select the most reccurent color for each split
            std::vector<int> s_colors(nb_owners, -1);
            for(int j = 0; j < nb_owners; ++j) {
                int c_max = -1, nb_max = 0;
                for(const int c : color_available[j]) {
                    if(nb_occurence[c] <= nb_max) continue;
                    nb_max = nb_occurence[c];
                    c_max = c;
                }
                s_colors[j] = c_max;
            }

            // get the list of used color (with no duplicates)
            std::vector<int> used_color(s_colors.begin(), s_colors.end());
            std::sort( used_color.begin(), used_color.end() );
            used_color.erase( std::unique( used_color.begin(), used_color.end() ), used_color.end() );

            // merge the color
            std::vector<std::vector<int>> v_owners(used_color.size());
            std::vector<std::vector<int>> v_rid(used_color.size());
            for(int i = 0; i < used_color.size(); ++i) {
                const int c = used_color[i];
                for(int j = 0; j < nb_owners; ++j) {
                    if(c != s_colors[j]) continue;
                    v_owners[i].push_back(owners[vid][j]);
                    v_rid[i].push_back(r_ids[vid][j]);
                }
            }

            // update data and map
            colors[vid] = s_colors[0];
            owners[vid] = v_owners[0];
            r_ids[vid] = v_rid[0];
            map.true_id[vid] = vid;

            map.off_split.push_back(map.weights.size());
            map.nb_split.push_back(used_color.size());
            map.weights.push_back(static_cast<scalar>(v_owners[0].size()) / static_cast<scalar>(nb_owners));
            map.id_split.push_back(vid);
            for(int i = 1; i < used_color.size(); ++i) {
                int new_id = colors.size();
                colors.push_back(s_colors[i]);
                owners.push_back(v_owners[i]);
                r_ids.push_back(v_rid[i]);

                map.weights.push_back(static_cast<scalar>(v_owners[0].size()) / static_cast<scalar>(nb_owners));
                map.id_split.push_back(vid);

                map.true_id.push_back(vid);
                for(int j = 0; j < v_owners[i].size(); j++) {
                    int index = v_owners[i][j] * e_nb_vert + v_rid[i][j];
                    map.topology[index] = new_id;
                }
            }
        }
        return map;
    }
};

struct GraphBalance  {
    static void Greedy(const Graph& graph, Coloration& coloration, const int nb_iteration = 1000000) {
        int nb_it = 0;
        std::map<int, std::set<int>> c_verts;
        for(int c = 0; c < coloration.nb_color; ++c) {
            c_verts[c] = std::set<int>();
        }

        for (int i = 0; i < graph.n; ++i) {
            int c = coloration.colors[i];
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
        while(color_change && nb_it < nb_iteration) {
            if(nb_it % 10000) std::cout << static_cast<scalar>(nb_it) / nb_iteration * 100.f<< "%" << std::endl;
            nb_it++;

            color_change = false;
            for(int i = 0; i < over_represented.size(); ++i) {
                int over_color = over_represented[i];
                // get all vertices of this color and try to assign to an under represented color
                for (int vid : c_verts[over_color]) {
                    std::set<int> used_colors;
                    for (const int neighbor : graph.adj[vid]) {
                        used_colors.insert(coloration.colors[neighbor]);
                    }

                    for(int j = 0; j < under_represented.size(); ++j) {
                        int under_color = under_represented[j];
                        if (used_colors.count(under_color)) continue;
                        coloration.colors[vid] = under_color;
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