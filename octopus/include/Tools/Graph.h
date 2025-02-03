#pragma once
#include "Core/Base.h"
#include <vector>
#include <set>
#include <algorithm>
struct Graph {
    Graph(const std::vector<std::set<int>>& adjacence) : n(adjacence.size()), adj(adjacence) {
        degree.resize(n);
        for(int i = 0; i < adjacence.size(); ++i) {
            degree[i] = adjacence[i].size();
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
        std::vector<int> colors(graph.n, -1);
        std::vector<int> saturation(graph.n, 0);

        while (true) {
            int maxSaturation = -1, maxDegree = -1, v = -1;
            for (int i = 0; i < graph.n; i++) {
                if (colors[i] == -1)
                {
                    if (saturation[i] > maxSaturation || (saturation[i] == maxSaturation && graph.degree[i] > maxDegree))
                    {
                        maxSaturation = saturation[i];
                        maxDegree = graph.degree[i];
                        v = i;
                    }
                }
            }

            if (v == -1) break;

            std::set<int> forbidden;
            for (const int neighbor : graph.adj[v]) {
                if (colors[neighbor] != -1) {
                    forbidden.insert(colors[neighbor]);
                }
            }

            int chosenColor = 0;
            while (forbidden.find(chosenColor) != forbidden.end()) {
                chosenColor++;
            }

            colors[v] = chosenColor;

            std::set<int> usedColors;
            for (const int neighbor : graph.adj[v]) {
                if (colors[neighbor] != -1) {
                    usedColors.insert(colors[neighbor]);
                }
            }
            for (const int neighbor : graph.adj[v]) {
                if (colors[neighbor] == -1) {
                    saturation[neighbor] = usedColors.size();
                }
            }
        }
        return { *std::max_element(colors.begin(), colors.end())+1, colors};
    }

private:
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