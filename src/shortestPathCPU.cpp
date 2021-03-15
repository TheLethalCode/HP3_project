#include "../include/graph.h"
#include <vector>
#include <utility>
#include <algorithm>
#include <set>

void djikstra(Graph &G, int s, int *dis) {
    dis[s] = 0;
    std::set < std::pair< int, int > > S;
    S.insert({0, s});
    
    while (!S.empty()) {
        int u = S.begin()->second;
        S.erase(S.begin());
        for (auto v : G.E[u]) {
            if (dis[v.first] > dis[u] + v.second) {
                if (dis[v.first] != INF) {
                    S.erase({dis[v.first], v.first});
                }
                dis[v.first] = dis[u] + v.second;
                S.insert({dis[v.first], v.first});
            }
        }
    }
}

void floydWarshall(int n, int **dis) {
    for (int k = 0; k <= n; k++) {
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= n; j++) {
                dis[i][j] = std::min(dis[i][j], dis[i][k] + dis[k][j]);
            }
        }
    }
}