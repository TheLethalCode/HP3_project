#include "graph.h"
#include <vector>
#include <utility>
#include <algorithm>
#include <set>

void djikstra(Graph &G, int s, int *dis) {
    for (int i = 0; i <= G.n; i++) {
        dis[i] = INF;
    } 
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

void floydWarshall(Graph &G, int **dis) {

    for (int i = 0; i <= G.n; i++) {
        for (int j = i + 1; j <= G.n; j++) {
            dis[i][j] = dis[j][i] = INF;
        }

        dis[i][i] = 0;
        for (auto v : G.E[i]) {
            dis[i][v.first] = v.second;
        }
    }

    for (int k = 0; k <= G.n; k++) {
        for (int i = 0; i <= G.n; i++) {
            for (int j = 0; j <= G.n; j++) {
                if (dis[i][k] < INF && dis[k][j] < INF) {
                    dis[i][j] = std::min(dis[i][j], dis[i][k] + dis[k][j]);
                }
            }
        }
    }
}