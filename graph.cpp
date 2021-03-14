#include "graph.h"
#include <iostream>
#include <algorithm>
#include <random>

void Graph::genGraph(int _n, int _m, int lim, int seed) {
    n = _n, m = _m;
    E.resize(n + 1);
    std::mt19937 rng(seed);
    for (int i = 1, a, b, c; i <= m; i++) {
        a = std::uniform_int_distribution<int>(0, n)(rng);
        b = std::uniform_int_distribution<int>(0, n)(rng);
        c = std::uniform_int_distribution<int>(1, lim)(rng);
        E[a].emplace_back(b, c);
        E[b].emplace_back(a, c);
    }
    normGraph();
    convGraph();
}

void Graph::readGraph(int _n, int _m) {
    n = _n, m = _m;
    E.resize(n + 1);
    for (int i = 1, a, b, c; i <= m; i++) {
        std::cin >> a >> b >> c;
        E[a].emplace_back(b, c);
        E[b].emplace_back(a, c);
    }
    normGraph();
    convGraph();
}

void Graph::normGraph() {
    int size = 0;
    for (int i = 0, x; i <= n; i++) {
        std::sort(E[i].begin(), E[i].end());
        auto it = std::unique(E[i].begin(), E[i].end(), [](std::pair<int,int> l, std::pair<int,int> r){
            return l.first == r.first;
        });
        x = it - E[i].begin();
        E[i].resize(x);
        std::shuffle(E[i].begin(), E[i].end(), std::mt19937());
        size += x;
    }
    m = size;
}

void Graph::convGraph() {
    for (int i = 0; i <= n; i++) {
        posV.emplace_back(packE.size());
        packE.emplace_back(E[i].size());
        packW.emplace_back(0);
        for (auto it : E[i]) {
            packE.emplace_back(it.first);
            packW.emplace_back(it.second);
        }
    }
}

int vecToArr(std::vector< int > &v, int **A) {
    *A = new int[v.size()];
    for (int i = 0; i < v.size(); i++) {
        (*A)[i] = v[i];
    }
    return v.size();
}