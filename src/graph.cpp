#include "../include/graph.h"
#include <iostream>
#include <algorithm>
#include <random>

// Take Input or Generate Graph
Graph::Graph(int argc, char* argv[]) {
    if (argc == 1 || atoi(argv[1]) != 1) {
        std::cin >> n >> m;
        readGraph();
    } else if (argc >= 4) {
        n = atoi(argv[2]), m = atoi(argv[3]);
        int lim = (argc >= 5? atoi(argv[4]): 20);
        int seed = (argc >= 6? atoi(argv[5]): 81);
        genGraph(lim, seed);
    } else {
        std::cerr << "Incorrect arguments " << std::endl;
        exit(EXIT_FAILURE);
    }
    normGraph();
    convGraph();
}

void Graph::genGraph(int lim, int seed) {
    E.resize(n + 1);
    std::mt19937 rng(seed);
    for (int i = 1, a, b, c; i <= m; i++) {
        a = std::uniform_int_distribution<int>(0, n)(rng);
        b = std::uniform_int_distribution<int>(0, n)(rng);
        c = std::uniform_int_distribution<int>(1, lim)(rng);
        E[a].emplace_back(b, c);
        E[b].emplace_back(a, c);
    }
}

void Graph::readGraph() {
    E.resize(n + 1);
    for (int i = 1, a, b, c; i <= m; i++) {
        std::cin >> a >> b >> c;
        E[a].emplace_back(b, c);
        E[b].emplace_back(a, c);
    }
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