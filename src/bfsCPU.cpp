#include "../include/bfsCPU.h"

void bfsCPU(int start, Graph &G, int *distance,
            int *parent, bool *visited) {
    distance[start] = 0;
    parent[start] = start;
    visited[start] = true;
    std::queue<int> Q;
    Q.push(start);

    while (!Q.empty()) {
        int u = Q.front();
        Q.pop();

        for (int i = 1; i <= G.packE[G.posV[u]]; i++) {
            int v = G.packE[G.posV[u]+i];
            if (!visited[v]) {
                visited[v] = true;
                distance[v] = distance[u] + 1;
                parent[v] = i;
                Q.push(v);
            }
        }
    }
}
