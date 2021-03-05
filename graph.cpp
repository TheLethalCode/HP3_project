#include <ctime>
#include <iostream>
#include "graph.h"

using namespace std;

void readGraph(Graph &G, int argc, char **argv) {
    int n;
    int m;
    bool hasInput = 1;
    if (argc == 1){
        hasInput = 0;
        n = rand() % 100 + 1;
        m = rand() % 200 + 1;
    } else {
        cin >> n >> m;
    }
    // cout << "n: " << n << " m: " << m << "\n";
    vector<vector<int> > adjecancyLists(n);
    for (int i = 0; i < m; i++) {
            int u, v;
            if (hasInput){
                cin >> u >> v; // 1-based indexing
                --u; --v;
            }
            else{
                u = rand() % n;
                v = rand() % n;
            }

            adjecancyLists[u].push_back(v);
            adjecancyLists[v].push_back(u);
    }

    for (int i = 0; i < n; i++) {
        G.edgesOffset.push_back(G.adjacencyList.size());
        G.edgesSize.push_back(adjecancyLists[i].size());
        for (auto &edge: adjecancyLists[i]) {
            G.adjacencyList.push_back(edge);
        }
    }

    G.numVertices = n;
    G.numEdges = G.adjacencyList.size();
}