#ifndef GRAPH
#define GRAPH
#include <vector>
#include <utility>

const int INF = 1000000007;

struct Graph {
    int n, m;
    std::vector < std::vector < std::pair< int, int > > > E;
    std::vector < int > posV, packE, packW; 
    Graph (int, char* argv[]);
    void genGraph(int, int);
    void readGraph();
    void normGraph();
    void convGraph();
};

#endif