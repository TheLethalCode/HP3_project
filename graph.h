#ifndef GRAPH
#define GRAPH
#include <vector>
#include <utility>

const int INF = 2000000005;

struct Graph {
    int n, m;
    std::vector < std::vector < std::pair< int, int > > > E;
    std::vector < int > posV, packE, packW; 
    void genGraph(int, int, int, int);
    void readGraph(int, int);
    void normGraph();
    void convGraph();
};
int vecToArr(std::vector< int >&, int**);
#endif