# HP3_project
High Perfomance Parallel Programming's End Term Project - Shortest Path Algorithms for Graphs on CUDA

## Executing Commands
- Parallel simple bfs (BFS)
  - `make BFS` 
  - `./BFS` - If graph is inputted
  - `./BFS 1 <numVertex> <numEdge> <weightLimit> <seed>`('weightLimit' and 'seed' are optional) - If randomly generate graph

- Single Source Shortest Path (SSSP)
  - `make SSSP` 
  - `./SSSP` - If graph is inputted
  - `./SSSP 1 <numVertex> <numEdge> <weightLimit> <seed>`('weightLimit' and 'seed' are optional) - If randomly generate graph

- All Pairs Shortest Path (Floyd Warshall) (APSP1)
  - `make APSP1` 
  - `./APSP1` - If graph is inputted
  - `./APSP1 1 <numVertex> <numEdge> <weightLimit> <seed>`('weightLimit' and 'seed' are optional) - If randomly generate graph

- All Pairs Shortest Path (Repeated SSSP) (APSP2)
  - `make APSP2` 
  - `./APSP2` - If graph is inputted
  - `./APSP2 1 <numVertex> <numEdge> <weightLimit> <seed>`('weightLimit' and 'seed' are optional) - If randomly generate graph
