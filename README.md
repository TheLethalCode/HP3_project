# HP3_project
High Perfomance Parallel Programming's End Term Project - Accelerating Graph Algorithms on GPU

## Algorithms Overview
- Breadth First Search (BFS)
  - Simple BFS (Executable: BFS)
  - Queue BFS (Executable: queueBFS)
  - Scan BFS (Executable: scanBFS)

- Single Source Shortest Path (SSSP)
  - Djikstra-like SSSP (Executable: SSSP)

- All Pairs Shortest Path (APSP)
  - Repeated SSSP (Executable: APSP1)
  - Naive Floyd-Warshall (Executable: APSP2)
  - Block Floyd-Warshall (Executable: APSP3)

The details of the above mentioned algorithms can be found in the report

## Graph Input

All input graphs must be **weighted**. Even if the algorithm being run is a BFS, the program still expects a weight which it will ignore promptly (any random weight or a common weight of 1 will do in such a case). 

There are three ways in which an executable takes in the input graph to run the corresponding algorithm on. It is decided by the command line arguments passed to the executable.

- Manual graph input through **Standard Input** (terminal)
  - Command: `<exec>`  (or) `<exec> 0`
- Graph input through **File**
  - Command: `<exec> 1 <file_name>`
  - file_name: Path to the input file (can be relative or absolute)
- **Random** graph generation
  - Command: `<exec> 2 <num_vertex> <num_edge> <weight_lim> <seed>` 
  - num_vertex: The number of vertices in the graph
  - num_edges: The number of edges in the graph (Note:- The generated graph will have a slightly lesser number owing to repeated edge generation)
  - weight_lim (Optional): The upper limit of the weight of the edges
  - seed (Optional): The seed passed to the mt19937 random engine.

In all the above commands, <exec> refers to the executable file for running

### Input Format

If the input is either manual or through a file, it should follow the following format.

- First Line:- `<num_vertex> <num_edge>` (The number of vertices and edges respectively)
- Next *num_edge* lines:- `<end1> <end2> <weight>` (The two endpoints of the edge and its weight) 

## Executing Commands
  - To generate all executables: `make all` (or) `make`
  - To generate a particular executable: `make <exec>` 
  - To delete all generated executables: `make clean`
