# HP3_project
High Perfomance Parallel Programming's End Term Project - Accelerating Graph Algorithms on GPU

Secondary Branch:- This branch does not have large graphs as a part of the datsets. Switch to primary branch for that.

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

## Datasets

We have two classes of graph, one for APSP, and the other for BFS and SSSP, as the APSP algorithms cannot process a graph of size more than a few thousand vertices.

### APSP Graphs
- Graph 1 (Random)
  - Vertices: 2,700
  - Edges: 1,808,853
  - Avg. Degree: 1,340
  - Source: Randomly generated using mt19937 random engine
- Graph 2 (Facebook Friends Circle)
  - Vertices: 4,039
  - Edges: 88,234
  - Avg. Degree: 44
  - Source: [SNAP](https://snap.stanford.edu/data/ego-Facebook.html)
- Graph 3 (Biological Gene Networks)
  - Vertices: 4,412
  - Edges: 108,818
  - Avg. Degree: 49
  - Source: [Network Repo](http://networkrepository.com/bio-HS-CX.php)
- Graph 4 (Bitcoin OTC Trust)
  - Vertices: 5881
  - Edges: 21492
  - Avg. Degree: 7
  - Source: [SNAP](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html)
- Graph 5 (Random)
  - Vertices: 7,500
  - Edges: 837,083
  - Avg. Degree: 223
  - Source: Randomly generated using mt19937 random engine
  
## Executing Commands
  - To generate all executables: `make all` (or) `make`
  - To generate a particular executable: `make <exec>` 
  - To delete all generated executables: `make clean`
