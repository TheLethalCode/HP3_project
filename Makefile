CC = nvcc

all: SSSP APSP1 APSP2 BFS scanBFS queueBFS

SSSP: src/core.cu src/graph.cpp src/shortestPathCPU.cpp src/SSSPutils.cu src/SSSPMain.cu
	${CC} $^ -o $@

APSP1: src/core.cu src/graph.cpp src/shortestPathCPU.cpp src/APSPutils.cu src/APSPMain1.cu
	${CC} $^ -o $@

APSP2: src/core.cu src/graph.cpp src/shortestPathCPU.cpp src/SSSPutils.cu src/APSPMain1.cu
	${CC} $^ -o $@

BFS: src/core.cu src/graph.cpp src/bfsCPU.cpp src/BFSutils.cu src/BFSMain.cu
	${CC} $^ -o $@

scanBFS: src/core.cu src/graph.cpp src/bfsCPU.cpp src/BFSutils.cu src/scanBFSMain.cu
	${CC} $^ -o $@

queueBFS: src/core.cu src/graph.cpp src/bfsCPU.cpp src/BFSutils.cu src/queueBFSMain.cu
	${CC} $^ -o $@

clean:
	rm -f SSSP APSP1 APSP2 BFS scanBFS queueBFS