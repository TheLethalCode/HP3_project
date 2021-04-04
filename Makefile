CC = nvcc

all: SSSP SSSP_coarsen_2 SSSP_coarsen_4 SSSP_coarsen_8 SSSP_coarsen_16 APSP1 APSP1_coarsen_2 APSP1_coarsen_4 APSP1_coarsen_8 APSP2 APSP2_coarsen_2 APSP2_coarsen_4 APSP2_coarsen_8 APSP2_coarsen_16  BFS

SSSP: src/core.cu src/graph.cpp src/shortestPathCPU.cpp src/SSSPutils.cu src/SSSPMain.cu
	${CC} $^ -o $@

SSSP_coarsen_16: src/core.cu src/graph.cpp src/shortestPathCPU.cpp src/SSSPutils_coarsen_16.cu src/SSSPMain_coarsen_16.cu
	${CC} $^ -o $@

SSSP_coarsen_8: src/core.cu src/graph.cpp src/shortestPathCPU.cpp src/SSSPutils_coarsen_8.cu src/SSSPMain_coarsen_8.cu
	${CC} $^ -o $@

SSSP_coarsen_4: src/core.cu src/graph.cpp src/shortestPathCPU.cpp src/SSSPutils_coarsen_4.cu src/SSSPMain_coarsen_4.cu
	${CC} $^ -o $@

SSSP_coarsen_2: src/core.cu src/graph.cpp src/shortestPathCPU.cpp src/SSSPutils_coarsen_2.cu src/SSSPMain_coarsen_2.cu
	${CC} $^ -o $@

APSP1: src/core.cu src/graph.cpp src/shortestPathCPU.cpp src/APSPutils.cu src/APSPMain1.cu
	${CC} $^ -o $@

APSP1_coarsen_2: src/core.cu src/graph.cpp src/shortestPathCPU.cpp src/APSPutils_coarsen_2.cu src/APSPMain1.cu
	${CC} $^ -o $@

APSP1_coarsen_4: src/core.cu src/graph.cpp src/shortestPathCPU.cpp src/APSPutils_coarsen_4.cu src/APSPMain1.cu
	${CC} $^ -o $@

APSP1_coarsen_8: src/core.cu src/graph.cpp src/shortestPathCPU.cpp src/APSPutils_coarsen_8.cu src/APSPMain1.cu
	${CC} $^ -o $@

APSP2: src/core.cu src/graph.cpp src/shortestPathCPU.cpp src/SSSPutils.cu src/APSPMain2.cu
	${CC} $^ -o $@


APSP2_coarsen_2: src/core.cu src/graph.cpp src/shortestPathCPU.cpp src/SSSPutils_coarsen_2.cu src/APSPMain2.cu
	${CC} $^ -o $@


APSP2_coarsen_4: src/core.cu src/graph.cpp src/shortestPathCPU.cpp src/SSSPutils_coarsen_4.cu src/APSPMain2.cu
	${CC} $^ -o $@


APSP2_coarsen_8: src/core.cu src/graph.cpp src/shortestPathCPU.cpp src/SSSPutils_coarsen_8.cu src/APSPMain2.cu
	${CC} $^ -o $@


APSP2_coarsen_16: src/core.cu src/graph.cpp src/shortestPathCPU.cpp src/SSSPutils_coarsen_16.cu src/APSPMain2.cu
	${CC} $^ -o $@

BFS: src/core.cu src/graph.cpp src/bfsCPU.cpp src/BFSutils.cu src/BFS.cu
	${CC} $^ -o $@

clean:
	rm -f SSSP APSP1 APSP2 BFS SSSP_coarsen_2 SSSP_coarsen_4 SSSP_coarsen_8 SSSP_coarsen_16