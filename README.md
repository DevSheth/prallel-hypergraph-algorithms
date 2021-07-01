# CS6023 - GPU Programming - Jan2021 - Course Project

### Name: 		Sheth Dev Yashpal
### Roll No:	CS17B106

## Introduction:

This Project implements 4 algorithms on HyperGraphs from [1] - HyperBFS, HyperBPath, HyperSSSP, HyperPageRank in CUDA. The standard serial implementation in C++ can be accessed at [2] which is the official github repository for the paper [1]. This project provides a parallel implementation to run on GPUs.

## Instructions for compiling and running the source code:
	
1. `cd` to the `src/` directory and run `make` to compile the CUDA code.
2. To test the code run the command: `./main $(ALG) ../testcases/winput1.txt ../testcases/output.txt` where ALG can be any of - BFS, BPath, SSSP, PageRank.
3. The input `testcases/` have two types of HyperGraphs normal and weighted. Use only weighted HyperGraphs (`winput*.txt`) for running CUDA code.
4. For compiling the sreial code, `cd` to `ligra/apps/hyper/` and run the `make` command.
5. In the `ligra/apps/hyper/` directory, run the command `./Hyper$(ALG) ../../../testcases/input1.txt` for testing serial code. For SSSP use weighted inputs (`winput*.txt`), otherwise use normal inputs (`input*.txt`).
6. BFS and BPath are randomised by parallelisation so you cannot compare outputs for them. You can compare the outputs of SSSP and PageRank by looking at the generated files in the `testcases/` directory - `output.txt` for the CUDA implementation output and `Hyper$(ALG)_output.txt` for the serial implementation output.
7. You can generate additional testcases by using the graph generators provided by ligra. `cd` to `ligra/utils/` and run `make`. Run the command `./randHypergraph -nv <number of vertices> -nh <number of HyperEdges> -c <cardinality of each HyperEdge> ../../testcases/input*.txt` to generate normal HyperGraphs and `./adjHypergraphAddWeights ../../testcases/input*.txt ../../testcases/winput*.txt` to generate corresponding weighted HyperGraphs.

## References:

[1] Julian Shun. 2020. Practical parallel hypergraph algorithms. In Proceedings of the 25th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '20). Association for Computing Machinery, New York, NY, USA, 232–249. DOI:https://doi.org/10.1145/3332466.3374527

[2] https://github.com/jshun/ligra
