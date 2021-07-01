#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include <bits/stdc++.h>

using namespace std;

#define debug 0

__global__ void useless(){}

__global__ void init(int n, int *arr, int val){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < n){
		arr[index] = val;
	}
}

__global__ void set_val(int *arr, int index, int val){
	arr[index] = val;
}

__global__ void BFS_step(int *frontier_in, int *frontier_out, int *off, int *adj, int *aux, int *parents){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int cnt = frontier_in[0];
	
	if(index < cnt){
		int c = 0;
		int s = frontier_in[index+1];
		
		for(int i=0; i<(off[s+1]-off[s]); i++){
			int d = adj[off[s] + i];
			
			if(atomicCAS(&parents[d], -1, s) == -1){
				aux[off[s] + c] = d;
				c++;
			}
		}

		int start = atomicAdd(&frontier_out[0], c);

		for(int i=0; i<c; i++){
			frontier_out[start + i + 1] = aux[off[s] + i];
		}
	}
}

void HyperBFS(int nv, int mv, int nh, int mh, int source, string outfile, int *offv, int *offh, int *adjv, int *adjh){
	
	int *parentsv;
	int *parentsh;
	cudaMalloc(&parentsv, nv * sizeof(int));
	cudaMalloc(&parentsh, nh * sizeof(int));
	init<<<(nv+31)/32, 32>>>(nv, parentsv, -1);
	init<<<(nh+31)/32, 32>>>(nh, parentsh, -1);

	int *auxv;
	int *auxh;
	cudaMalloc(&auxv, mv * sizeof(int));
	cudaMalloc(&auxh, mh * sizeof(int));

	int *frontierv;
	int *frontierh;
	cudaMalloc(&frontierv, (nv + 1) * sizeof(int));
	cudaMalloc(&frontierh, (nh + 1) * sizeof(int));

	int *check = (int *) malloc(sizeof(int));
	
	set_val<<<1,1>>>(frontierh, 0, 0);
	set_val<<<1,1>>>(frontierv, 0, 1);
	set_val<<<1,1>>>(frontierv, 1, source);
	set_val<<<1,1>>>(parentsv, source, source);

	while(1){
		// HyperBFS main loop
		
		cudaMemcpy(check, frontierv, sizeof(int), cudaMemcpyDeviceToHost);
		if(*check == 0) break;
		
		BFS_step<<<(*check+31)/32, 32>>>(frontierv, frontierh, offv, adjv, auxv, parentsh);

		set_val<<<1,1>>>(frontierv, 0, 0);

		if(debug){
			int *frontier = (int *) malloc((nh + 1) * sizeof(int));
			cudaMemcpy(frontier, frontierh, (nh + 1) * sizeof(int), cudaMemcpyDeviceToHost);

			cout << "frontierh ";
			for(int i=0; i<=nh; i++){
				cout << frontier[i] << " ";
			}
			cout << endl;

			int *parents = (int *) malloc(nh * sizeof(int));
			cudaMemcpy(parents, parentsh, nh * sizeof(int), cudaMemcpyDeviceToHost);

			cout << "parentsh ";
			for(int i=0; i<nh; i++){
				cout << parents[i] << " ";
			}
			cout << endl;
		}

		cudaMemcpy(check, frontierh, sizeof(int), cudaMemcpyDeviceToHost);
		if(*check == 0) break;
		
		BFS_step<<<(*check+31)/32, 32>>>(frontierh, frontierv, offh, adjh, auxh, parentsv);

		set_val<<<1,1>>>(frontierh, 0, 0);

		if(debug){
			int *frontier = (int *) malloc((nv + 1) * sizeof(int));
			cudaMemcpy(frontier, frontierv, (nv + 1) * sizeof(int), cudaMemcpyDeviceToHost);

			cout << "frontierv ";
			for(int i=0; i<=nv; i++){
				cout << frontier[i] << " ";
			}
			cout << endl;

			int *parents = (int *) malloc(nv * sizeof(int));
			cudaMemcpy(parents, parentsv, nv * sizeof(int), cudaMemcpyDeviceToHost);

			cout << "parentsv ";
			for(int i=0; i<nv; i++){
				cout << parents[i] << " ";
			}
			cout << endl;
		}
	}
	cudaDeviceSynchronize();
}

__global__ void init_neg(int n, int *arr, int *neg){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < n){
		arr[index] = -neg[index];
	}
}

__global__ void BPath_step(int *frontier_in, int *frontier_out, int *off, int *adj, int *aux, int *parents, int *worklist){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int cnt = frontier_in[0];
	
	if(index < cnt){
		int c = 0;
		int s = frontier_in[index+1];
		
		for(int i=0; i<(off[s+1]-off[s]); i++){
			int d = adj[off[s] + i];

			int old = atomicAdd(&parents[d], 1);
			if(old == -1){
				parents[d] = s;
			}
			else{
				if(atomicCAS(&worklist[d], 0, 1) == 0){
					aux[off[s] + c] = d;
					c++;
				}
			}
		}

		int start = atomicAdd(&frontier_out[0], c);

		for(int i=0; i<c; i++){
			frontier_out[start + i + 1] = aux[off[s] + i];
		}
	}
}

void HyperBPath(int nv, int mv, int nh, int mh, int source, string outfile, int *offv, int *offh, int *adjv, int *adjh, int *incntv, int *incnth){

	int *parentsv;
	int *parentsh;
	int *worklist;
	cudaMalloc(&parentsv, nv * sizeof(int));
	cudaMalloc(&parentsh, nh * sizeof(int));
	cudaMalloc(&worklist, nh * sizeof(int));
	init<<<(nv+31)/32, 32>>>(nv, parentsv, -1);
	init_neg<<<(nh+31)/32, 32>>>(nh, parentsh, incnth);
	init<<<(nh+31)/32, 32>>>(nh, worklist, 0);

	int *auxv;
	int *auxh;
	cudaMalloc(&auxv, mv * sizeof(int));
	cudaMalloc(&auxh, mh * sizeof(int));

	int *frontierv;
	int *frontierh;
	cudaMalloc(&frontierv, (nv + 1) * sizeof(int));
	cudaMalloc(&frontierh, (nh + 1) * sizeof(int));

	int *check = (int *) malloc(sizeof(int));
	
	set_val<<<1,1>>>(frontierh, 0, 0);
	set_val<<<1,1>>>(frontierv, 0, 1);
	set_val<<<1,1>>>(frontierv, 1, source);
	set_val<<<1,1>>>(parentsv, source, source);

	while(1){
		// HyperBFS main loop
		
		cudaMemcpy(check, frontierv, sizeof(int), cudaMemcpyDeviceToHost);
		if(*check == 0) break;
		
		BPath_step<<<(*check+31)/32, 32>>>(frontierv, frontierh, offv, adjv, auxv, parentsh, worklist);

		set_val<<<1,1>>>(frontierv, 0, 0);
		init<<<(nh+31)/32, 32>>>(nh, worklist, 0);

		if(debug){
			int *frontier = (int *) malloc((nh + 1) * sizeof(int));
			cudaMemcpy(frontier, frontierh, (nh + 1) * sizeof(int), cudaMemcpyDeviceToHost);

			cout << "frontierh ";
			for(int i=0; i<=nh; i++){
				cout << frontier[i] << " ";
			}
			cout << endl;

			int *parents = (int *) malloc(nh * sizeof(int));
			cudaMemcpy(parents, parentsh, nh * sizeof(int), cudaMemcpyDeviceToHost);

			cout << "parentsh ";
			for(int i=0; i<nh; i++){
				cout << parents[i] << " ";
			}
			cout << endl;
		}

		cudaMemcpy(check, frontierh, sizeof(int), cudaMemcpyDeviceToHost);
		if(*check == 0) break;
		
		BFS_step<<<(*check+31)/32, 32>>>(frontierh, frontierv, offh, adjh, auxh, parentsv);

		set_val<<<1,1>>>(frontierh, 0, 0);

		if(debug){
			int *frontier = (int *) malloc((nv + 1) * sizeof(int));
			cudaMemcpy(frontier, frontierv, (nv + 1) * sizeof(int), cudaMemcpyDeviceToHost);

			cout << "frontierv ";
			for(int i=0; i<=nv; i++){
				cout << frontier[i] << " ";
			}
			cout << endl;

			int *parents = (int *) malloc(nv * sizeof(int));
			cudaMemcpy(parents, parentsv, nv * sizeof(int), cudaMemcpyDeviceToHost);

			cout << "parentsv ";
			for(int i=0; i<nv; i++){
				cout << parents[i] << " ";
			}
			cout << endl;
		}
	}
	cudaDeviceSynchronize();
}

__global__ void SSSP_step(int *frontier_in, int *frontier_out, int *off, int *adj, int *wgh, int *aux, int *visit, int *shortest_in, int *shortest_out){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int cnt = frontier_in[0];
	
	if(index < cnt){
		int c = 0;
		int s = frontier_in[index+1];
		
		for(int i=0; i<(off[s+1]-off[s]); i++){
			
			int d = adj[off[s] + i];
			int newdist = shortest_in[s] + wgh[off[s] + i];
			
			int old = shortest_out[d];
			if(newdist < old){
				atomicMin(&shortest_out[d], newdist);

				if(atomicCAS(&visit[d], 0, 1) == 0){
					aux[off[s] + c] = d;
					c++;
				}
			}
		}

		int start = atomicAdd(&frontier_out[0], c);

		for(int i=0; i<c; i++){
			frontier_out[start + i + 1] = aux[off[s] + i];
		}
	}
}

void HyperSSSP(int nv, int mv, int nh, int mh, int source, string outfile, int *offv, int *offh, int *adjv, int *adjh, int *wghv, int *wghh){
	
	int *visitv;
	int *visith;
	cudaMalloc(&visitv, nv * sizeof(int));
	cudaMalloc(&visith, nh * sizeof(int));
	init<<<(nv+31)/32, 32>>>(nv, visitv, 0);
	init<<<(nh+31)/32, 32>>>(nh, visith, 0);

	int *shortestv;
	int *shortesth;
	cudaMalloc(&shortestv, nv * sizeof(int));
	cudaMalloc(&shortesth, nh * sizeof(int));
	init<<<(nv+31)/32, 32>>>(nv, shortestv, INT_MAX/2);
	init<<<(nh+31)/32, 32>>>(nh, shortesth, INT_MAX/2);

	int *auxv;
	int *auxh;
	cudaMalloc(&auxv, mv * sizeof(int));
	cudaMalloc(&auxh, mh * sizeof(int));

	int *frontierv;
	int *frontierh;
	cudaMalloc(&frontierv, (nv + 1) * sizeof(int));
	cudaMalloc(&frontierh, (nh + 1) * sizeof(int));

	int *check = (int *) malloc(sizeof(int));
	
	set_val<<<1,1>>>(frontierh, 0, 0);
	set_val<<<1,1>>>(frontierv, 0, 1);
	set_val<<<1,1>>>(frontierv, 1, source);
	set_val<<<1,1>>>(shortestv, source, 0);

	int round = 0;
	while(1){
		// HyperSSSP main loop

		if(round == nv-1){
			init<<<(nv+31)/32, 32>>>(nv, shortestv, -INT_MAX/2);
			break;
		}
		
		cudaMemcpy(check, frontierv, sizeof(int), cudaMemcpyDeviceToHost);
		if(*check == 0) break;
		
		SSSP_step<<<(*check+31)/32, 32>>>(frontierv, frontierh, offv, adjv, wghv, auxv, visith, shortestv, shortesth);

		set_val<<<1,1>>>(frontierv, 0, 0);
		init<<<(nh+31)/32, 32>>>(nh, visith, 0);

		if(debug){
			int *frontier = (int *) malloc((nh + 1) * sizeof(int));
			cudaMemcpy(frontier, frontierh, (nh + 1) * sizeof(int), cudaMemcpyDeviceToHost);

			cout << "frontierh ";
			for(int i=0; i<=nh; i++){
				cout << frontier[i] << " ";
			}
			cout << endl;

			int *shortest = (int *) malloc(nh * sizeof(int));
			cudaMemcpy(shortest, shortesth, nh * sizeof(int), cudaMemcpyDeviceToHost);

			cout << "shortesth ";
			for(int i=0; i<nh; i++){
				cout << shortest[i] << " ";
			}
			cout << endl;
		}

		cudaMemcpy(check, frontierh, sizeof(int), cudaMemcpyDeviceToHost);
		if(*check == 0) break;
		
		SSSP_step<<<(*check+31)/32, 32>>>(frontierh, frontierv, offh, adjh, wghh, auxh, visitv, shortesth, shortestv);

		set_val<<<1,1>>>(frontierh, 0, 0);
		init<<<(nv+31)/32, 32>>>(nv, visitv, 0);
		round++;

		if(debug){
			int *frontier = (int *) malloc((nv + 1) * sizeof(int));
			cudaMemcpy(frontier, frontierv, (nv + 1) * sizeof(int), cudaMemcpyDeviceToHost);

			cout << "frontierv ";
			for(int i=0; i<=nv; i++){
				cout << frontier[i] << " ";
			}
			cout << endl;

			int *shortest = (int *) malloc(nv * sizeof(int));
			cudaMemcpy(shortest, shortestv, nv * sizeof(int), cudaMemcpyDeviceToHost);

			cout << "shortestv ";
			for(int i=0; i<nv; i++){
				cout << shortest[i] << " ";
			}
			cout << endl;
		}
	}
	cudaDeviceSynchronize();

	ofstream fout;
	fout.open(outfile);

	int *shortest;

	shortest = (int *) malloc(nv * sizeof(int));
	cudaMemcpy(shortest, shortestv, nv * sizeof(int), cudaMemcpyDeviceToHost);

	for(int i=0; i<nv; i++){
		fout << shortest[i] << " ";
	}
	fout << endl;

	shortest = (int *) malloc(nh * sizeof(int));
	cudaMemcpy(shortest, shortesth, nh * sizeof(int), cudaMemcpyDeviceToHost);

	for(int i=0; i<nh; i++){
		fout << shortest[i] << " ";
	}
	fout << endl;

	fout.close();
}

__global__ void init_index(int n, int *arr){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < n){
		arr[index+1] = index;
	}
}

__global__ void init_float(int n, float *arr, float val){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < n){
		arr[index] = val;
	}
}

__global__ void PageRank_step(int *frontier_in, int *off, int *adj, float *pval_in, float *pval_out){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int cnt = frontier_in[0];
	
	if(index < cnt){
		
		int s = frontier_in[index+1];
		float add_val = pval_in[s] / (off[s+1] - off[s]);
		
		for(int i=0; i<(off[s+1]-off[s]); i++){
			
			int d = adj[off[s] + i];

			atomicAdd(&pval_out[d], add_val);
		}
	}
}

__global__ void PageRank_norm(int n, float *pval, float damp, float addconst){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(index < n){
		pval[index] = damp * pval[index] + addconst;
	}
}

void HyperPageRank(int nv, int mv, int nh, int mh, int source, string outfile, int *offv, int *offh, int *adjv, int *adjh, int maxiter){
	
	float *pvalv;
	float *pvalh;
	cudaMalloc(&pvalv, nv * sizeof(float));
	cudaMalloc(&pvalh, nh * sizeof(float));
	init_float<<<(nv+31)/32, 32>>>(nv, pvalv, 1.0/((float)nv));

	int *frontierv;
	int *frontierh;
	cudaMalloc(&frontierv, (nv + 1) * sizeof(int));
	cudaMalloc(&frontierh, (nh + 1) * sizeof(int));
	init_index<<<(nv+31)/32, 32>>>(nv, frontierv);
	init_index<<<(nh+31)/32, 32>>>(nh, frontierh);

	set_val<<<1,1>>>(frontierh, 0, nh);
	set_val<<<1,1>>>(frontierv, 0, nv);

	float damp = 0.85;
	float addconstv = (1.0 - damp)*(1/(float) nv);
	float addconsth = (1.0 - damp)*(1/(float) nh);

	for(int iter = 0; iter < maxiter; iter++){
		// HyperPageRank main loop
		
		init_float<<<(nh+31)/32, 32>>>(nh, pvalh, 0.0);
		PageRank_step<<<(nv+31)/32, 32>>>(frontierv, offv, adjv, pvalv, pvalh);

		if(debug){
			float *pval = (float *) malloc(nh * sizeof(float));
			cudaMemcpy(pval, pvalh, nh * sizeof(float), cudaMemcpyDeviceToHost);

			cout << "pvalh ";
			for(int i=0; i<nh; i++){
				printf("%.6f ", pval[i]);
			}
			cout << endl;
		}

		init_float<<<(nv+31)/32, 32>>>(nv, pvalv, 0.0);
		PageRank_step<<<(nh+31)/32, 32>>>(frontierh, offh, adjh, pvalh, pvalv);
		PageRank_norm<<<(nv+31)/32, 32>>>(nv, pvalv, damp, addconstv);

		if(debug){
			float *pval = (float *) malloc(nv * sizeof(float));
			cudaMemcpy(pval, pvalv, nv * sizeof(float), cudaMemcpyDeviceToHost);

			cout << "pvalv ";
			for(int i=0; i<nv; i++){
				printf("%.6f ", pval[i]);
			}
			cout << endl;
		}
	}
	cudaDeviceSynchronize();

	ofstream fout;
	fout.open(outfile);

	float *pval = (float *) malloc(nv * sizeof(float));
	cudaMemcpy(pval, pvalv, nv * sizeof(float), cudaMemcpyDeviceToHost);

	for(int i=0; i<nv; i++){
		fout << setprecision(6) << pval[i] << " ";
	}
	fout << endl;

	fout.close();
}

// main code 
int main(int argc, char **argv){

	string algorithm(argv[1]);
	string infile(argv[2]);
	string outfile(argv[3]);

	ifstream fin;
	fin.open(infile);

	// read hypergraph parameters
	string no_use;
	fin >> no_use;
	int nv, mv, nh, mh;
	fin >> nv;
	fin >> mv;
	fin >> nh;
	fin >> mh;

	int *offv = (int *) malloc((nv + 1) * sizeof(int));
	int *offh = (int *) malloc((nh + 1) * sizeof(int));
	int *adjv = (int *) malloc(mv * sizeof(int));
	int *adjh = (int *) malloc(mh * sizeof(int));
	int *wghv = (int *) malloc(mv * sizeof(int));
	int *wghh = (int *) malloc(mh * sizeof(int));
	int *incntv = (int *) malloc(nv * sizeof(int));
	int *incnth = (int *) malloc(nh * sizeof(int));
	
	// read vertex offsets
	for(int i=0; i<nv; i++){
		fin >> offv[i];
	}
	offv[nv] = mv;

	// read vertex adjacency lists
	for(int i=0; i<mv; i++){
		fin >> adjv[i];
		incnth[adjv[i]]++;
	}

	// read vertex weights list
	for(int i=0; i<mv; i++){
		fin >> wghv[i];
	}

	// read hyperedge offsets
	for(int i=0; i<nh; i++){
		fin >> offh[i];
	}
	offh[nh] = mh;

	// read hyperedge adjacency lists
	for(int i=0; i<mh; i++){
		fin >> adjh[i];
		incntv[adjh[i]]++;
	}

	// read hyperedge weights list
	for(int i=0; i<mh; i++){
		fin >> wghh[i];
	}

	fin.close();

	// copy all arrays to GPU
	int *gpu_offv;
	int *gpu_offh;
	int *gpu_adjv;
	int *gpu_adjh;
	int *gpu_wghv;
	int *gpu_wghh;
	int *gpu_incntv;
	int *gpu_incnth;
	
	cudaMalloc(&gpu_offv, (nv + 1) * sizeof(int));
	cudaMalloc(&gpu_offh, (nh + 1) * sizeof(int));
	cudaMalloc(&gpu_adjv, mv * sizeof(int));
	cudaMalloc(&gpu_adjh, mh * sizeof(int));
	cudaMalloc(&gpu_wghv, mv * sizeof(int));
	cudaMalloc(&gpu_wghh, mh * sizeof(int));
	cudaMalloc(&gpu_incntv, nv * sizeof(int));
	cudaMalloc(&gpu_incnth, nh * sizeof(int));
	
	cudaMemcpy(gpu_offv, offv, (nv + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_offh, offh, (nh + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_adjv, adjv, mv * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_adjh, adjh, mh * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_wghv, wghv, mv * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_wghh, wghh, mh * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_incntv, incntv, nv * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_incnth, incnth, nh * sizeof(int), cudaMemcpyHostToDevice);

	// timing variables
	cudaEvent_t start, stop;
	float milliseconds;

	// to avoid first extra time
	useless<<<1,1>>>();
	cudaDeviceSynchronize();

	if(algorithm == "BFS"){
		for(int i=0; i<4; i++){
			milliseconds = 0;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start, 0);

			// Call BFS on HyperGraph
			HyperBFS(nv, mv, nh, mh, 0, outfile, gpu_offv, gpu_offh, gpu_adjv, gpu_adjh);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);
			printf("Time taken by HyperBFS function to execute is: %.6f ms\n", milliseconds);
		}
	}

	if(algorithm == "BPath"){
		for(int i=0; i<4; i++){
			milliseconds = 0;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start, 0);

			// Call BPath on HyperGraph
			HyperBPath(nv, mv, nh, mh, 0, outfile, gpu_offv, gpu_offh, gpu_adjv, gpu_adjh, gpu_incntv, gpu_incnth);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);
			printf("Time taken by HyperBPath function to execute is: %.6f ms\n", milliseconds);
		}
	}

	if(algorithm == "SSSP"){
		for(int i=0; i<4; i++){
			milliseconds = 0;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start, 0);

			// Call SSSP on HyperGraph
			HyperSSSP(nv, mv, nh, mh, 0, outfile, gpu_offv, gpu_offh, gpu_adjv, gpu_adjh, gpu_wghv, gpu_wghh);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);
			printf("Time taken by HyperSSSP function to execute is: %.6f ms\n", milliseconds);
		}
	}

	if(algorithm == "PageRank"){
		for(int i=0; i<4; i++){
			milliseconds = 0;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start, 0);

			// Call BFS on HyperGraph
			HyperPageRank(nv, mv, nh, mh, 0, outfile, gpu_offv, gpu_offh, gpu_adjv, gpu_adjh, 1);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);
			printf("Time taken by HyperPageRank function to execute is: %.6f ms\n", milliseconds);
		}
	}

	return 0;
}