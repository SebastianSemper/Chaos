#include "buddha_gpu.h"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void initRandom(curandState *state){
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < GPUThreads){
		curand_init((unsigned int) clock64(),2*idx,0,&state[idx]);
	}
}

__global__ void iterateGPU(
	float *indices,
	curandState *state
){
	//position in problem
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
		
	//check if within problem bounds
	if (idx < GPUThreads){
		int succSamples(0);
		while(succSamples < samplesGPUThread){
			unsigned int steps(0);
			unsigned int tries(0);
			
			float ReC(2.0*curand_uniform(&state[idx])-1.5);
			float ImC(2.0*curand_uniform(&state[idx])-1.0);
			float p(sqrtf((ReC-0.25)*(ReC-0.25) + ImC*ImC));
			
			while(( ((ReC+1)*(ReC+1) + ImC*ImC) < 0.0625) || (ReC < p - 2*p*p + 0.25) ){
				ReC = 2.0*curand_uniform(&state[idx])-1.5;
				ImC = 2.0*curand_uniform(&state[idx])-1.0;
				p = sqrtf((ReC-0.25)*(ReC-0.25) + ImC*ImC);
				tries++;
				if (tries > 10){
					printf("thread %d tried %d times. \n",idx,tries);
					indices[idx*samplesGPUThread*(maxCalcIter+1) + succSamples*(maxCalcIter+1)] = 0;
					for (int i = succSamples; i < samplesGPUThread; i++){
						indices[idx*samplesGPUThread*(maxCalcIter+1) + i*(maxCalcIter+1)] = 0;
					}
					return;
				}
			}
			
			/*
			indices[idx*samplesGPUThread*(maxCalcIter+1) + succSamples*(maxCalcIter+1)] = maxCalcIter;//int(0.25*(ReC + 2.5)*(float)outputSize)*outputSize + int(0.25*(ImC + 2.0)*(float)outputSize);
			
			for (int i = 1; i <= maxCalcIter; i++){
				indices[idx*samplesGPUThread*(maxCalcIter+1) + succSamples*(maxCalcIter+1) + i] = idx;
			}
			
			//printf("used %d steps for (%f,%f) -  index: %d\n",k,ReC,ImC,indices[idx*samplesPerThread*(maxCI+1) + succSamples*(maxCI+1)]);
			succSamples++;
			*/
// 			
			
			
			//initialize values for sequence
			float ReZ(ReC);
			float ImZ(ImC);
			float ReZ_;
			
			//iterate until maximum iterations reached or sequence diverges for sure
			for (unsigned int i = maxCalcIter; (i--)&&(ReZ*ReZ + ImZ*ImZ < 4);steps++){
				//get next values in sequence
				ReZ_ = ReZ;
				ReZ *= ReZ;
				ReZ += ReC - ImZ*ImZ;
				
				ImZ *= 2*ReZ_;
				ImZ += ImC;
				
				//save index of visited entries in the grid
				//but first check if within bounds
				if ((ReZ+0.5)*(ReZ+0.5) + ImZ*ImZ < 4){
					indices[idx*samplesGPUThread*(maxCalcIter+1) + succSamples*(maxCalcIter+1) + (steps+1)] = int(0.25*(ReZ + 2.5)*(float)outputSize)*outputSize + int(0.25*(ImZ + 2.0)*(float)outputSize);
				} else {
					indices[idx*samplesGPUThread*(maxCalcIter+1) + succSamples*(maxCalcIter+1) + (steps+1)] = -1;
				}
				
			}
			
			
			//check if point diverged and spent at least the minimum iterations in the set
			if ((ReZ*ReZ + ImZ*ImZ >= 4)&&(steps >= float(minCalcIter))){
				indices[idx*samplesGPUThread*(maxCalcIter+1) + succSamples*(maxCalcIter+1)] = float(steps);
				printf("steps: %d,%d,\n",int(indices[idx*samplesGPUThread*(maxCalcIter+1) + succSamples*(maxCalcIter+1)]),tries);
				succSamples++;
			}
			
		}
		
	}
	
}

void doGPU(
	float *res
){
	dim3 blockSize(16);
	dim3 gridSize(float(GPUThreads)/float(blockSize.x) + 1);
	
	//alloc mem for ints
	float *indices_D;
	gpuErrchk(
		cudaMalloc((void **) &indices_D,GPUThreads*samplesGPUThread*(maxCalcIter+1)*sizeof(float))
	);
	
	float *indices_H;
	gpuErrchk(
		cudaMallocHost((void **) &indices_H,GPUThreads*samplesGPUThread*(maxCalcIter+1)*sizeof(float))
	);
	
	curandState *state;
	gpuErrchk(
		cudaMalloc((void **) &state,GPUThreads*sizeof(curandState))
	);
	
	size_t free,total;
	cudaMemGetInfo(&free,&total);
	
	printf("free %d - total %d \n",free/1024/1024,total/1024/1024);
	
	initRandom <<< gridSize,blockSize >>> (state);
	iterateGPU <<< gridSize,blockSize >>> (indices_D,state);
	
	cudaMemcpy(indices_H, indices_D, GPUThreads*samplesGPUThread*(maxCalcIter+1)*sizeof(float) , cudaMemcpyDeviceToHost);
	
	
	for (int i = 0; i < GPUThreads*samplesGPUThread; i++){
		
		//res[int(indices_H[i*(maxCalcIter+1)])]++;
		
		//printf("s: %d\n",int(indices_H[i*(maxCalcIter+1)]));
		printf("%d,",int(indices_H[i*(maxCalcIter+1)]));
		for (int j = 1; j < int(indices_H[i*(maxCalcIter+1)]);j++){
			float ind = indices_H[i*(maxCalcIter+1) + j];
			if (ind != -1){
				res[int(ind)]++;
			}
			//printf("%d,",int(indices_H[i*(maxCalcIter+1) + j]));
		}
		printf("\n------\n");
	}
	
	cudaFree(indices_D);
	cudaFreeHost(indices_H);
	
	//iterate and fill ints
	
	//draw it into the res
	
}