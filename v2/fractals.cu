#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <omp.h>
#include <ImfRgbaFile.h>
#include <ImfNamespace.h>
#include <half.h>
#include <string>

//how many mandelbrot iterations
#define mandelMaxIter 4096
//pan and zoom within the set
#define mandelZoom 0.00001
#define mandelSx -0.77568377
#define mandelSy 0.13646737
//maximum iterations for a starting point
#define buddhaMaxIter 30000
//minimum iterations for a starting point
#define buddhaMinIter 25000
//samples to generate for each kernel call
#define buddhaIterSamples 2048000
//how many iterations to spend for interior checking
#define buddhaTestIter 20000


//CUDA Error fetching
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void writeExr(
	unsigned int size,			//image size
	float *vals,		//data to write
	std::string path	//path to write in to
){
	//allocate array for pixels
	OPENEXR_IMF_NAMESPACE::Rgba *p;
	gpuErrchk(
		cudaMallocHost((void **) &p,size*size*sizeof(OPENEXR_IMF_NAMESPACE::Rgba))
	);
	
	//norm everything into [0,1]
	float maxData = -99999;
	float minData =  99999;
	for (unsigned int i = size*size; i-- ;){
		if (vals[i] > maxData){
			maxData = vals[i];
		}
		if (vals[i] < minData){
			minData = vals[i];
		}
	}
	
	//write all the values into the pixels
	for (unsigned int i = size*size; i--;){
		p[i].r = (-minData + vals[i])/(maxData-minData);
		p[i].g = p[i].r;
		p[i].b = p[i].r;
		p[i].a = 1;
	}
	
	//construct outputfile
	OPENEXR_IMF_NAMESPACE::RgbaOutputFile file (path.c_str(), size, size, OPENEXR_IMF_NAMESPACE::WRITE_RGBA);
	
	//write it out
	file.setFrameBuffer (p, 1, size);
	file.writePixels (size);
}

//Test if CUDA is properly installed
bool cudaTest(){
	int devCuda = 0;
	cudaSetDevice(devCuda);
	cudaDeviceProp propCuda;
	if (cudaGetDeviceProperties(&propCuda, devCuda) == cudaSuccess)
	{
		printf("Using your %s with %dMB VRAM, %d multi processors at %dMHz clock speed \n       and %d KB shared memory per block. \n",
			propCuda.name,
			(int)propCuda.totalGlobalMem/(1024*1024),
			(int)propCuda.multiProcessorCount,
			(int)propCuda.clockRate/1000,
			(int)propCuda.sharedMemPerBlock/1024
		);
		return 1;
	}
	else {
		printf("-ERRO- You don't seem to have a CUDA enabled Device! Aborting...\n");
		return 0;
	}
}

//Kernel for the Mandelbrot set
__global__ void calcMandelbrot(
	float *res,		//final result
	unsigned int gridSize	//size of the final image
	
){
	//get position in problem
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int idy = blockIdx.y*blockDim.y + threadIdx.y;
	
	//check if within the bounds of the image
	if ((idx < gridSize) && (idy < gridSize)){
		//get the starting value with the position in the grid
		float ReC(mandelZoom*2.0*(float(idx)/float(gridSize) - 0.5)+mandelSx);
		float ImC(mandelZoom*2.0*(float(idy)/float(gridSize) - 0.5)+mandelSy);
		
		//declare local result
		float localRes = 0;
		
		//iteration values
		float ReZneu = 0, ReZalt = ReC, ImZneu = 0, ImZalt = ImC;
		
		//iterate until maximum iterations reached or sequence diverges for sure
		for (unsigned int i = mandelMaxIter; (i--)&&(ReZalt*ReZalt + ImZalt*ImZalt < 4);){
			//calculate next element in sequence
			ReZneu = ReC + ReZalt*ReZalt - ImZalt*ImZalt;
			ImZneu = ImC + 2*ReZalt*ImZalt;
			ImZalt = ImZneu;
			ReZalt = ReZneu;
			
			//count the iterations
			localRes++;
		}
		
		//finally add number of iterations
		res[gridSize*idx + idy] = localRes;
	}
}

//wrapper function that calls the mandelbrot kernel
void doMandelbrot(
	unsigned int N			//grid size
){
	printf("Starting Mandelbrot of size %dx%d and %d iterations.\n",N,N,mandelMaxIter);
	
	//how many threads should be in one block?
	dim3 blockSize(32,16);
	
	//how many blocks do we need?
	dim3 gridSize(float(N)/float(blockSize.x) + 1,float(N)/float(blockSize.y) + 1);
	
	//allocate memory on device
	float *apple_D;
	gpuErrchk(
		cudaMalloc((void **) &apple_D,N*N*sizeof(float))
	);
	
	//allocate memory on host
	float *apple_H;
	gpuErrchk(
		cudaMallocHost((void **) &apple_H,N*N*sizeof(float))
	);
	
	//start time measurement
	float startTime = omp_get_wtime();
	
	//call the kernel
	calcMandelbrot <<< gridSize,blockSize >>> (apple_D, N);
	
	//copy back all the memory
	cudaMemcpy(apple_H, apple_D, N*N*sizeof(float) , cudaMemcpyDeviceToHost);
	float timeDiff = ((float) (omp_get_wtime() - startTime));
	printf("Finished with Mandelbrot. Calculation took %fs.\nNow saving plot...",timeDiff);
	
	//save the resulting image
	writeExr(N,apple_H,"mandelbrot.exr");
	
	//I can haz some fr33 mmry plz?
	cudaFree(apple_D);
	cudaFreeHost(apple_H);
	
	printf(" Done!\n");
}

//initialize some states for random number generation
__global__ void initRandom(curandState *state){
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < buddhaIterSamples){
		curand_init(1234,idx,0,&state[idx]);
	}
}

//fill an array with uniformly distributed floats
__global__ void genSamples(
	curandState *state,
	float *Re,
	float *Im
){
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < buddhaIterSamples){
		float Re_;
		float Im_;
		float p;
		float ReT,ImT,ReT_;
		while(1){
			Re_ = 2*curand_uniform(&state[idx])-1.5;
			Im_ = 2*curand_uniform(&state[idx])-1;
			
			//check if it is in the bulbs
			p = sqrtf((Re_-0.25)*(Re_-0.25) + Im_*Im_);			
			if (( ((Re_+1)*(Re_+1) + Im_*Im_) > 0.0625) && (Re_ > p - 2*p*p + 0.25)){
				ReT = Re_;
				ImT = Im_;
				//if yes do some iterations to check if it is near the set
				for (unsigned int i = buddhaTestIter; (i--)&&(ReT*ReT + ImT*ImT < 4); ){
					ReT_ = ReT;
					ReT *= ReT;
					ReT += Re_ - ImT*ImT;
					ImT *= 2*ReT_;
					ImT += Im_;
				}
				//if yes return those values
				if (ReT*ReT + ImT*ImT < 4){
					Re[idx] = Re_;
					Im[idx] = Im_;
					return;
				}
			}
		}
		
	}
}

//buddabrot kernel
/*
 * In contrast to the mandelbrot kernel we do not call a kernel for 
 * each pixel but for each random starting point
 */

__global__ void calcBuddhabrot(
	float *res,			//resulting image
	float *Re,			//real parts of the starting samples
	float *Im,			//imaginary parts of the starting samples
	unsigned int gridSize		//size of the result
	
){
	//position in problem
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
		
	//check if within problem bounds
	if (idx < buddhaIterSamples){
		//allocate memory to store points we been to during iteration
		unsigned int visited[buddhaMaxIter];
		
		//initialize step counter
		unsigned int steps = 0;
		
		//start with the sampled values
		float ReC(Re[idx]);
		float ImC(Im[idx]);
		
		//initialize values for sequence
		float ReZ(ReC);
		float ImZ(ImC);
		float ReZ_;
		
		//iterate until maximum iterations reached or sequence diverges for sure
		for (unsigned int i = buddhaMaxIter; (i--)&&(ReZ*ReZ + ImZ*ImZ < 4);){
			//get next values in sequence
			ReZ_ = ReZ;
			ReZ *= ReZ;
			ReZ += ReC - ImZ*ImZ;
			
			ImZ *= 2*ReZ_;
			ImZ += ImC;
			
			//save index of visited entries in the grid
			//but first check if within bounds
			if ((0.8*abs(ReZ+0.25) < 1)&&(0.8*abs(ImZ) < 1)){
				visited[steps] = int(0.5*(0.8*ReZ + 1.25)*(float)gridSize)*gridSize + int(0.5*(0.8*ImZ + 1.0)*(float)gridSize);
				steps++;
			}
			
		}
		
		 
		//check if point diverged and spent at least the minimum iterations in the set
		if ((ReZ*ReZ + ImZ*ImZ >= 4)&&(steps >= buddhaMinIter)){
			//add something to the visited entries in the grid
			for (unsigned int i = buddhaMinIter; i--;){
				res[visited[i]]++;
			}
			for (unsigned int i = buddhaMinIter; i < steps; i++){
				res[visited[i]]++;
			}
		}
		
	}
	
}

//wrapper function for the buddabrot calculation
void doBuddhabrot(
	unsigned int size,
	unsigned int iterCount
){
	printf("Starting Buddhabrot of size %dx%d and %d samples.\n",size,size,iterCount*buddhaIterSamples);
	
	//allocate memory for the result on the device
	float *buddha_D;
	gpuErrchk(
		cudaMalloc((void **) &buddha_D,size*size*sizeof(float))
	);
	
	//allocate memory for the result on the host
	float *buddha_H;
	gpuErrchk(
		cudaMallocHost((void **) &buddha_H,size*size*sizeof(float))
	);
		
	//allocate memory for the random starting points on the device only
	float *Re_D;
	gpuErrchk(
		cudaMalloc((void **) &Re_D,buddhaIterSamples*sizeof(float))
	);
	float *Im_D;
	gpuErrchk(
		cudaMalloc((void **) &Im_D,buddhaIterSamples*sizeof(float))
	);
	
	//allocate memory for the random states
	curandState *state;
	gpuErrchk(
		cudaMalloc((void **) &state,buddhaIterSamples*sizeof(curandState))
	);
	
	//set block size - 1D this time - see kernel why
	dim3 blockSize(256);
	dim3 gridSize(float(buddhaIterSamples)/float(blockSize.x) + 1);
	
	//initialize the random states
	initRandom <<< gridSize,blockSize >>> (state);
	
	//start time measurement and set starting array to zero
	float startTime = omp_get_wtime();
	for (unsigned int i = 0; i < size*size; i++){
		buddha_H[i] = 0;
	}
	
	//push it to the device
	cudaMemcpy(buddha_D, buddha_H, size*size*sizeof(float) , cudaMemcpyHostToDevice);
	
	//to save some memory we repeat the random sampling several times with fixed sample count
	//and just repeat this as many times as wanted
	for (unsigned int i = iterCount; i--;){
		printf("Step: %d of %d\n",i+1,iterCount);
		genSamples <<< gridSize,blockSize >>> (state,Re_D,Im_D);
		calcBuddhabrot <<< gridSize,blockSize >>> (buddha_D, Re_D, Im_D, size);
	}
	
	//get result from device
	cudaMemcpy(buddha_H, buddha_D, size*size*sizeof(float) , cudaMemcpyDeviceToHost);
	float timeDiff = ((float) (omp_get_wtime() - startTime));
	printf("Finished with Buddhabrot. Calculation took %fs.\nNow saving plot...",timeDiff);
	
	//plot result
	//cairoVisPlot(buddha_H,size,size,size,size,"buddhabrot.png",1);
	writeExr(size,buddha_H,"buddhabrot.exr");
	
	//I can haz some fr33 mmry plz?
	cudaFree(buddha_D);
	cudaFreeHost(buddha_H);
	cudaFree(state);
	cudaFree(Re_D);
	cudaFree(Im_D);
	
	printf(" Done!\n");
}

int main(){
	
	
	//check if CUDA is present
	if(cudaTest()){
		
		/*
		 * start the mandelbrot calculation - see how far you can go with the resolution
		 */
		//doMandelbrot(5000);
		
		/*
		 * start the buddhabrot calculation
		 * but bear in mind that doubling the side length of the image means that you need about
		 * four times the amount of starting samples to create an equally dense result!
		 * Happy rendering!
		 */
		//doBuddhabrot(6000,300);
		doBuddhabrot(1000,5);
	}
	
	return 0;
}