//include CUDA - just store the visited positions
//use half all the way
//call cuda after constructing threads
//also calc all starting points within the kernel

#include <iostream>
#include <ImfRgbaFile.h>
#include <ImfNamespace.h>
#include <half.h>
#include <string>
#include <thread>
#include <random>
#include <vector>
#include <cmath>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define outputSize 2000

#define samplesPerThread 50
#define numThreads 8

#define maxCalcIter 60000
#define minCalcIter 40000

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
	OPENEXR_IMF_NAMESPACE::Rgba *p = new OPENEXR_IMF_NAMESPACE::Rgba[size*size];
			
	//norm everything into [0,1]
	float maxData = 0;
	float minData = 99999;
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

void iterateGPU(
	float *res
){
	//alloc mem for ints
	unsigned int *apple_D;
	gpuErrchk(
		cudaMalloc((void **) &apple_D,N*N*sizeof(int))
	);
	//iterate and fill ints
	
	//draw it into the res
	
	cudaMalloc()
}

void iterateCPU(
	float *res
){
	//generate starting point
	std::random_device generator;
	std::uniform_real_distribution<double> distribution(-1.5,1.5);
	
	double ReC,ImC;
	double ReS,ImS,ReS_;
	unsigned int steps;
	
	unsigned int visitedPos[maxCalcIter];
	
	unsigned int succSamples(0);
	
	//iterate over it
	while(succSamples < samplesPerThread){
		steps = 0;
		ReC = distribution(generator)-0.4;
		ImC = distribution(generator);
		double p(sqrt((ReC-0.25)*(ReC-0.25) + ImC*ImC));
		while (( ((ReC+1)*(ReC+1) + ImC*ImC) < 0.0625) || (ReC < p - 2*p*p + 0.25)){
			ReC = distribution(generator)-0.4;
			ImC = distribution(generator);
			p = sqrt((ReC-0.25)*(ReC-0.25) + ImC*ImC);
		}
		ReS = ReC;
		ImS = ImC;
		for (unsigned int j = maxCalcIter; (ReS*ReS + ImS*ImS < 4)&&(j--); steps++ ){
			ReS_ = ReS;
			ReS *= ReS;
			ReS += ReC - ImS*ImS;
			ImS *= 2*ReS_;
			ImS += ImC;
			if ((ReS+0.5)*(ReS+0.5) + ImS*ImS < 4){
				visitedPos[steps] = int((ReS+2.5)*0.25*outputSize)*outputSize + int((ImS+2)*0.25*outputSize);
				
			}
		}
		if ((steps > minCalcIter)&&(ReS*ReS + ImS*ImS > 4)){
			succSamples++;
			std::cout << succSamples << " - " << ReC << "-" << ImC   << std::endl;
			for (int j = 0; j<steps;j++){
				//std::cout << visitedPos[j] << std::endl;
				res[visitedPos[j]]++;
			}
		}
	}
}

int main(){
	float *outImg = new float[outputSize*outputSize];
	
	std::vector<std::thread> threads;
	for (int i = numThreads; i--; ){
		threads.push_back(std::thread(iterateCPU,outImg));
	}
	for (auto& t : threads){
		t.join();
	}
	
	writeExr(outputSize,outImg,"buddha_cpu.exr");
	
	delete(outImg);
}