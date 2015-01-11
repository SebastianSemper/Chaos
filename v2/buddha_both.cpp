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

#include <buddha_cpu.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>




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



int main(){
	float *outImg = new float[outputSize*outputSize];
	
	
	
	writeExr(outputSize,outImg,"buddha_cpu.exr");
	
	delete(outImg);
}