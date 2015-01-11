#include <iostream>
#include <ImfRgbaFile.h>
#include <ImfNamespace.h>
#include <half.h>
#include <string>
#include <thread>
#include <random>
#include <vector>
#include <cmath>

#include "buddha_cpu.h"
#include "buddha_gpu.h"
#include "consts.h"

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

int main(){
	float *outImg = new float[outputSize*outputSize];
	
	std::thread cpuThread(doCPU,outImg);
	
	//std::thread gpuThread(doGPU,outImg);
	
	//gpuThread.join();
	cpuThread.join();
	
	writeExr(outputSize,outImg,"output.exr");
	
	delete(outImg);
}