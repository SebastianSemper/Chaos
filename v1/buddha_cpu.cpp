#include "buddha_cpu.h"

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
	while(succSamples < samplesCPUThread){
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

void doCPU(
	float *res
){
	std::vector<std::thread> threads;
	for (int i = CPUThreads; i--; ){
		threads.push_back(std::thread(iterateCPU,res));
	}
	for (auto& t : threads){
		t.join();
	}
}