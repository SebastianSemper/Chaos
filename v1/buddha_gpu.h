#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "consts.h"


void doGPU(
	float *res
);