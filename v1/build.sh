#/bin/bash
g++ -std=c++11 -O3 -I/usr/include/OpenEXR/ -I/usr/local/cuda/include buddha_cpu.cpp -c -o buddha_cpu.o
g++ -std=c++11 -O3 -I/usr/include/OpenEXR/ -I/usr/local/cuda/include main.cpp -c -o main.o

nvcc -ccbin=g++-4.7 -O3 -use_fast_math -I/usr/local/cuda/include -L/usr/lib64/ buddha_gpu.cu -c -o buddha_gpu.o 

nvcc -O3 -use_fast_math -I/usr/include/OpenEXR/ -I/usr/local/cuda/include -L/usr/lib64/ -lpthread -lHalf -lIlmImf -lm -lgomp -lcuda -lcudart  *.o  -o main