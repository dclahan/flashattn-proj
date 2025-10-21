#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

// implement scaled dot product attention (softmax(Q @ K^T * softmax_scale) @ V)

// const int Bc = min(ceil(prop.sharedMemPerBlock/sizeof(float)/(4*d)), (float)N);

__global__