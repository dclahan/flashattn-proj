#ifndef FLASH_ATTN_H
#define FLASH_ATTN_H

float* flash_forward(float* Q, float* K, float* V, int B, int nh, int N, int d);
// float* flash_forward(float*, float*, float*, int, int , int, int);

#endif
