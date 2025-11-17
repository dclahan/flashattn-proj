CUDA Attention
=== GPU Information ===
Name: NVIDIA GeForce RTX 2080 Ti
Compute Capability: 7.5
Global Memory: 10.5683 GB
Shared Memory per Block: 48 KB
Registers per Block: 65536
Warp Size: 32
Max Threads per Block: 1024
Max Threads per SM: 1024
Number of SMs: 68
=================================

=================================
Configuration: B=32, nh=16, N=64, d=64
Total Q size: 8 MB

Generating '/tmp/nsys-report-5389.qdstrm'
[1/8] [========================100%] report1.nsys-rep
[2/8] [========================100%] report1.sqlite
[3/8] Executing 'nvtx_sum' stats report
SKIPPED: /home/dgc9773/gpu/proj/flashattn-proj/src/report1.sqlite does not contain NV Tools Extension (NVTX) data.
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)    Min (ns)   Max (ns)    StdDev (ns)            Name         
 --------  ---------------  ---------  ------------  ------------  --------  -----------  ------------  ----------------------
     73.8      655,012,069         40  16,375,301.7  10,057,635.0   132,660  178,010,000  30,337,160.8  poll                  
     25.6      227,623,007        691     329,411.0      18,522.0     1,014   84,242,469   3,236,896.3  ioctl                 
      0.2        1,796,864         31      57,963.4      13,021.0     7,585    1,146,087     202,553.8  mmap64                
      0.1          933,988          3     311,329.3     217,966.0   159,520      556,502     214,327.3  pthread_create        
      0.1          487,999         56       8,714.3       8,681.5     4,047       17,016       2,524.2  open64                
      0.1          474,889          3     158,296.3       4,355.0     2,855      467,679     267,934.3  fwrite                
      0.1          464,231         43      10,796.1       7,392.0     2,058       43,310       8,962.0  fopen                 
      0.0          278,941         10      27,894.1      27,601.0    21,851       35,168       4,268.9  sem_timedwait         
      0.0          214,810         16      13,425.6       9,135.5     1,754       64,751      14,927.1  mmap                  
      0.0          120,269         36       3,340.8       2,834.0     1,199       15,208       2,414.4  fclose                
      0.0           61,520         52       1,183.1       1,125.0     1,000        2,427         260.3  fcntl                 
      0.0           61,205          1      61,205.0      61,205.0    61,205       61,205           0.0  fgets                 
      0.0           39,312          6       6,552.0       6,897.0     2,805        9,131       2,524.8  open                  
      0.0           38,921         12       3,243.4       3,263.0     2,381        4,402         586.7  write                 
      0.0           35,912          6       5,985.3       6,204.0     3,677        8,172       1,857.5  munmap                
      0.0           33,800          3      11,266.7       9,037.0     6,662       18,101       6,036.7  pipe2                 
      0.0           32,905          2      16,452.5      16,452.5    10,410       22,495       8,545.4  socket                
      0.0           27,136          2      13,568.0      13,568.0     8,855       18,281       6,665.2  fread                 
      0.0           25,934          8       3,241.8       1,186.5     1,001       13,289       4,394.2  putc                  
      0.0           25,542         14       1,824.4       1,481.5     1,041        3,624         812.0  read                  
      0.0           14,275          1      14,275.0      14,275.0    14,275       14,275           0.0  connect               
      0.0            3,474          1       3,474.0       3,474.0     3,474        3,474           0.0  bind                  
      0.0            3,070          2       1,535.0       1,535.0     1,504        1,566          43.8  pthread_cond_broadcast
      0.0            2,269          2       1,134.5       1,134.5     1,063        1,206         101.1  dup                   
      0.0            1,611          1       1,611.0       1,611.0     1,611        1,611           0.0  listen                

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)    Max (ns)    StdDev (ns)            Name         
 --------  ---------------  ---------  ------------  ------------  ----------  -----------  ------------  ----------------------
     93.3      233,303,426          8  29,162,928.3     159,677.0       4,752  232,332,942  82,093,100.9  cudaMalloc            
      5.0       12,455,332          1  12,455,332.0  12,455,332.0  12,455,332   12,455,332           0.0  cudaDeviceSynchronize 
      1.2        3,022,126          4     755,531.5     985,402.5      27,194    1,024,127     486,468.4  cudaMemcpy            
      0.4        1,115,920          6     185,986.7     206,978.0       6,968      288,797     115,575.2  cudaFree              
      0.1          180,337          1     180,337.0     180,337.0     180,337      180,337           0.0  cudaLaunchKernel      
      0.0           81,677          3      27,225.7      10,483.0       5,807       65,387      33,131.3  cudaMemset            
      0.0            1,526          1       1,526.0       1,526.0       1,526        1,526           0.0  cuModuleGetLoadingMode

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ----------------------------------------------------------------------------------------------------
    100.0       12,454,378          1  12,454,378.0  12,454,378.0  12,454,378  12,454,378          0.0  flash_attn_forward_kernel(const float *, const float *, const float *, int, int, int, int, int, int…

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)           Operation          
 --------  ---------------  -----  ---------  ---------  --------  --------  -----------  ----------------------------
     99.4        2,726,042      4  681,510.5  900,798.0    13,984   910,462    445,042.0  [CUDA memcpy Host-to-Device]
      0.6           17,728      3    5,909.3    1,856.0     1,824    14,048      7,048.3  [CUDA memset]               

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)           Operation          
 ----------  -----  --------  --------  --------  --------  -----------  ----------------------------
     25.297      4     6.324     8.389     0.131     8.389        4.129  [CUDA memcpy Host-to-Device]
      8.651      3     2.884     0.131     0.131     8.389        4.767  [CUDA memset]          