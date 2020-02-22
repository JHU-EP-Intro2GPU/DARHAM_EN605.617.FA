//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h> //srand and rand
#include <math.h>

// add function d
__global__ void add(int *a, int *b, int *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}

__global__ void subtract(int *a, int *b, int *c, int n){
                // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] - b[id];
}

 __global__ void mult(int *a, int *b, int *c, int n){
            // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] * b[id];
}

__global__ void mod(int *a, int *b, int *c, int n){

    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] % b[id];
}



int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;

	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads/blockSize;

    printf("Using %d Threads and %d BlockSize\n",totalThreads, blockSize); 
    
    
	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;

		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

    // Host input vectors
    int *h_a, *h_b;
    //Host output vectors for different functions "h_c_func"
    int *h_c_add,*h_c_sub,*h_c_mult,*h_c_mod;

    // Device input vectors
    int *d_a, *d_b;
    //Device output vector
    int *d_c_add,*d_c_sub,*d_c_mult,*d_c_mod;

    // Size, in bytes, of each vector
    size_t bytes = totalThreads*sizeof(int);

    // Allocate memory for each vector on host
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c_add = (int*)malloc(bytes);
    h_c_sub = (int*)malloc(bytes);
    h_c_mult = (int*)malloc(bytes);
    h_c_mod = (int*)malloc(bytes);

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c_add, bytes);
    cudaMalloc(&d_c_sub, bytes);
    cudaMalloc(&d_c_mult, bytes);
    cudaMalloc(&d_c_mod, bytes);


    //initialize the input vectors
    for(int i = 0;i<totalThreads;i++){
      //first array is 0 through number of threads
		  h_a[i] = i;
      // second array is a random number between 0 and 3
			h_b[i] = rand() % 4;
    }


	//copy both input arrays from host to device
  cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);


	//performing add function
  printf("Performing Add function\n");
  add<<<numBlocks, totalThreads>>>(d_a, d_b, d_c_add, totalThreads);

  //performing subtract function
	printf("Performing subtract function\n");
	subtract<<<numBlocks, totalThreads>>>(d_a, d_b, d_c_sub, totalThreads);

	//performing mult function
  printf("Performing mult function\n");
	mult<<<numBlocks, totalThreads>>>(d_a, d_b, d_c_mult, totalThreads);

	//performing mod fuction
  printf("Performing mod function\n");
  mod<<<numBlocks, totalThreads>>>(d_a, d_b, d_c_mod, totalThreads);

	//copy the output arrays from device to host
  cudaMemcpy( h_c_add, d_c_add, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy( h_c_sub, d_c_sub, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy( h_c_mult, d_c_mult, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy( h_c_mod, d_c_mod, bytes, cudaMemcpyDeviceToHost);


  //free up space on our GPU
  cudaFree(d_c_add);
  cudaFree(d_c_sub);
  cudaFree(d_c_mult);
  cudaFree(d_c_add);

  return 0;
}
