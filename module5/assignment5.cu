//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h> //srand and rand
#include <math.h>

/*
	Profile functions. Taken and modified from https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
*/
void profileCopiesHostToDevice(int        *d_a,
                               int        *h_a,
                               int        *d_b,
				                       int        *h_b,
                               const unsigned int  bytes,
                               const char         *desc){

  // events for timing
  cudaEvent_t startEvent, stopEvent;

  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);

  //start a recording event and execute the transfer after
  cudaEventRecord(startEvent, 0);
  cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);
  cudaEventRecord(stopEvent, 0); //stop
  cudaEventSynchronize(stopEvent);

  float time;
  cudaEventElapsedTime(&time, startEvent, stopEvent);
  printf("\n%s transfers Host to Device Time Elaped: %f ms, Bandwidth (MB/s): %f\n\n",desc, time, bytes * 1e-3 / time);

  // clean up events
  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);
}

void profileCopiesDeviceToHost( int *h_c_add, int *d_c_add, int *h_c_sub, int *d_c_sub,
								                int *h_c_mult, int *d_c_mult, int *h_c_mod, int *d_c_mod,
                                const unsigned int bytes, const char *desc){

  // events for timing
  cudaEvent_t startEvent, stopEvent;

  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);

  cudaEventRecord(startEvent, 0);

  cudaMemcpy( h_c_add, d_c_add, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy( h_c_sub, d_c_sub, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy( h_c_mult, d_c_mult, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy( h_c_mod, d_c_mod, bytes, cudaMemcpyDeviceToHost);

  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent);

  float time;
  cudaEventElapsedTime(&time, startEvent, stopEvent);
  printf("\n%s transfers Device To Host Time Elaped: %f ms, Bandwidth (MB/s): %f\n\n",desc,time, bytes * 1e-3 / time);

  // clean up events
  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);

}


/*
				Arithmetic Functions Using shared Memory
*/
// Add Function
__global__ void add_shared(int *a, int *b, int *c, int n){
    extern __shared__ int res[]; 
    
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
	
    // Make sure we do not go out of bounds
    if (id < n)
        res[threadIdx.x] = a[id] + b[id];

	__syncthreads(); // wait for all threads in the block to finish
	
	c[threadIdx.x] = res[threadIdx.x]; 
}

// subtract function
__global__ void subtract_shared(int *a, int *b, int *c, int n){
   extern __shared__ int res[]; 
    
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
	
    // Make sure we do not go out of bounds
    if (id < n)
        res[threadIdx.x] = a[id] - b[id];

	__syncthreads(); // wait for all threads in the block to finish
	
	c[threadIdx.x] = res[threadIdx.x]; 
        
}

// multiply function
 __global__ void mult_shared(int *a, int *b, int *c, int n){
    extern __shared__ int res[]; 
     // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
	
    // Make sure we do not go out of bounds
    if (id < n)
        res[threadIdx.x] = a[id] * b[id];

	__syncthreads(); // wait for all threads in the block to finish
	
	c[threadIdx.x] = res[threadIdx.x]; 
        
}

// Moudulus function
__global__ void mod_shared(int *a, int *b, int *c, int n){
 extern __shared__ int res[]; 
    
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
	
    // Make sure we do not go out of bounds
    if (id < n)
        res[threadIdx.x] = a[id] % b[id];

	__syncthreads(); // wait for all threads in the block to finish
	
	c[threadIdx.x] = res[threadIdx.x]; 
}


/*
				Arithmetic Functions Using Constant Memory
*/
// Add Function
__global__ void add_const(int *a, int *b, int *c, int n){
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}

// subtract function
__global__ void subtract_const(int *a, int *b, int *c, int n){
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] - b[id];
}

// multiply function
 __global__ void mult_const(int *a, int *b, int *c, int n){
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] * b[id];
}

// Moudulus function
__global__ void mod_const(int *a, int *b, int *c, int n){

    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] % b[id];
}






// Constant Memory Implementation
void execute_arithmetic_constMem(int totalThreads, int numBlocks){


	printf("\t\t*****Executing Arithmetic Functions Using Constant Memory*****\n");
		// Host input vectors
	int *h_a, *h_b;
	//Host output vectors for different functions "h_c_func"
	int *h_c_add,*h_c_sub,*h_c_mult,*h_c_mod;

	// Device input vectors
	int *d_a, *d_b;
	//Device output vector
	int *d_c_add,*d_c_sub,*d_c_mult,*d_c_mod;

	// Size, in bytes, of each vector
	const unsigned int bytes = totalThreads*sizeof(int);

	// Allocate memory for each vector on host Pinned
	cudaMallocHost((void**)&h_a, bytes);
	cudaMallocHost((void**)&h_b, bytes);
	cudaMallocHost((void**)&h_c_add, bytes);
	cudaMallocHost((void**)&h_c_sub, bytes);
	cudaMallocHost((void**)&h_c_mult, bytes);
	cudaMallocHost((void**)&h_c_mod, bytes);

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


  //printf the first 7 elements of input arrays
  printf("Array 1: ");
  for(int i = 0; i<7; i++){
   printf("%d ", h_a[i]);
  }
  printf("\nArray 2: ");
  for(int i = 0; i<7; i++){
    printf("%d ", h_b[i]);
  }
  printf("\n\n");


	//copy both input arrays from host to device and profile it (see profileCopiesHostToDevice)
	profileCopiesHostToDevice(d_a, h_a, d_b, h_b, bytes, "Pinned");

	//Perform Arithmetic Functions and print time elapsed
	//performing add function
	printf("  Performing Add function\n");
	add_const<<<numBlocks, totalThreads>>>(d_a, d_b, d_c_add, totalThreads);
        cudaDeviceSynchronize();
	//performing subtract function
	printf("  Performing subtract function\n");
	subtract_const<<<numBlocks, totalThreads>>>(d_a, d_b, d_c_sub, totalThreads);
        cudaDeviceSynchronize();
	//performing mult function
	printf("  Performing mult function\n");
	mult_const<<<numBlocks, totalThreads>>>(d_a, d_b, d_c_mult, totalThreads);
        cudaDeviceSynchronize();
	//performing mod fuction
	printf("  Performing mod function\n");
	mod_const<<<numBlocks, totalThreads>>>(d_a, d_b, d_c_mod, totalThreads);
        cudaDeviceSynchronize();
	//copy the output arrays from device to host
	profileCopiesDeviceToHost(h_c_add,d_c_add,h_c_sub, d_c_sub,h_c_mult, d_c_mult,h_c_mod, d_c_mod, bytes, "Pinned");

  // printf the first 7 elements of the results
  printf("Arithmetic Results: \n");
	printf("Add: ");
	for(int i = 0; i<7; i++){
		printf("%d ", h_c_add[i]);
	}
  printf("\nSubtract: ");

	for(int i = 0; i<7; i++){
		printf("%d ", h_c_sub[i]);
	}
  printf("\nMultiply: ");
	for(int i = 0; i<7; i++){
		printf("%d ", h_c_mult[i]);
	}
  printf("\nMultiply: ");
	for(int i = 0; i<7; i++){
		printf("%d ", h_c_mod[i]);
	}
	printf("\n\n");

	//free up space on our GPU
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c_add);
	cudaFree(d_c_sub);
	cudaFree(d_c_mult);
	cudaFree(d_c_add);

	//free up space on our CPU use cudaFreeHost since pinnned
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c_add);
	cudaFreeHost(h_c_sub);
	cudaFreeHost(h_c_mult);
	cudaFreeHost(h_c_mod);

}

void perform_add_shared(int numBlocks, int totalThreads, int *d_a, int *d_b, int *d_c_add){
	
  float time;
  // events for timing
  cudaEvent_t startEvent, stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);
	
	//start a recording event and execute the Kernels after
  cudaEventRecord(startEvent, 0);
  //performing add function
  printf("  Performing Add function...");
  add_shared<<<numBlocks, totalThreads, totalThreads*sizeof(int)>>>(d_a, d_b, d_c_add, totalThreads);
  cudaDeviceSynchronize();
  cudaEventRecord(stopEvent, 0); //stop
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&time, startEvent, stopEvent);
  printf(" Elapsed Time: %f\n", time);
   // clean up events
  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);
}


//Shared Memory Implementation
void execute_arithmetic_sharedMem(int totalThreads, int numBlocks){

printf("\t\t*****Executing Arithmetic Functions Using Shared Memory*****\n");

	// Host input vectors
	int *h_a, *h_b;
	//Host output vectors for different functions "h_c_func"
	int *h_c_add,*h_c_sub,*h_c_mult,*h_c_mod;

	// Device input vectors
	int *d_a, *d_b;
	//Device output vector
	int *d_c_add,*d_c_sub,*d_c_mult,*d_c_mod;

	// Size, in bytes, of each vector
	const unsigned int bytes = totalThreads*sizeof(int);

	// Allocate memory for each vector on host Pinned
	cudaMallocHost((void**)&h_a, bytes);
	cudaMallocHost((void**)&h_b, bytes);
	cudaMallocHost((void**)&h_c_add, bytes);
	cudaMallocHost((void**)&h_c_sub, bytes);
	cudaMallocHost((void**)&h_c_mult, bytes);
	cudaMallocHost((void**)&h_c_mod, bytes);

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


  //printf the first 7 elements of input arrays
  printf("Array 1: ");
  for(int i = 0; i<7; i++){
   printf("%d ", h_a[i]);
  }
  printf("\nArray 2: ");
  for(int i = 0; i<7; i++){
    printf("%d ", h_b[i]);
  }
  printf("\n\n");


	//copy both input arrays from host to device and profile it (see profileCopiesHostToDevice)
	profileCopiesHostToDevice(d_a, h_a, d_b, h_b, bytes, "Pinned");


    
    perform_add_shared(numBlocks, totalThreads, d_a, d_b, d_c_add);
    


  
	//performing subtract function
	printf("  Performing subtract function\n");
	subtract_shared<<<numBlocks, totalThreads, totalThreads*sizeof(int)>>>(d_a, d_b, d_c_sub, totalThreads);
	cudaDeviceSynchronize();
	//performing mult function
	printf("  Performing mult function\n");
	mult_shared<<<numBlocks, totalThreads, totalThreads*sizeof(int)>>>(d_a, d_b, d_c_mult, totalThreads);
	cudaDeviceSynchronize();
	//performing mod fuction
	printf("  Performing mod function\n");
	mod_shared<<<numBlocks, totalThreads, totalThreads*sizeof(int)>>>(d_a, d_b, d_c_mod, totalThreads);
	cudaDeviceSynchronize();

	//copy the output arrays from device to host
	profileCopiesDeviceToHost(h_c_add,d_c_add,h_c_sub, d_c_sub,h_c_mult, d_c_mult,h_c_mod, d_c_mod, bytes, "Pinned");

  




  // printf the first 7 elements of the results
  printf("Arithmetic Results: \n");
	printf("Add: ");
	for(int i = 0; i<7; i++){
		printf("%d ", h_c_add[i]);
	}
  printf("\nSubtract: ");

	for(int i = 0; i<7; i++){
		printf("%d ", h_c_sub[i]);
	}
  printf("\nMultiply: ");
	for(int i = 0; i<7; i++){
		printf("%d ", h_c_mult[i]);
	}
  printf("\nMultiply: ");
	for(int i = 0; i<7; i++){
		printf("%d ", h_c_mod[i]);
	}
	printf("\n\n");

	//free up space on our GPU
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c_add);
	cudaFree(d_c_sub);
	cudaFree(d_c_mult);
	cudaFree(d_c_add);

	//free up space on our CPU use cudaFreeHost since pinnned
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c_add);
	cudaFreeHost(h_c_sub);
	cudaFreeHost(h_c_mult);
	cudaFreeHost(h_c_mod);
}



int main(int argc, char** argv)
{

	int totalThreads = (1 << 10);
	int blockSize = 256;

      //User wants to run the Global vs Pinned Examples
      if( argc > 2 && argc < 4){
        // Ensure the user supplies both number of threads and block size
        // otherwise use default values
        totalThreads = atoi(argv[1]);
        blockSize = atoi(argv[2]);
      }

    	int numBlocks = totalThreads/blockSize;

      printf("\nUsing %d Threads and %d BlockSize\n",totalThreads, blockSize);

    	// validate command line arguments
    	if (totalThreads % blockSize != 0) {
    		++numBlocks;
    		totalThreads = numBlocks*blockSize;

    		printf("Warning: Total thread count is not evenly divisible by the block size\n");
    		printf("The total number of threads will be rounded up to %d\n", totalThreads);
    	}

    	// Lets see what we are working with and calculate the Amount of data we are transfering
    	cudaDeviceProp prop;
    	cudaGetDeviceProperties(&prop,0);
    	const unsigned int bytes = totalThreads*sizeof(int);
    	printf("\nDevice: %s\n", prop.name);
    	printf("Transfer size (MB): %d\n\n", bytes * bytes / totalThreads);


    	//Execute Pageable Arithmetic
    	execute_arithmetic_sharedMem(totalThreads, numBlocks);
      //Execute The Pinned Arithmetic
      //execute_arithmetic_constMem(totalThreads, numBlocks);

  

  return 0;
}
