//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h> //srand and rand
#include <math.h>

#define MAXSTRLEN 1024
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
  printf("\n%s transfers Host to Device Time Elaped: %f ms, Bandwidth (MB/s): %f\n\n",desc, time*1e3, bytes * 1e-3 / time);

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
  printf("\n%s transfers Device To Host Time Elaped: %f ms, Bandwidth (MB/s): %f\n\n",desc,time*1e3, bytes * 1e-3 / time);

  // clean up events
  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);

}


/*
				Arithmetic Functions
*/
// Add Function
__global__ void add(int *a, int *b, int *c, int n){
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}

// subtract function
__global__ void subtract(int *a, int *b, int *c, int n){
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] - b[id];
}

// multiply function
 __global__ void mult(int *a, int *b, int *c, int n){
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] * b[id];
}

// Moudulus function
__global__ void mod(int *a, int *b, int *c, int n){

    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] % b[id];
}


/*
		Caeser Excercise
*/

__global__ void caeser( char* message, int shift){

	// Get our global thread ID
  int id = blockIdx.x*blockDim.x+threadIdx.x;

	if( message[id] >= 'a' && message[id] <= 'z'){
    // handle lower case
		message[id] = message[id] + shift; // shift by the amount

		if( message[id] > 'z')
			message[id] = message[id] - 'z' + 'a' - 1; //wrap around if we hit the end of the alphabet

    if( message[id] < 'a')
      message[id] = message[id] + 'z' - 'a' + 1; //for decryption, wrap around if we hit the behind letter a

	}
	else if(message[id] >= 'A' && message[id] <= 'Z'){
    //handle upper case
		message[id] = message[id] + shift;

		if( message[id] > 'Z')
			message[id] = message[id] - 'Z' + 'A' - 1; //wrap around if we hit the end of the alphabet

    if( message[id] < 'A')
      message[id] = message[id] + 'Z' - 'A' + 1;
		}
}


// Pageable Memory Implementation
void execute_arithmetic_pageable(int totalThreads, int numBlocks){


	printf("\t\t*****Executing Arithmetic Functions Using Pageable Memory*****\n");
	// read command line arguments
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

	// Allocate memory for each vector on host Pageable
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

    //printf("Made It here line 164\n");
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


	//copy both input arrays from host to device and profile it
	profileCopiesHostToDevice(d_a, h_a, d_b, h_b, bytes, "Pageable");


	//performing add function
	printf("  Performing Add function\n");
	add<<<numBlocks, totalThreads>>>(d_a, d_b, d_c_add, totalThreads);
  cudaDeviceSynchronize();
	//performing subtract function
	printf("  Performing subtract function\n");
	subtract<<<numBlocks, totalThreads>>>(d_a, d_b, d_c_sub, totalThreads);
  cudaDeviceSynchronize();
	//performing mult function
	printf("  Performing mult function\n");
	mult<<<numBlocks, totalThreads>>>(d_a, d_b, d_c_mult, totalThreads);
  cudaDeviceSynchronize();
	//performing mod fuction
	printf("  Performing mod function\n");
	mod<<<numBlocks, totalThreads>>>(d_a, d_b, d_c_mod, totalThreads);
  cudaDeviceSynchronize();
	//copy the output arrays from device to host and profile it
	profileCopiesDeviceToHost(h_c_add,d_c_add,h_c_sub, d_c_sub,h_c_mult, d_c_mult,h_c_mod, d_c_mod, bytes, "Pageable");
  cudaDeviceSynchronize();

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

	//free up space on our CPU
	free(h_a);
	free(h_b);
	free(h_c_add);
	free(h_c_sub);
	free(h_c_mult);
	free(h_c_mod);

}

void execute_arithmetic_pinned(int totalThreads, int numBlocks){

printf("\t\t*****Executing Arithmetic Functions Using Pinned Memory*****\n");

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

	//performing add function
	printf("  Performing Add function\n");
	add<<<numBlocks, totalThreads>>>(d_a, d_b, d_c_add, totalThreads);
  cudaDeviceSynchronize();
	//performing subtract function
	printf("  Performing subtract function\n");
	subtract<<<numBlocks, totalThreads>>>(d_a, d_b, d_c_sub, totalThreads);
  cudaDeviceSynchronize();
	//performing mult function
	printf("  Performing mult function\n");
	mult<<<numBlocks, totalThreads>>>(d_a, d_b, d_c_mult, totalThreads);
  cudaDeviceSynchronize();
	//performing mod fuction
	printf("  Performing mod function\n");
	mod<<<numBlocks, totalThreads>>>(d_a, d_b, d_c_mod, totalThreads);
  cudaDeviceSynchronize();
	//copy the output arrays from device to host
	profileCopiesDeviceToHost(h_c_add,d_c_add,h_c_sub, d_c_sub,h_c_mult, d_c_mult,h_c_mod, d_c_mod, bytes, "Pinned");
  cudaDeviceSynchronize();

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


void execute_caesar_cypher(void){

  printf("\n\t\t*****Executing Caeser Cypher*****\n");

  //Hardcoded value for string
	char h_str[MAXSTRLEN] = "Hello World, My name is Computer";
  printf("\nStarting off with string: %s\n", h_str);

  //shift by this amount
  int shift = 2;
  printf("Using Shift: %d\n", shift);

  //device character array
  char *d_str;
  //allocate the string array on the device
  cudaMalloc((void** )&d_str, MAXSTRLEN*sizeof(char));
  cudaMemcpy(d_str, h_str, MAXSTRLEN*sizeof(char), cudaMemcpyHostToDevice);

  //use a dim3 data type and calculate the amount of blocks
  dim3 block(256);
	dim3 grid((MAXSTRLEN+block.x-1)/block.x);

  //invoke the kernal
  caeser<<<grid,block>>>(d_str, shift);

  //copy from device to host
  cudaMemcpy(h_str, d_str, MAXSTRLEN*sizeof(char), cudaMemcpyDeviceToHost);

  printf("After Caeser Cypher     : %s\n\n", h_str);

  //lets add the additive inverse to get the original message (decryption)
  printf("Apply Additive Inverse  : %d\n", -shift);

  caeser<<<grid,block>>>(d_str, -shift);

  cudaMemcpy(h_str, d_str, MAXSTRLEN*sizeof(char), cudaMemcpyDeviceToHost);

  printf("Original String         : %s\n",h_str );

  cudaFree(d_str);

}


int main(int argc, char** argv)
{

	int totalThreads = (1 << 10);
	int blockSize = 256;




	if (argc == 2) {
    // Using a flag -caeser to see if the user wants to run the Caeser Cypher Example
    if (strcmp(argv[1], "-caeser") == 0)
      execute_caesar_cypher();
    else
      printf("Wrong Arguments. Options: -caeser | NumberThreads BlockSize\n");

  }else{
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
    	execute_arithmetic_pageable(totalThreads, numBlocks);
      //Execute The Pinned Arithmetic
      execute_arithmetic_pinned(totalThreads, numBlocks);

  }

  return 0;
}
