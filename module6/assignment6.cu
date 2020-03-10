/*******************************************************************************

This program compares all types of memory learned so far and compares their
Performance

Author: Said Darham
*******************************************************************************/

//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h> //srand and rand
#include <math.h>

// Constant data declaration
#define WORKSIZE 1024 // define a default worksize for constant data

// Define different memroy types
#define GLOBAL 1
#define SHARED 2
#define CONSTANT 3
#define REGISTER 4

// Arrays for constant memory on device
__device__ __constant__ int d_a_const[WORKSIZE];
__device__ __constant__ int d_b_const[WORKSIZE];




/*******************************************************************************

                            Print Array and Data Info

*******************************************************************************/
void printArray(int *array, int n, const char* funcDesc){
    //helper function to Print the array of n elements and what function is used
    printf("\n%s: ", funcDesc);
    for(int i = 0; i<n; i++){
      printf("%d ", array[i]);
    }
}
void printDataInfo(int totalThreads){
  //print device info as well as the amount of data for a thread size
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop,0);
  const unsigned int bytes = totalThreads*sizeof(int);
  printf("\nDevice: %s\n", prop.name);
  printf("Transfer size (MB): %d\n\n", bytes * bytes / totalThreads);
}


/*******************************************************************************

        PROFILE DATA TRANSFER FUNCTIONS (Host2Device and Device2Host)

*******************************************************************************/
void profileCopiesHostToDevice(int        *d_a,
                               int        *h_a,
                               int        *d_b,
				                       int        *h_b,
                               const unsigned int  bytes,
                               const char         *desc,
                               const int    memType){

  // events for timing
  cudaEvent_t startEvent, stopEvent;
  // create an event
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);

  //start a recording event and execute the transfer after
  cudaEventRecord(startEvent, 0);

  // Use either cudaMemcpy or cudaMemcpyToSymbol depending on memory TYPE
  if(memType != CONSTANT){
    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);
  }else if(memType == CONSTANT){
    cudaMemcpyToSymbol( d_a_const, h_a, bytes,0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol( d_b_const, h_b, bytes,0, cudaMemcpyHostToDevice);
  }

  cudaEventRecord(stopEvent, 0); //stop
  cudaEventSynchronize(stopEvent);
  //get the elapsed time and print info
  float time;
  cudaEventElapsedTime(&time, startEvent, stopEvent);
  printf("\n%s Transfers Host to Device Time Elaped: %f ms, Bandwidth (MB/s): %f\n\n", desc, time, bytes * 1e-3 / time);

  // clean up events
  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);
}

void profileCopiesDeviceToHost( int *h_c_add, int *d_c_add, int *h_c_sub, int *d_c_sub,
								                int *h_c_mult, int *d_c_mult, int *h_c_mod, int *d_c_mod,
                                const unsigned int bytes, const char *desc){


  // events for timing
  cudaEvent_t startEvent, stopEvent;
  // create an event
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);
  //start a recording event and execute the transfer after
  cudaEventRecord(startEvent, 0);

  cudaMemcpy( h_c_add, d_c_add, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy( h_c_sub, d_c_sub, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy( h_c_mult, d_c_mult, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy( h_c_mod, d_c_mod, bytes, cudaMemcpyDeviceToHost);

  cudaEventRecord(stopEvent, 0);//stop
  cudaEventSynchronize(stopEvent);
  //get the elapsed time and print info
  float time;
  cudaEventElapsedTime(&time, startEvent, stopEvent);
  printf("\n%s transfers Device To Host Time Elaped: %f ms, Bandwidth (MB/s): %f\n",desc,time, bytes * 1e-3 / time);

  // clean up events
  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);
}




/*******************************************************************************

        ARITHMETIC KERNEL FUNCTIONS (GLOBAL, SHARED, CONSTANT, REGISTER)

*******************************************************************************/
// Add Functions
__global__ void add(int *a, int *b, int *c, int n){
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}
__global__ void addShared(int *a, int *b, int *c, int n){
    extern __shared__ int res[]; //using shared memory
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    // Make sure we do not go out of bounds
    if (id < n)
        res[threadIdx.x] = a[id] + b[id];

	__syncthreads(); // wait for all threads in the block to finish
	c[threadIdx.x] = res[threadIdx.x];//since threads from different blocks cannot talk, use thread index instead
}
__global__ void addConst( int *c, int n){
    // Get our global thread ID
    const unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    // use the constant data declared
    if (id < n)
        c[id] = d_a_const[id] + d_b_const[id];
}
__global__ void addReg(int *a, int *b, int *c, int n){
  // Get our global thread ID
  const unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;
  if (id < n){
    int tempA, tempB, tempC;//use variables in register
    tempA = a[id];
    tempB = b[id];
    tempC = tempA + tempB;
    c[id] = tempC;
  }
}

// subtract functions
__global__ void subtract(int *a, int *b, int *c, int n){
                // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] - b[id];
}
__global__ void subtractShared(int *a, int *b, int *c, int n){
   extern __shared__ int res[]; //using shared memory

    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        res[threadIdx.x] = a[id] - b[id];

	__syncthreads(); // wait for all threads in the block to finish

	c[threadIdx.x] = res[threadIdx.x];

}
__global__ void subtractConst(int *c, int n){
    // Get our global thread ID
    const unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = d_a_const[id] - d_b_const[id];
}
__global__ void subtractReg(int *a, int *b, int *c, int n){
  // Get our global thread ID
  const unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;
  if (id < n){
    int tempA, tempB, tempC;//use variables in register
    tempA = a[id];
    tempB = b[id];
    tempC = tempA - tempB;
    c[id] = tempC;
  }
}

// multiply functions
__global__ void mult(int *a, int *b, int *c, int n){
           // Get our global thread ID
   int id = blockIdx.x*blockDim.x+threadIdx.x;

   // Make sure we do not go out of bounds
   if (id < n)
       c[id] = a[id] * b[id];
}
__global__ void multShared(int *a, int *b, int *c, int n){
    extern __shared__ int res[]; //using shared memory
     // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        res[threadIdx.x] = a[id] * b[id];

	__syncthreads(); // wait for all threads in the block to finish

	c[threadIdx.x] = res[threadIdx.x];

}
__global__ void multConst(int *c, int n){
   // Get our global thread ID
   const unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;

   // Make sure we do not go out of bounds
   if (id < n)
       c[id] = d_a_const[id] * d_b_const[id];
}
__global__ void multReg(int *a, int *b, int *c, int n){
  // Get our global thread ID
  const unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;
  if (id < n){
    int tempA, tempB, tempC;//use variables in register
    tempA = a[id];
    tempB = b[id];
    tempC = tempA * tempB;
    c[id] = tempC;
  }
}

// Moudulu functions
__global__ void mod(int *a, int *b, int *c, int n){

    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] % b[id];
}
__global__ void modShared(int *a, int *b, int *c, int n){
 extern __shared__ int res[];

    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        res[threadIdx.x] = a[id] % b[id];

	__syncthreads(); // wait for all threads in the block to finish

	c[threadIdx.x] = res[threadIdx.x];
}
__global__ void modConst(int *c, int n){

    // Get our global thread ID
    const unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = d_a_const[id] % d_b_const[id];
}
__global__ void modReg(int *a, int *b, int *c, int n){
  // Get our global thread ID
  const unsigned int id = blockIdx.x*blockDim.x+threadIdx.x;
  if (id < n){
    int tempA, tempB, tempC;
    tempA = a[id]; //use variables in register
    tempB = b[id];
    tempC = tempA % tempB;
    c[id] = tempC;
  }
}


/*******************************************************************************

        EXECUTION AND PROFILING OF ARITHMETIC FUNCTIONS

*******************************************************************************/
void executeGlobalFunctions(int numBlocks, int totalThreads,
                            int *d_a, int *d_b, int *d_c_add,
                            int *d_c_sub,int *d_c_mult,int *d_c_mod){

      float time;
      // events for timing
      cudaEvent_t startEvent, stopEvent;
      cudaEventCreate(&startEvent);
      cudaEventCreate(&stopEvent);

      //start a recording event and execute the Kernels after
      cudaEventRecord(startEvent, 0);
      //invoking global memory kernels
      add<<<numBlocks, totalThreads, totalThreads*sizeof(int)>>>(d_a, d_b, d_c_add, totalThreads);
      subtract<<<numBlocks, totalThreads, totalThreads*sizeof(int)>>>(d_a, d_b, d_c_sub, totalThreads);
      mult<<<numBlocks, totalThreads, totalThreads*sizeof(int)>>>(d_a, d_b, d_c_mult, totalThreads);
      mod<<<numBlocks, totalThreads, totalThreads*sizeof(int)>>>(d_a, d_b, d_c_mod, totalThreads);

      cudaEventRecord(stopEvent, 0); //stop
      cudaEventSynchronize(stopEvent);
      //get the elapsed time and print info
      cudaEventElapsedTime(&time, startEvent, stopEvent);
      printf("Global Memory Functions  Elapsed Time: %f\n", time);
       // clean up events
      cudaEventDestroy(startEvent);
      cudaEventDestroy(stopEvent);

}

void executeSharedFunctions(int numBlocks, int totalThreads,
                            int *d_a, int *d_b, int *d_c_add, int *d_c_sub, int *d_c_mult, int *d_c_mod){

      float time;
      // events for timing
      cudaEvent_t startEvent, stopEvent;
      cudaEventCreate(&startEvent);
      cudaEventCreate(&stopEvent);

      //start a recording event and execute the Kernels after
      cudaEventRecord(startEvent, 0);
      //invoking shared memory kernels
      addShared<<<numBlocks, totalThreads, totalThreads*sizeof(int)>>>(d_a, d_b, d_c_add, totalThreads);
      subtractShared<<<numBlocks, totalThreads, totalThreads*sizeof(int)>>>(d_a, d_b, d_c_sub, totalThreads);
      multShared<<<numBlocks, totalThreads, totalThreads*sizeof(int)>>>(d_a, d_b, d_c_mult, totalThreads);
      modShared<<<numBlocks, totalThreads, totalThreads*sizeof(int)>>>(d_a, d_b, d_c_mod, totalThreads);

      cudaEventRecord(stopEvent, 0); //stop
      cudaEventSynchronize(stopEvent);
      cudaEventElapsedTime(&time, startEvent, stopEvent);
      printf("Shared Memory Functions Elapsed Time: %f\n", time);
       // clean up events
      cudaEventDestroy(startEvent);
      cudaEventDestroy(stopEvent);
}

void executeConstantFunctions(int numBlocks, int totalThreads,
                              int *d_c_add, int *d_c_sub,
                              int *d_c_mult, int *d_c_mod){

      float time;
      // events for timing
      cudaEvent_t startEvent, stopEvent;
      cudaEventCreate(&startEvent);
      cudaEventCreate(&stopEvent);

      //start a recording event and execute the Kernels after
      cudaEventRecord(startEvent, 0);
      //invoking constant memory kernels
      addConst<<<numBlocks, totalThreads, totalThreads*sizeof(int)>>>(d_c_add, totalThreads);
      subtractConst<<<numBlocks, totalThreads, totalThreads*sizeof(int)>>>(d_c_sub, totalThreads);
      multConst<<<numBlocks, totalThreads, totalThreads*sizeof(int)>>>( d_c_mult, totalThreads);
      modConst<<<numBlocks, totalThreads, totalThreads*sizeof(int)>>>(d_c_mod, totalThreads);

      cudaEventRecord(stopEvent, 0); //stop
      cudaEventSynchronize(stopEvent);
      cudaEventElapsedTime(&time, startEvent, stopEvent);
      printf("Constant Memory Functions Elapsed Time: %f\n", time);
       // clean up events
      cudaEventDestroy(startEvent);
      cudaEventDestroy(stopEvent);


}
void executeRegisterFunctions(int numBlocks, int totalThreads,
                            int *d_a, int *d_b, int *d_c_add,
                            int *d_c_sub,int *d_c_mult,int *d_c_mod){

      float time;
      // events for timing
      cudaEvent_t startEvent, stopEvent;
      cudaEventCreate(&startEvent);
      cudaEventCreate(&stopEvent);

      //start a recording event and execute the Kernels after
      cudaEventRecord(startEvent, 0);

      //invoke kernels using register memory
      addReg<<<numBlocks, totalThreads, totalThreads*sizeof(int)>>>(d_a, d_b, d_c_add, totalThreads);
      subtractReg<<<numBlocks, totalThreads, totalThreads*sizeof(int)>>>(d_a, d_b, d_c_sub, totalThreads);
      multReg<<<numBlocks, totalThreads, totalThreads*sizeof(int)>>>( d_a, d_b, d_c_mult, totalThreads);
      modReg<<<numBlocks, totalThreads, totalThreads*sizeof(int)>>>(d_a, d_b, d_c_mod, totalThreads);

      cudaEventRecord(stopEvent, 0); //stop
      cudaEventSynchronize(stopEvent);
      cudaEventElapsedTime(&time, startEvent, stopEvent);
      printf("Register Memory Functions Elapsed Time: %f\n", time);
       // clean up events
      cudaEventDestroy(startEvent);
      cudaEventDestroy(stopEvent);
}




/*******************************************************************************

        EXECUTION AND PROFILING OF EACH TYPE OF DEVICE MEMORY USED

*******************************************************************************/
void gpuTest(int numBlocks, int totalThreads, const int memoryType){


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

	// Allocate memory for each intput and output vector on GPU
  if(memoryType != CONSTANT){
    //do not need these vectors for constant memory
    cudaMalloc(&d_a, bytes);
  	cudaMalloc(&d_b, bytes);
  }
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

	//For a selected memory type, copy data form host to device, invoke the kernels and copy data from device to host
  switch(memoryType){
    case GLOBAL:
      printf("\t\t*****Executing Arithmetic Functions Using Global Memory*****\n");
      profileCopiesHostToDevice(d_a, h_a, d_b, h_b, bytes, "Global", GLOBAL);
      executeGlobalFunctions(numBlocks, totalThreads,d_a,d_b,d_c_add,d_c_sub,d_c_mult,d_c_mod);
      profileCopiesDeviceToHost(h_c_add,d_c_add,h_c_sub, d_c_sub,h_c_mult, d_c_mult,h_c_mod, d_c_mod, bytes,"Shared");
      break;
    case SHARED:
      printf("\t\t*****Executing Arithmetic Functions Using Shared Memory*****\n");
      profileCopiesHostToDevice(d_a, h_a, d_b, h_b, bytes, "Shared", SHARED);
      executeSharedFunctions(numBlocks, totalThreads,d_a,d_b,d_c_add,d_c_sub,d_c_mult,d_c_mod);
      profileCopiesDeviceToHost(h_c_add,d_c_add,h_c_sub, d_c_sub,h_c_mult, d_c_mult,h_c_mod, d_c_mod, bytes,"Shared");
      break;
    case CONSTANT:
      printf("\t\t*****Executing Arithmetic Functions Using Constant Memory*****\n");
      profileCopiesHostToDevice(d_a_const, h_a, d_b_const, h_b, bytes, "Constant",CONSTANT);
      executeConstantFunctions(numBlocks, totalThreads,d_c_add,d_c_sub,d_c_mult,d_c_mod);
      profileCopiesDeviceToHost(h_c_add,d_c_add,h_c_sub, d_c_sub,h_c_mult, d_c_mult,h_c_mod, d_c_mod, bytes,"Constant");
      break;
    case REGISTER:
      printf("\t\t*****Executing Arithmetic Functions Using Register Memory*****\n");
      profileCopiesHostToDevice(d_a, h_a, d_b, h_b, bytes, "Register", REGISTER);
      executeRegisterFunctions(numBlocks, totalThreads,d_a,d_b,d_c_add,d_c_sub,d_c_mult,d_c_mod);
      profileCopiesDeviceToHost(h_c_add,d_c_add,h_c_sub, d_c_sub,h_c_mult, d_c_mult,h_c_mod, d_c_mod, bytes,"Register");
      break;
    default:
      printf("Unknown Memory Type!\n");
      break;
  }

  // print the first 7 elements of the inputs and results
  printArray(h_a, 7,"Array 1");
  printArray(h_b, 7,"Array 2");
  printf("\n");
  printArray(h_c_add, 7, "Add Results");
  printArray(h_c_sub, 7, "Subtract Results");
  printArray(h_c_mult, 7, "Multiply Results");
  printArray(h_c_mod, 7, "Modulus Results");
  printf("\n\n");

	//free up space on  GPU
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


/*******************************************************************************

                            MAIN

*******************************************************************************/
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

  // Print device and data transfer info
  printDataInfo(totalThreads);

  // execute global memory test
  gpuTest(numBlocks, totalThreads, GLOBAL);
  // execute shared memory test
  gpuTest(numBlocks, totalThreads, SHARED);
  // execute constant memory test
  gpuTest(numBlocks, totalThreads, CONSTANT);
  // execute register memory test
  gpuTest(numBlocks, totalThreads, REGISTER);

  return 0;
}
