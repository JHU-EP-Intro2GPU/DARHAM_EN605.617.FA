/*******************************************************************************

This program uses the Thrust library to perform vector arithmetic .

Author: Said Darham
*******************************************************************************/

#include <iostream>
#include <stdlib.h> //srand and rand
#include <math.h>

//Thrust libraries headers
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>


//Timer struct declaration. Using CUDA EVENTS
typedef struct timer{
  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;
  float time_ms;
} timerEvent;

/*******************************************************************************

                        PROFILER FUNCTIONS USING EVENTS

*******************************************************************************/
void startEventTimer(timerEvent *timer){
  /* startEventTimer()
     Creates and starts recording an event
  */
  cudaEventCreate(&timer->startEvent);
  cudaEventCreate(&timer->stopEvent);
  cudaEventRecord(timer->startEvent);
}

void stopEventTimer(timerEvent *timer){
  /* stopEventTimer()
     Stops an event and calculates the elapsed time between start and stop event
  */
  cudaEventRecord(timer->stopEvent);
  cudaEventSynchronize(timer->stopEvent);
  cudaEventElapsedTime(&timer->time_ms, timer->startEvent, timer->stopEvent);

}
void freeEventTimer(timerEvent *timer){
  /*  freeEventTimer()
      cleans up the events
  */
  cudaEventDestroy(timer->startEvent);
  cudaEventDestroy(timer->stopEvent);
}

void checkDevices(void){
  //Check and print devices name
  cudaDeviceProp prop;
  int deviceCount; //number of devices found
  int devId = 0; // default device Id

  cudaGetDeviceCount(&deviceCount);

  if(deviceCount == 0){
    std::cout << "No GPU Device Found\n";
    exit(0);
  }else if(deviceCount == 1){
    cudaSetDevice(devId); //set the device 0 as default
  }

  std::cout << "Number Of Devices Found: " << deviceCount << std::endl;
  //Print device names and some basic associated properties
  for (int i = 0; i<deviceCount; i++){
    cudaGetDeviceProperties(&prop,i);
    std::cout << "Device " << i << " Name: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
  }
}

void printArray(thrust::host_vector<int> array, int n){
    //helper function to Print the array of n elements and what function is used
    for(int i = 0; i<10; i++){
      std::cout << array[i] << ' ';
    }
    std::cout << std::endl;
}


/*******************************************************************************

ARITHMETIC KERNEL FUNCTIONS

*******************************************************************************/
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
// Moudulu function
__global__ void mod(int *a, int *b, int *c, int n){
  // Get our global thread ID
  int id = blockIdx.x*blockDim.x+threadIdx.x;

  // Make sure we do not go out of bounds
  if (id < n)
  c[id] = a[id] % b[id];
}

__host__ static __inline__ int myRand(){
  //to be used with thrust's generate() to supply with random numbers b/w 0-3
  return ((int)rand() % 4);
}



/*******************************************************************************

CUDA KERNELS TEST

*******************************************************************************/
void executeCudaTest(int numBlocks, int blockSize, int totalThreads){

  std::cout << "\n\t\t*****Executing Arithmetic Functions Using CUDA kernels*****" << std::endl;

  // Host input/output vectors
  int *h_a, *h_b, *h_c_add,*h_c_sub,*h_c_mult,*h_c_mod;

  // Device input/output vectors
  int *d_a, *d_b, *d_c_add,*d_c_sub,*d_c_mult,*d_c_mod;

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

  //create a struct which will contain info for timing using events
  timerEvent timer;

  //Transfer and Profile data from host to device and profile using EVENTS
  startEventTimer(&timer);
  cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);

  //Execute  the kernel arithmetic functions
  add<<<numBlocks, blockSize>>>(d_a, d_b, d_c_add, totalThreads);
  subtract<<<numBlocks, blockSize>>>(d_a, d_b, d_c_sub, totalThreads);
  mult<<<numBlocks, blockSize>>>(d_a, d_b, d_c_mult, totalThreads);
  mod<<<numBlocks, blockSize>>>(d_a, d_b, d_c_mod, totalThreads);

  //Transfer data from device to host
  cudaMemcpy(h_c_add, d_c_add, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_c_sub, d_c_sub, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_c_mult, d_c_mult, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_c_mod, d_c_mod, bytes, cudaMemcpyDeviceToHost);
  stopEventTimer(&timer);

  std::cout << "Time Elaplsed For CUDA kernels: " << timer.time_ms << " ms" << std::endl;

  //destroy Event timer
  freeEventTimer(&timer);

  //free up space on our GPU
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c_add);
	cudaFree(d_c_sub);
	cudaFree(d_c_mult);
	cudaFree(d_c_mod);

	//free up space on our CPU use cudaFreeHost since pinnned
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c_add);
	cudaFreeHost(h_c_sub);
	cudaFreeHost(h_c_mult);
	cudaFreeHost(h_c_mod);
}


/*******************************************************************************

THRUST TEST

*******************************************************************************/
void executeThrustTest(int totalThreads){

  std::cout << "\n\t\t*****Executing Arithmetic Functions Using Thrust*****" << std::endl;

  // host vectors
  thrust::host_vector<int> h_a(totalThreads);
  thrust::host_vector<int> h_b(totalThreads);

  // device vectors
  thrust::device_vector<int> d_c_add(totalThreads);
  thrust::device_vector<int> d_c_sub(totalThreads);
  thrust::device_vector<int> d_c_mult(totalThreads);
  thrust::device_vector<int> d_c_mod(totalThreads);

  //Initialize data arrays
  for( int i = 0; i < totalThreads; i++)
    h_a[i] = i;

  //generate random data on the host_vector
  thrust::generate(h_b.begin(), h_b.end(), myRand);

  //create a struct which will contain info for timing using events
  timerEvent timer;
  startEventTimer(&timer);

  //copy vectors from host to devices
  thrust::device_vector<int> d_a = h_a;
  thrust::device_vector<int> d_b = h_b;

  //perform arithmetic functions
  thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_c_add.begin(), thrust::plus<int>());
  thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_c_sub.begin(), thrust::minus<int>());
  thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_c_mult.begin(), thrust::multiplies<int>());
  thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_c_mod.begin(), thrust::modulus<int>());

  //copy results from device to host
  thrust::host_vector<int> h_c_add = d_c_add;
  thrust::host_vector<int> h_c_sub = d_c_sub;
  thrust::host_vector<int> h_c_mult = d_c_mult;
  thrust::host_vector<int> h_c_mod = d_c_mod;

  stopEventTimer(&timer);

  std::cout << "Time Elaplsed For Arithmetic using Thrust: " << timer.time_ms << " ms" << std::endl;

  //destroy Event timer
  freeEventTimer(&timer);
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
  std::cout << "\nUsing " << totalThreads << " Threads and " << blockSize << " BlockSize\n" ;

  // validate command line arguments
  if (totalThreads % blockSize != 0) {
    ++numBlocks;
    totalThreads = numBlocks*blockSize;
    std::cout << "Warning: Total thread count is not evenly divisible by the block size\n";
    std::cout << "The total number of threads will be rounded up to %d\n";
  }

  // get number of devices and print some basic properties
  checkDevices();

  executeCudaTest( numBlocks, blockSize, totalThreads);

  executeThrustTest( totalThreads );

  return 0;
}
