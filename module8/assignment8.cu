/*******************************************************************************

This program uses two libraries from the CUDA toolkit "cuFFT" and "cuRand"

executeCudaRNG() routine generates a normally distributed random number arrays

executeCudaFFT() routine gives an example on how to use the cuFFT library to get the
Frequency spectrum of a two tone signal and see the frequency components of the
time domain signal

Author: Said Darham
*******************************************************************************/
#include <iostream>
#include <stdlib.h> //srand and rand
#include <math.h>
#include <iomanip> //for setting float precision

#include <curand.h>
#include <curand_kernel.h>//cuRand header files
#include <cufft.h>//cuFFT

#define SEED 1234
#define MAXLEN 1000
#define SAMPLERATE 500
#define PI 3.14159265358979323846

//Timer struct declaration. Using CUDA EVENTS
typedef struct timer{
  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;
  float time_ms;
} timerEvent;

typedef float2 Complex;

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

/*******************************************************************************

Helper Functions

*******************************************************************************/
void printArray(float *array, const int n){
    //helper function to print the array of n elements
    for(int i = 0; i<n; i++){
      std::cout << std::fixed << std::setprecision(4) << array[i] << "\n";
    }
    std::cout << std::endl;
}

void printComplexArray( Complex *array, const int n){
  //helper function to Print a complex array of n elements
  for(int i = 0; i < 50; i++) //CHANGE from 50 to n
    std::cout << std::fixed << std::setprecision(4) << array[i].x << " " << array[i].y << "i" << std::endl;

}

void generateSignal( Complex *array, const int n){
  //helper function to initialize a complex baseband signal containing 2 frequencies 5 and 10 Hz
  float samplePeriod = (float)1 / (float)SAMPLERATE; //sampling period of digital signal
  for(int iSample = 0; iSample < n; iSample++){
    array[iSample].x = 0.7*cos(2*PI*5*iSample*samplePeriod) + cos(2*PI*10*iSample*samplePeriod);
    array[iSample].y = 0.7*sin(2*PI*5*iSample*samplePeriod) + sin(2*PI*10*iSample*samplePeriod);
  }
}

/*TODO: Consider implementing these on Device GPU for performance    */
void getFrequencies(float *array, float *amplitude, float *frequency,const int n){
  //Returns array of frequency components and corresponding amplitude from spectrum
  std::cout << "Extracting frequency components and amplitude...\n";
  float threshold = 0.5;
  for(int freqIdx = 0; freqIdx<n; freqIdx++){
    if( array[freqIdx] > threshold ){
      std::cout << std::fixed << std::setprecision(4) << "Amplitude: " << array[freqIdx] << " Frequency: " << freqIdx * (float)SAMPLERATE / (float)MAXLEN << " Hz" << std::endl;
    }
  }
  std::cout << std::endl;
}

void magFFT(Complex *a, float *result,  const int n){
  // computes the magnitude of the complex FFT signal and scales it by the length
  //TODO: consider using hypotf() from CUDA math library (device function)
  for(int i = 0; i<n; i++){
    result[i] = sqrt( pow(a[i].x / (float)n, 2.0) + pow(a[i].y / (float)n, 2.0) );
  }
}


/*******************************************************************************

Kernel Functions

*******************************************************************************/
__global__ void initStates(const unsigned int seed, curandState_t *states){
  //initialize the states for each thread
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init( seed, tid, 0, &states[tid]);

}

__global__ void rngNormal(float *dRand, curandState_t *states){
  //generate a batch of normally distributed random numbers
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  dRand[tid] = curand_normal(&states[tid]);
}


/*******************************************************************************

Test Functions

*******************************************************************************/
void executeCudaRNG(int totalThreads, int numBlocks, int blockSize){

  std::cout << "\nExecuting Random Number Generator Using cuRAND...";
  //host and device random number array
  float *hRand, *dRand;
  // Size, in bytes, of each vector
  const unsigned int bytes = totalThreads*sizeof(float);
  //random number generator (rgn) states
  curandState_t *states;

  // Start a timer
  timerEvent timer;
  startEventTimer(&timer);

  //allocate host and device memory
  hRand = (float *)malloc(bytes);
  cudaMalloc((void **)&dRand, bytes );
  cudaMalloc((void**) &states, totalThreads * sizeof(curandState_t));

  //initialize states
  initStates<<<numBlocks, totalThreads>>>(SEED, states);

  //Generate normally distributed data (float) with mean 0 and standard deviation of 1 ~N(0,1)
  rngNormal<<<numBlocks, blockSize>>>(dRand, states);

  //copy results from device to host
  cudaMemcpy(hRand, dRand, bytes, cudaMemcpyDeviceToHost);

  stopEventTimer(&timer);
  std::cout << "Elapsed Time: " << timer.time_ms << " ms\n" << std::endl;

  //print the array if you want
  //printArray(hRand, totalThreads);

  //clean up
  cudaFree(states);
  cudaFree(dRand);
  free(hRand);
}

void executeCudaFFT(void){

  std::cout << "Executing FFT Example using cuFFT...";

  //initialize host variables
  Complex *hSig = new Complex[MAXLEN]; //complex baseband signal
  Complex *hSig_w = new Complex[MAXLEN]; //spectrum of time domain signal
  float *hMagSig_w = new float[MAXLEN]; //magnitude of spectrum (host)

  //arrays containing a vector of the frequency components and corresponding amplitudes
  float *amplitude = new float[MAXLEN];
  float *frequency = new float[MAXLEN];

  //size of complex signal
  int bytes = MAXLEN * sizeof(Complex);

  //initialize host complex array withh a simple two tone signal
  generateSignal(hSig, MAXLEN);

  // Start a timer
  timerEvent timer;
  startEventTimer(&timer);

  //initialize and allocate the device signal
  cufftComplex *dSig;
  cudaMalloc((void **)&dSig, bytes);
  cudaMemcpy(dSig, hSig, bytes, cudaMemcpyHostToDevice);

  //Executing FFT on device
  cufftHandle plan;
  cufftPlan1d(&plan, MAXLEN, CUFFT_C2C, 1);
  cufftExecC2C(plan, (cufftComplex *)dSig, (cufftComplex *)dSig, CUFFT_FORWARD);
  cudaMemcpy(hSig_w, dSig, bytes, cudaMemcpyDeviceToHost);

  stopEventTimer(&timer);
  std::cout << "Elapsed Time: " << timer.time_ms << " ms\n" << std::endl;

  //Compute the magnitude of the fourier transformed signal
  magFFT(hSig_w, hMagSig_w, MAXLEN);

  //extract the amplitude and frequencies
  getFrequencies(hMagSig_w, amplitude, frequency, MAXLEN);

  //clean up
  delete hSig, hSig_w, hMagSig_w;
  cufftDestroy(plan);
  cudaFree(dSig);
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


  //Execute random number generator test on GPU using cuRand()
  executeCudaRNG(totalThreads, numBlocks, blockSize);

  //Execute FFT computation using cuFFT()
  executeCudaFFT();


  return 0;
}
