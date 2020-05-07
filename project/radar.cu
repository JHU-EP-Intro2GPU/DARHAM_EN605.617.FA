/*******************************************************************************

Project

This project showcases how to use CUDA signal processing library to perform
stpectrum analysis.

This code read in In phase and Quadrature data from the file, performs spectrum
analysis on the data and extracts the range and speed of a target.

MATLAB is used to simulate a radar signal and the effects of the environment
and a moving target. The radar signal is in the form of IQ data which
is a complex 2D array (I + jQ). The output is the relative range and radial
velocity of the target to the target.

Author: Said Darham
*******************************************************************************/
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <array>

#include <curand.h>
#include <curand_kernel.h>//cuRand header files
#include <cufft.h>//cuFFT
#include <iomanip> //for setting float precision

//Timer struct declaration. Using CUDA EVENTS
typedef struct timer{
  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;
  float time_ms;
} timerEvent;

typedef float2 Complex;

#define SAMPLERATE 150e6
#define PI 3.14159265358979323846
#define LIGHTSPEED 300000000

// having to use some predefined parameters, if calling from the model these
// would be passed as input the program
#define RANGEFFTLENGTH 2048 //fast time (range samples) N-point FFT length
#define DOPPLERFFTLENGTH 256 //slow time (pulses) N-point FFT
#define SWEEPTIME  7.3333e-06
#define CENTERFREQ 12e9

//Static threshold. normally in sophisticated radar signal processing this is value
//dynamic such as Constant False Alarm Rate algorithms.
#define THRESHOLD 2

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
void getRngSpeed(float magResult[][DOPPLERFFTLENGTH], float *rngGrid, float *speedGrid){
  // extracts the range from the range-doppler matrix
  for(int i = 0; i<RANGEFFTLENGTH; i++){
    for(int j = 0; j<DOPPLERFFTLENGTH; j++){
      if(magResult[i][j] >= THRESHOLD)
        std::cout << "Target Detected.\n" << "Range: " << rngGrid[i] << " m; Speed: " << speedGrid[j] << " m/s." << std::endl;
    }
  }
}

void magFFT(Complex *a, float *result,  const int n){
  // computes the magnitude of the complex FFT signal and scales it by the length
  //TODO: consider using hypotf() from CUDA math library (device function)
  for(int i = 0; i<n; i++){
    result[i] = sqrt( pow(a[i].x  , 2.0) + pow(a[i].y , 2.0) );
  }
}

float* calcRngGrid( int nPoint, float Fs, float sweepSlope){
  // calculates the range grid of the range doppler map. This is the rows or
  // first dimension of the post FFT data cube
  float *rngGrid = new float[nPoint];
  float freq_res = Fs/nPoint;
  for (int i = 0; i<nPoint; i++){
    rngGrid[i] = LIGHTSPEED*(i*freq_res - Fs/2) / sweepSlope / 2;
  }
  return rngGrid;
}

float* calcSpeedGrid( int nPoint, float Fs, float waveLength){
  // calculates the speed grid of the range doppler map. This is the columns or
  // second dimension of the post FFT data cube
  //make sure nPoint is event, may have to negate it
  float *speedGrid = new float[nPoint];
  float freq_res = Fs/nPoint;
  for (int i = 0; i<nPoint; i++){
    speedGrid[i] = (i*freq_res - Fs/2)*waveLength / 2;
  }
  return speedGrid;
}

void fftshift( Complex *fftDat, int N, int n){
    // shift zero-frequency components to center of spectrum
    int j;
    Complex temp;
    for(int i = 0; i <N;i++){

      temp = fftDat[n-1];
      for (j = n-1; j > 0; j--)
        fftDat[j] = fftDat[j - 1];

      fftDat[j] = temp;
    }
}

/*******************************************************************************

Range Doppler Response and Radar Function

*******************************************************************************/
//TODO: seperate functionality and clean up
void executeRangeDopplerResponse(std::string realIQFileName, std::string imagIQFileName){

  // Main function to calculate the range doppler map of raw radar IQ data and
  // extract range and speed of a target if present

  // Read In phase (I) and Quadrature (Q) Data
  // Read I Data
  std::vector <std::vector <float> > iData;
  std::ifstream infileReal( realIQFileName );
  while (infileReal){
    std::string s;
    if (!getline( infileReal, s )) break;
    std::istringstream ss( s );
    std::vector <float> iRecord;
    while (ss){
     std::string s;
     float f;
     if (!getline( ss, s, ';' )) break;
     f = std::stof(s);
     iRecord.push_back( f );
    }
    iData.push_back( iRecord );
   }
   if (!infileReal.eof())
     std::cerr << "Fooey!\n";

   // Read Q Data
   std::vector <std::vector <float> > qData;
   std::ifstream infileImag( imagIQFileName );
   while (infileImag){
     std::string s;
     if (!getline( infileImag, s )) break;
     std::istringstream ss( s );
     std::vector <float> qRecord;
     while (ss){
       std::string s;
       float f;
       if (!getline( ss, s, ';' )) break;
       f = stof(s);
       qRecord.push_back( f );
     }
     qData.push_back( qRecord );
    }
    if (!infileImag.eof())
    std::cerr << "Fooey!\n";

    // allocate host data
    Complex *hSig = new Complex[iData.size()]; //complex baseband signal
    Complex *hSig_w = new Complex[RANGEFFTLENGTH]; //spectrum of time domain signal
    float *hMagSig_w = new float[RANGEFFTLENGTH]; //magnitude of spectrum (host)
    Complex radarData[RANGEFFTLENGTH][iData[0].size()];
    Complex fftradarData[RANGEFFTLENGTH][DOPPLERFFTLENGTH];
    float magResult[RANGEFFTLENGTH][DOPPLERFFTLENGTH];

    //calculate the range and speed grid of the rand-doppler map data
    float sweepSlope = SAMPLERATE / SWEEPTIME;
    float waveLength = LIGHTSPEED / CENTERFREQ;
    float prf = SAMPLERATE / iData.size(); //pulse to pulse repitition frequency
    float *rngGrid = calcRngGrid(RANGEFFTLENGTH, SAMPLERATE,sweepSlope);
    float *speedGrid = calcSpeedGrid(DOPPLERFFTLENGTH, prf, waveLength);

    //initialize and allocate the device signal
    int bytes = iData.size() * sizeof(Complex);
    int fftBytes = RANGEFFTLENGTH * sizeof(Complex);
    cufftComplex *dSig, *fftSig;
    cudaMalloc((void **)&dSig, bytes);
    cudaMalloc((void **)&fftSig, fftBytes);

    // initiate fft handles to perform fast and slow time processing
    cufftHandle plan, doppplan;
    cufftPlan1d(&plan, RANGEFFTLENGTH, CUFFT_C2C, 1);
    cufftPlan1d(&doppplan, DOPPLERFFTLENGTH, CUFFT_C2C, 1);

    // Start a timer
    timerEvent timer;
    startEventTimer(&timer);

    //Range Processing
    //Process FFT of range samples (fast time)
    for(int i = 0; i<iData[0].size(); i++){
      //this next loop will operate on the fast time or range samples
      for(int j = 0; j<iData.size(); j++){
        hSig[j].x = iData[j][i];
        hSig[j].y = qData[j][i];
      }
      cudaMemcpy(dSig, hSig, bytes, cudaMemcpyHostToDevice);
      cufftExecC2C(plan, (cufftComplex *)dSig, (cufftComplex *)fftSig, CUFFT_FORWARD);
      cudaDeviceSynchronize();
      cudaMemcpy(hSig_w, fftSig, fftBytes, cudaMemcpyDeviceToHost);

      //shift the fft output so that it is within -Fs/2 < freq < Fs/2
      fftshift(hSig_w, RANGEFFTLENGTH/2, RANGEFFTLENGTH);

      //build the post range processed data cube
      for(int k = 0; k<RANGEFFTLENGTH; k++){
            radarData[k][i] = hSig_w[k];
      }
    }
    stopEventTimer(&timer);
    std::cout << "Range Processing Time Elapsed: " << timer.time_ms << " ms\n" << std::endl;

    // DOPPLER PROCESSING
    // this next step implements the same procedure as above but across
    // the slow time i.e. across columns
    Complex *hdoppFFT = new Complex[iData[0].size()];
    Complex *hdoppFFT_w = new Complex[DOPPLERFFTLENGTH];
    cufftComplex *ddoppFFT, *ddoppFFTout;
    int doppfftBytes = DOPPLERFFTLENGTH * sizeof(Complex);
    int doppfftBytes2 = iData[0].size() * sizeof(Complex);

    cudaMalloc((void **)&ddoppFFT, doppfftBytes2);
    cudaMalloc((void **)&ddoppFFTout, doppfftBytes);

    startEventTimer(&timer);
    for(int i = 0; i<RANGEFFTLENGTH; i++){
      for(int j = 0; j<iData[0].size(); j++){
        hdoppFFT[j] = radarData[i][j];
      }
      //perform the slow time / dopper
      cudaMemcpy(ddoppFFT, hdoppFFT, doppfftBytes2, cudaMemcpyHostToDevice);
      cufftExecC2C(doppplan, (cufftComplex *)ddoppFFT, (cufftComplex *)ddoppFFTout, CUFFT_FORWARD);
      cudaDeviceSynchronize();
      cudaMemcpy(hdoppFFT_w, ddoppFFTout, doppfftBytes, cudaMemcpyDeviceToHost);

      fftshift(hdoppFFT_w, DOPPLERFFTLENGTH/2, DOPPLERFFTLENGTH);

      for(int k = 0; k<DOPPLERFFTLENGTH; k++){
            fftradarData[i][k] = hdoppFFT_w[k];
      }
    }
    stopEventTimer(&timer);
    std::cout << "Doppler Processing Time Elapsed: " << timer.time_ms << " ms\n" << std::endl;

    // calculate the magnitude of the data cube
    for(int i = 0; i<RANGEFFTLENGTH; i++){
      for(int j = 0; j<DOPPLERFFTLENGTH; j++){
        magResult[i][j] = sqrt( pow(fftradarData[i][j].x  , 2.0) + pow(fftradarData[i][j].y , 2.0) );
      }
    }

    //Extract the range and speed of the target from the range doppler map
    getRngSpeed( magResult, rngGrid, speedGrid);

  //clean up
  delete hSig, hSig_w, hMagSig_w;
  cufftDestroy(plan);
  cufftDestroy(doppplan);

  cudaFree(dSig);
  cudaFree(fftSig);

}




/*******************************************************************************

MAIN

*******************************************************************************/
int main(int argc, char** argv)
{

  std::string realIQFileName, imagIQFileName;
  if( argc > 1){
    std::string realIQFileName(argv[1]);
    std::string imagIQFileName(argv[2]);
  }else{
    realIQFileName = "x_real.dat";
    imagIQFileName = "x_imag.dat";
  }


  //Range doppler response
  executeRangeDopplerResponse(realIQFileName, imagIQFileName);

  return 0;
}
