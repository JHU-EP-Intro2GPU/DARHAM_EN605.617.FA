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
#define THRESHOLD 2.2

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
void getRngSpeed(float **rngDoppMag, float *rngGrid, float *speedGrid){
  // extracts the range from the range-doppler matrix
  for(int i = 0; i<RANGEFFTLENGTH; i++){
    for(int j = 0; j<DOPPLERFFTLENGTH; j++){
      if(rngDoppMag[i][j] >= THRESHOLD)
        std::cout << "Target Detected.\n" << "Range: " << rngGrid[i] << " m; Speed: " << speedGrid[j] << " m/s." << std::endl;
    }
  }
}

void abs(Complex **rngDoppMatrix, float **rngDoppMag){
  // computes the magnitude of the complex range doppler matrix
  for(int i = 0; i<RANGEFFTLENGTH; i++){
    for(int j = 0; j<DOPPLERFFTLENGTH; j++){
      rngDoppMag[i][j] = sqrt( pow(rngDoppMatrix[i][j].x  , 2.0) + pow(rngDoppMatrix[i][j].y , 2.0) );
    }
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

void readData( std::vector <std::vector <float> > &iData, std::vector <std::vector <float> > &qData,
               std::string realIQFileName, std::string imagIQFileName){

  // Read In phase (I) and Quadrature (Q) Data
  // Read I Data
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

}

// Perform the fast time processing. this simply takes the FFT of each pulse
void executeRangeProcessing(std::vector <std::vector <float> > &iData, std::vector <std::vector <float> > &qData,
                            Complex **rngFFTMatrix){
  //Range Processing
  //Process FFT of range samples (fast time)
  size_t nPulses = iData[0].size();
  size_t rngSamples = iData.size();

  //host signal and spectrum
  Complex *hSig = new Complex[rngSamples]; //complex baseband signal
  Complex *hSig_w = new Complex[RANGEFFTLENGTH]; //spectrum of time domain signal

  //device signal and spectrum
  cufftComplex *dSig,*dSig_w;
  // different byte size for baseband signal and spectrum (rngSample vs RANGEFFTLENGTH)
  int bytes = rngSamples * sizeof(Complex);
  cudaMalloc((void **)&dSig, bytes);

  int fftBytes = RANGEFFTLENGTH * sizeof(Complex);
  cudaMalloc((void **)&dSig_w, fftBytes);

  // CUDA's FFT Handle
  cufftHandle plan;
  cufftPlan1d(&plan, RANGEFFTLENGTH, CUFFT_C2C, 1);

  // for each pulse (or sweep) comput the FFT across range samples
  for(int i = 0; i<nPulses; i++){

    //this next loop will operate across fast time or range samples
    for(int j = 0; j<rngSamples; j++){
      hSig[j].x = iData[j][i];
      hSig[j].y = qData[j][i];
    }

    cudaMemcpy(dSig, hSig, bytes, cudaMemcpyHostToDevice);
    cufftExecC2C(plan, (cufftComplex *)dSig, (cufftComplex *)dSig_w, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    cudaMemcpy(hSig_w, dSig_w, fftBytes, cudaMemcpyDeviceToHost);

    //shift the fft output so that it is within -Fs/2 < freq < Fs/2
    fftshift(hSig_w, RANGEFFTLENGTH/2, RANGEFFTLENGTH);

    //build the post range processed data cube
    for(int k = 0; k<RANGEFFTLENGTH; k++){
          rngFFTMatrix[k][i] = hSig_w[k];
    }
  }

  delete hSig, hSig_w;
  cudaFree(dSig);
  cudaFree(dSig_w);
}

// perform doppler processing. This takes as input the post range processing matrix
// takes the FFT across pulses
void executeDopplerProcessing(int nPulses, Complex **rngFFTMatrix, Complex **rngDoppMatrix){

  // DOPPLER PROCESSING
  // this next step implements the same procedure as above but across
  // the slow time i.e. across columns

  //host signal across pulses
  Complex *hSigPulse = new Complex[nPulses]; // signal across pulses
  Complex *hSigPulse_w = new Complex[DOPPLERFFTLENGTH]; //spectrum

  cufftComplex *dSigPulse, *dSigPulse_w;
  int bytes = nPulses * sizeof(Complex);
  int fftbytes = DOPPLERFFTLENGTH * sizeof(Complex);

  cudaMalloc((void **)&dSigPulse, bytes);
  cudaMalloc((void **)&dSigPulse_w, fftbytes);

  // initiate fft handles to perform fast and slow time processing
  cufftHandle plan;
  cufftPlan1d(&plan, DOPPLERFFTLENGTH, CUFFT_C2C, 1);

  for(int i = 0; i<RANGEFFTLENGTH; i++){
    for(int j = 0; j<nPulses; j++){
      hSigPulse[j] = rngFFTMatrix[i][j];
    }
    //perform the slow time / doppler
    cudaMemcpy(dSigPulse, hSigPulse, bytes, cudaMemcpyHostToDevice);
    cufftExecC2C(plan, (cufftComplex *)dSigPulse, (cufftComplex *)dSigPulse_w, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    cudaMemcpy(hSigPulse_w, dSigPulse_w, fftbytes, cudaMemcpyDeviceToHost);

    fftshift(hSigPulse_w, DOPPLERFFTLENGTH/2, DOPPLERFFTLENGTH);

    for(int k = 0; k<DOPPLERFFTLENGTH; k++){
          rngDoppMatrix[i][k] = hSigPulse_w[k];
    }
  }

  delete hSigPulse, hSigPulse_w;
  cudaFree(dSigPulse);
  cudaFree(dSigPulse_w);
}


/*******************************************************************************

Range Doppler Response and Radar Target Function

*******************************************************************************/
void executeTargetDetection(std::string realIQFileName, std::string imagIQFileName){

  // Main function to calculate the range doppler map of raw radar IQ data and
  // extract range and speed of a target if present

  //Read in In Phase and Quadrature data
  std::vector <std::vector <float> > iData, qData;
  readData( iData , qData, realIQFileName, imagIQFileName);

  int nPulses  = iData[0].size(); // number of pulses (columns)
  int rngSamples = iData.size(); // number of range samples (rows)

  // Initialze the radar data matrices
  Complex **rngFFTMatrix = new Complex *[RANGEFFTLENGTH]; // post range processed matrix using FFT
  Complex **rngDoppMatrix = new Complex *[RANGEFFTLENGTH]; // post doppler processed matrix using FFT
  float **rngDoppMag = new float *[RANGEFFTLENGTH]; // magnitude response of the range-doppler matrix

  for(int i = 0; i < RANGEFFTLENGTH; i++){
    rngFFTMatrix[i] = new Complex[nPulses];
    rngDoppMatrix[i] = new Complex[DOPPLERFFTLENGTH];
    rngDoppMag[i] = new float[DOPPLERFFTLENGTH];
  }


   // Start a timer
  timerEvent timer;
  startEventTimer(&timer);

  //perform the range processing across range samples
  executeRangeProcessing( iData, qData , rngFFTMatrix);

  stopEventTimer(&timer);
  std::cout << "Range Processing Time Elapsed: " << timer.time_ms << " ms\n" << std::endl;

  // perform the doppler processing across pulses
  executeDopplerProcessing( nPulses, rngFFTMatrix, rngDoppMatrix);

  stopEventTimer(&timer);
  std::cout << "Doppler Processing Time Elapsed: " << timer.time_ms << " ms\n" << std::endl;

  // calculate the magnitude of the range doppler matrix
  abs( rngDoppMatrix, rngDoppMag );

  //calculate the range and speed grid of the rand-doppler map data
  // the range and speed grid are simply frequency components (FFT)
  // using the sample rate aand wavelength we can calculate the grids
  float sweepSlope = SAMPLERATE / SWEEPTIME;
  float waveLength = LIGHTSPEED / CENTERFREQ;
  float prf = SAMPLERATE / rngSamples; //pulse to pulse repitition frequency
  float *rngGrid = calcRngGrid(RANGEFFTLENGTH, SAMPLERATE,sweepSlope);
  float *speedGrid = calcSpeedGrid(DOPPLERFFTLENGTH, prf, waveLength);

  //Extract the range and speed of the target from the range doppler map
  getRngSpeed( rngDoppMag, rngGrid, speedGrid);

  delete rngFFTMatrix, rngDoppMatrix;
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
  executeTargetDetection(realIQFileName, imagIQFileName);

  return 0;
}
