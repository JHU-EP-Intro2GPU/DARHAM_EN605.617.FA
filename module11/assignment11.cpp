/*

this program is the edited version of convolution.cpp

It has been modified to generate a random 49x49 signal based on user input and uses
a 7x7 filter to perform the convolution

*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <chrono>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

// Constants
const unsigned int inputSignalWidth  = 49;
const unsigned int inputSignalHeight = 49;

cl_float inputSignal[inputSignalHeight][inputSignalWidth];
const unsigned int outputSignalWidth  = 43;
const unsigned int outputSignalHeight = 43;

cl_float outputSignal[outputSignalHeight][outputSignalWidth];

const unsigned int maskWidth  = 7;
const unsigned int maskHeight = 7;

cl_float mask[maskHeight][maskWidth] =
{
  {0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25},
  {0.25, 0.50, 0.50, 0.50, 0.50, 0.50, 0.25},
  {0.25, 0.50, 0.75, 0.75, 0.75, 0.50, 0.25},
  {0.25, 0.50, 0.75, 1.00, 0.75, 0.50, 0.25},
  {0.25, 0.50, 0.75, 0.75, 0.75, 0.50, 0.25},
  {0.25, 0.50, 0.50, 0.50, 0.50, 0.50, 0.25},
  {0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25},
};

///
// Function to check and handle OpenCL errors
inline void
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void CL_CALLBACK contextCallback(
	const char * errInfo,
	const void * private_info,
	size_t cb,
	void * user_data)
{
	std::cout << "Error occured during context use: " << errInfo << std::endl;
	// should really perform any clearup and so on at this point
	// but for simplicitly just exit.
	exit(1);
}

//
//	main() for Convoloution example
//
int main(int argc, char** argv)
{
  cl_int errNum;
  cl_uint numPlatforms;
	cl_uint numDevices;
  cl_platform_id * platformIDs;
	cl_device_id * deviceIDs;
  cl_context context = NULL;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem inputSignalBuffer;
	cl_mem outputSignalBuffer;
	cl_mem maskBuffer;

  // use default seed, otw use user supplied seed
  int seed = 1234;
  if( argc > 1 )
    seed = atoi(argv[1]);

  //generate random signal based on seed
  srand(seed);
  for(int y = 0; y < inputSignalHeight; y++){
    for(int x = 0; x < inputSignalWidth; x++){
      inputSignal[y][x] = rand()%4;
    }
  }

  // First, select an OpenCL platform to run on.
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr( (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), "clGetPlatformIDs");

	platformIDs = (cl_platform_id *)alloca(
       		sizeof(cl_platform_id) * numPlatforms);

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr(
	   (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
	   "clGetPlatformIDs");

	// Iterate through the list of platforms until we find one that supports
	// a CPU device, otherwise fail with an error.
	deviceIDs = NULL;
	cl_uint i;
	for (i = 0; i < numPlatforms; i++){
    errNum = clGetDeviceIDs(
                            platformIDs[i],
                            CL_DEVICE_TYPE_GPU,
                            0,
                            NULL,
                            &numDevices);

    if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND){
			checkErr(errNum, "clGetDeviceIDs");
    }
	  else if (numDevices > 0){
      deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
			errNum = clGetDeviceIDs(
                      				platformIDs[i],
                      				CL_DEVICE_TYPE_GPU,
                      				numDevices,
                      				&deviceIDs[0],
                      				NULL);
			checkErr(errNum, "clGetDeviceIDs");
			break;
	   }
	}

	// Check to see if we found at least one CPU device, otherwise return
  // 	if (deviceIDs == NULL) {
  // 		std::cout << "No CPU device found" << std::endl;
  // 		exit(-1);
  // 	}

  // Next, create an OpenCL context on the selected platform.
  cl_context_properties contextProperties[] =
  {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[i],
        0
  };
  context = clCreateContext(
                        		contextProperties,
                        		numDevices,
                            deviceIDs,
                        		&contextCallback,
                        		NULL,
                        		&errNum);
	checkErr(errNum, "clCreateContext");

	std::ifstream srcFile("assignment11.cl");
  checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading Convolution.cl");

	std::string srcProg(
                      std::istreambuf_iterator<char>(srcFile),
                      (std::istreambuf_iterator<char>()));

	const char * src = srcProg.c_str();
	size_t length = srcProg.length();

	// Create program from source
	program = clCreateProgramWithSource(
                                  		context,
                                  		1,
                                  		&src,
                                  		&length,
                                  		&errNum);
	checkErr(errNum, "clCreateProgramWithSource");

	// Build program
	errNum = clBuildProgram(
                      		program,
                      		numDevices,
                      		deviceIDs,
                      		NULL,
                      		NULL,
                      		NULL);

  if (errNum != CL_SUCCESS){
    // Determine the reason for the error
    char buildLog[16384];
    clGetProgramBuildInfo(
                      		program,
                      		deviceIDs[0],
                      		CL_PROGRAM_BUILD_LOG,
                          sizeof(buildLog),
                      		buildLog,
                      		NULL);

    std::cerr << "Error in kernel: " << std::endl;
    std::cerr << buildLog;
  	checkErr(errNum, "clBuildProgram");
  }

	// Create kernel object
	kernel = clCreateKernel(
                      		program,
                      		"convolve",
                      		&errNum);
	checkErr(errNum, "clCreateKernel");

	// Now allocate buffers
	inputSignalBuffer = clCreateBuffer(
                                  		context,
                                  		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  		sizeof(cl_uint) * inputSignalHeight * inputSignalWidth,
                                  		static_cast<void *>(inputSignal),
                                  		&errNum);
	checkErr(errNum, "clCreateBuffer(inputSignal)");

	maskBuffer = clCreateBuffer(
                          		context,
                          		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          		sizeof(cl_uint) * maskHeight * maskWidth,
                          		static_cast<void *>(mask),
                          		&errNum);
	checkErr(errNum, "clCreateBuffer(mask)");

	outputSignalBuffer = clCreateBuffer(
                                  		context,
                                  		CL_MEM_WRITE_ONLY,
                                  		sizeof(cl_uint) * outputSignalHeight * outputSignalWidth,
                                  		NULL,
                                  		&errNum);
	checkErr(errNum, "clCreateBuffer(outputSignal)");

	// Pick the first device and create command queue.
	queue = clCreateCommandQueue(
                            		context,
                            		deviceIDs[0],
                            		0,
                            		&errNum);
	checkErr(errNum, "clCreateCommandQueue");

  errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &maskBuffer);
  errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &inputSignalWidth);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &maskWidth);
	checkErr(errNum, "clSetKernelArg");

	const size_t globalWorkSize[2] = { outputSignalWidth, outputSignalHeight };
  const size_t localWorkSize[2]  = { 1, 1 };


  auto start = std::chrono::steady_clock::now();
    // Queue the kernel up for execution across the array
  errNum = clEnqueueNDRangeKernel(
                              		queue,
                              		kernel,
                              		2,
                              		NULL,
                                      globalWorkSize,
                              		localWorkSize,
                                      0,
                              		NULL,
                              		NULL);
	checkErr(errNum, "clEnqueueNDRangeKernel");

	errNum = clEnqueueReadBuffer(
                            		queue,
                            		outputSignalBuffer,
                            		CL_TRUE,
                                    0,
                            		sizeof(cl_uint) * outputSignalHeight * outputSignalHeight,
                            		outputSignal,
                                    0,
                            		NULL,
                            		NULL);
	checkErr(errNum, "clEnqueueReadBuffer");
  auto end = std::chrono::steady_clock::now();

  // Output the result buffer
  for (int y = 0; y < outputSignalHeight; y++){
	   for (int x = 0; x < outputSignalWidth; x++){
       //put a 0 next to the number if single digit to make look better in command line
       if( outputSignal[y][x] < 10 && outputSignal[y][x] >= 0 )
        std::cout << "0";
        //printf to make the output look better in command line
        printf("%.2f ", outputSignal[y][x]);
        //std::cout << std::setprecision(4) << outputSignal[y][x] << " ";
		}
		std::cout << std::endl;
	}

  std::cout << std::endl << "Convolution Kernel Time Elapsed: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-3 << " ms\n"<< std::endl;

  return 0;
}
