//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// HelloWorld.cpp
//
//    This is a simple example that demonstrates basic OpenCL setup and
//    use.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <thread>


#include "info.hpp"

//constant
#define NUM_BUFFER_ELEMENTS 10000
const std::vector<std::string> validKernelStr = {"add","sub", "mult"};

/*******************************************************************************

PROFILER FUNCTION USING OPENCL EVENTS

*******************************************************************************/
double executionTime(cl_event &event)
{
    cl_ulong start, end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);

    return (double)1.0e-6 * (end - start); // convert nanoseconds to microseconds on return
}


void CL_CALLBACK kernelFinished(cl_event event, cl_int status, void* data){
  std::string *str = static_cast<std::string*>(data);
  double t = executionTime(event);
  std::cout << *str << " Time: " << t << " us." << std::endl;
}



/*******************************************************************************

HELPER FUNCTIONS TO SETUP OPENCL

*******************************************************************************/
///
//  Create an OpenCL context on the first available platform using
//  either a GPU or CPU depending on what is available.
//
cl_context CreateContext(){
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0){
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS){
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS){
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }

    return context;
}

///
//  Create a command queue on the first device available on the
//  context
//
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device, int deviceNum, cl_command_queue_properties qProp){
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS){
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0){
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS){
        delete [] devices;
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0], qProp, NULL);
    if (commandQueue == NULL){
        delete [] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }

    *device = devices[0];
    delete [] devices;
    return commandQueue;
}

///
//  Create an OpenCL program from the kernel source file
//
cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName){
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open()){
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        NULL, NULL);
    if (program == NULL){
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS){
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

///
//  Cleanup any created OpenCL resources
//
void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, std::vector<cl_kernel> kernel){

    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    for(int i = 0; i<kernel.size(); i++){
      if (kernel[i] != 0)
          clReleaseKernel(kernel[i]);
    }

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);
}

/*******************************************************************************

MAIN

*******************************************************************************/
int main(int argc, char** argv)
{
  cl_context context = 0;
  cl_command_queue commandQueue = 0;
  cl_program program = 0;
    cl_device_id device = 0;
    cl_int errNum;
    bool profile = true;

    //start a timer at the begining of the program
    auto start = std::chrono::steady_clock::now();

    //parse input arguements
    std::vector<std::string> opKernelStr;
    cl_command_queue_properties qProp = CL_QUEUE_PROFILING_ENABLE; //this flag ensures that in-order queue is enabled by default
    for(int i = 1; i < argc; i++){
      std::string arg(argv[i]);
      if( arg == "-outorder"){
        qProp = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
        profile = false; //the profiling using events garantees an in order queue and acts as a synchronization point
      }else if(std::find(validKernelStr.begin(), validKernelStr.end(), argv[i]) != validKernelStr.end()){
        //found valid kernel
        opKernelStr.push_back(argv[i]);
      }else{
        std::cout << argv[i] << " kernel is not implemented!.\n";
      }
    }

    //initialize number of events based on user input
    size_t numKernels = opKernelStr.size();
    std::vector<cl_kernel> kernels(numKernels);;
    cl_event *events = new cl_event[numKernels];
    std::vector<cl_mem> bufferA(numKernels), bufferB(numKernels), bufferResult(numKernels);

    // Create an OpenCL context on first available platform
    context = CreateContext();
    if (context == NULL){
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }

    // Create a command-queue on the first device available
    // on the created context
    commandQueue = CreateCommandQueue(context, &device, 0, qProp);
    if (commandQueue == NULL){
        Cleanup(context, commandQueue, program, kernels);
        return 1;
    }

    // Create OpenCL program from HelloWorld.cl kernel source
    program = CreateProgram(context, device, "assignment13.cl");
    if (program == NULL){
        Cleanup(context, commandQueue, program, kernels);
        return 1;
    }

    // Create memory objects that will be used as arguments to
    // kernel.  First create host memory arrays that will be
    // used to store the arguments to the kernel
    // initialize input arrays
    int *a = new int[NUM_BUFFER_ELEMENTS];
    int *b = new int[NUM_BUFFER_ELEMENTS];
    int *result = new int[NUM_BUFFER_ELEMENTS ];
    for(int i = 0; i < NUM_BUFFER_ELEMENTS ; i++){
        a[i] = i;
        b[i] = 2*i;
    }

    //print the contents of the two data vectors (but only 10 elements)
    std::cout << "Array 1: ";
    for (int i = 0; i < 10; i++){
        std::cout << a[i] << ' ';
    }
    std::cout << "\nArray 2: ";
    for (int i = 0; i < 10; i++){
        std::cout << b[i] << ' ';
    }
    std::cout << "\n\n";
    if(profile) std::cout << "Kernel Performance Using Events: " << std::endl;

    //iterate through the arithmetic kernels
    for(size_t i = 0; i < numKernels; i++){

      // Create OpenCL kernel
      kernels[i] = clCreateKernel(program, opKernelStr[i].c_str(), NULL);
      if (kernels[i] == NULL){
          std::cerr << "Failed to create kernel" << std::endl;
          Cleanup(context, commandQueue, program, kernels);
          return 1;
      }

      //create three buffers , A, B and Result for each kernel
      bufferA[i] = clCreateBuffer(
                                  context,
                                  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  sizeof(int) * NUM_BUFFER_ELEMENTS ,
                                  a,
                                  NULL);


      bufferB[i] = clCreateBuffer(
                                  context,
                                  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  sizeof(int) * NUM_BUFFER_ELEMENTS,
                                  b,
                                  NULL);

      bufferResult[i] = clCreateBuffer(
                                  context,
                                  CL_MEM_READ_WRITE,
                                  sizeof(int) * NUM_BUFFER_ELEMENTS,
                                  NULL,
                                  NULL);



      // Set the kernel arguments (result, a, b)
      errNum = clSetKernelArg(kernels[i], 0, sizeof(cl_mem), &bufferA[i]);
      errNum |= clSetKernelArg(kernels[i], 1, sizeof(cl_mem), &bufferB[i]);
      errNum |= clSetKernelArg(kernels[i], 2, sizeof(cl_mem), &bufferResult[i]);
      if (errNum != CL_SUCCESS){
          std::cerr << "Error setting kernel arguments."   << std::endl;
          Cleanup(context, commandQueue, program, kernels);
          return 1;
      }

      size_t gWI = NUM_BUFFER_ELEMENTS;
      // Queue the kernel up for execution across the array
      errNum = clEnqueueNDRangeKernel(commandQueue, kernels[i], 1, NULL,
                                      (const size_t*)&gWI, (const size_t*)NULL,
                                      0, NULL, &events[i]);
      if (errNum != CL_SUCCESS){
          std::cerr << "Error queuing kernel for execution." << std::endl;
          Cleanup(context, commandQueue, program, kernels);
          return 1;
      }

      // use the concept of clEvents from each kernel to perform profiling
      if( profile ){
        //let the user know that the kernel is finished by setting a callback
        errNum = clSetEventCallback(events[i], CL_COMPLETE, &kernelFinished, static_cast<void*>(&opKernelStr[i]));
        if(errNum < 0)
            std::cout << "Could not set callback for " << opKernelStr[i].c_str() << std::endl;
      }

    }

    //wait untill all events are finished
    //wait for all other events
    clWaitForEvents(numKernels, events);

    //read all buffers
    for(int j = 0; j < numKernels; j++){
      // Read the output buffer back to the Host
      errNum = clEnqueueReadBuffer(commandQueue, bufferResult[j], CL_TRUE,
                                     0, NUM_BUFFER_ELEMENTS * sizeof(int), result,
                                     numKernels, events , NULL);

      if (errNum != CL_SUCCESS){
          std::cerr << "Error reading result buffer." << std::endl;
          Cleanup(context, commandQueue, program, kernels);
          return 1;
      }

      // Output the result buffer (only 10 elements)
      std::cout << "\n" << opKernelStr[j] << " Result: ";
      for (int i = 0; i < 10; i++){
          std::cout << result[i] << " ";
      }
    }

    auto end = std::chrono::steady_clock::now();
    std::cout << "\n\nProgram Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n" << std::endl;

    Cleanup(context, commandQueue, program, kernels);

    return 0;
}
