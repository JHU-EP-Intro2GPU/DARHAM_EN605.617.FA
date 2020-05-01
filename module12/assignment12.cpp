//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// raytracer.cpp
//
//    This is a (very) simple raytracer that is intended to demonstrate
//    using OpenCL buffers.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include "info.hpp"

#define DEFAULT_PLATFORM 0


#define NUM_BUFFER_ELEMENTS 16
#define NUM_SUB_BUFFER_ELEMENTS 2

// Function to check and handle OpenCL errors
inline void
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void initializeInputArray(int argc, char **argv, float *inputArray){

  if( argc != NUM_BUFFER_ELEMENTS + 1){
    std::cout << "Using Default Values...." << std::endl;
    for(int i = 0; i<NUM_BUFFER_ELEMENTS; i++){

      inputArray[i] = i;

    }
  }else{
    for(int i = 1; i<argc; i++){
      inputArray[i-1] = std::atof(argv[i]) ;
    }
  }

}

///
//	main() for simple buffer and sub-buffer example
//
int main(int argc, char** argv)
{

    auto start = std::chrono::steady_clock::now();
    cl_int errNum;
    cl_uint numPlatforms;
    cl_uint numDevices;
    cl_platform_id * platformIDs;
    cl_device_id * deviceIDs;
    cl_context context;
    cl_program program;
    std::vector<cl_kernel> kernels;
    std::vector<cl_command_queue> queues;
    std::vector<cl_mem> subBuffers;
    float *inputArray = new float[NUM_BUFFER_ELEMENTS];
    float *outputArray = new float[NUM_BUFFER_ELEMENTS];

    int platform = DEFAULT_PLATFORM;

    std::cout << "\n\n\tSimple buffer and sub-buffer Example" << std::endl;







    // First, select an OpenCL platform to run on.
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr(
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
        "clGetPlatformIDs");

    platformIDs = (cl_platform_id *)alloca(
            sizeof(cl_platform_id) * numPlatforms);

    std::cout << "Number of platforms: \t" << numPlatforms << std::endl;

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr(
       (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS),
       "clGetPlatformIDs");

    std::ifstream srcFile("assignment12.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading assignment12.cl");

    std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

    const char * src = srcProg.c_str();
    size_t length = srcProg.length();

    deviceIDs = NULL;
    DisplayPlatformInfo(
        platformIDs[platform],
        CL_PLATFORM_VENDOR,
        "CL_PLATFORM_VENDOR");

    errNum = clGetDeviceIDs(
        platformIDs[platform],
        CL_DEVICE_TYPE_ALL,
        0,
        NULL,
        &numDevices);
    if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
    {
        checkErr(errNum, "clGetDeviceIDs");
    }

    deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
    errNum = clGetDeviceIDs(
        platformIDs[platform],
        CL_DEVICE_TYPE_ALL,
        numDevices,
        &deviceIDs[0],
        NULL);
    checkErr(errNum, "clGetDeviceIDs");

    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[platform],
        0
    };

    context = clCreateContext(
        contextProperties,
        numDevices,
        deviceIDs,
        NULL,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateContext");

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
        "-I.",
        NULL,
        NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
            program,
            deviceIDs[0],
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog),
            buildLog,
            NULL);

            std::cerr << "Error in OpenCL C source: " << std::endl;
            std::cerr << buildLog;
            checkErr(errNum, "clBuildProgram");
    }

    // create buffers and sub-buffers
    initializeInputArray(argc, argv, inputArray);

    // create a single buffer to cover all the input data
    cl_mem buffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sizeof(float) * NUM_BUFFER_ELEMENTS,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");

    //  create sub buffers from main buffer
    int numSubBuffers = NUM_BUFFER_ELEMENTS / NUM_SUB_BUFFER_ELEMENTS;
    for (unsigned int i = 0; i < numSubBuffers; i++)
    {
        cl_buffer_region region =
            {
                NUM_SUB_BUFFER_ELEMENTS * i * sizeof(float),
                NUM_SUB_BUFFER_ELEMENTS * sizeof(float)
            };
        cl_mem subbuffer = clCreateSubBuffer(
            buffer,
            CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region,
            &errNum);
        checkErr(errNum, "clCreateSubBuffer");

        subBuffers.push_back(subbuffer);
    }

    int size = NUM_SUB_BUFFER_ELEMENTS;

    // Create command queues
    for (unsigned int i = 0; i < numSubBuffers; i++)
    {
        cl_command_queue queue =
            clCreateCommandQueue(
                context,
                deviceIDs[0],
                0,
                &errNum);
        checkErr(errNum, "clCreateCommandQueue");

        queues.push_back(queue);

        cl_kernel kernel = clCreateKernel(
            program,
            "mean",
            &errNum);
        checkErr(errNum, "clCreateKernel(mean)");

        errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&subBuffers[i]);
        errNum = clSetKernelArg(kernel, 1, sizeof(cl_uint), &size);
        checkErr(errNum, "clSetKernelArg(mean)");

        kernels.push_back(kernel);
    }

    // Write input array data into buffer
    errNum = clEnqueueWriteBuffer(
        queues[0],
        buffer,
        CL_TRUE,
        0,
        sizeof(float) * NUM_BUFFER_ELEMENTS,
        (void*)inputArray,
        0,
        NULL,
        NULL);

    std::vector<cl_event> events;


    // call kernel for each queue
    for (unsigned int i = 0; i < queues.size(); i++)
    {
        cl_event event;

        size_t gWI = NUM_SUB_BUFFER_ELEMENTS;

        errNum = clEnqueueNDRangeKernel(
            queues[i],
            kernels[i],
            1,
            NULL,
            (const size_t*)&gWI,
            (const size_t*)NULL,
            0,
            0,
            &event);

        events.push_back(event);
    }

    // Technically don't need this as we are doing a blocking read
    // with in-order queue.
    clWaitForEvents(events.size(), &events[0]);


    // Read back computed data from main buffer
    clEnqueueReadBuffer(
        queues[0],
        buffer,
        CL_TRUE,
        0,
        sizeof(float) * NUM_BUFFER_ELEMENTS * numDevices,
        (void*)outputArray,
        0,
        NULL,
        NULL);

    //stop the timer and display to user
    auto end = std::chrono::steady_clock::now();
    std::cout << std::endl << "Program Time Elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n"<< std::endl;

    // Display input array and the averaged results
    int elems;
    std::cout << "Printing Results...\nInput Array:";
    for (elems = 0; elems <  NUM_BUFFER_ELEMENTS; elems++)
      std::cout << " " << inputArray[elems];
    std::cout << std::endl;
    std::cout << "Output Array:";
    for (elems = 0; elems <  NUM_BUFFER_ELEMENTS; elems++)
        std::cout << " " << outputArray[elems];


    std::cout << std::endl;
    return 0;
}
