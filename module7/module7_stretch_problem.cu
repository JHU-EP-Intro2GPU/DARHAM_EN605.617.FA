#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h> // for pow()

#define NUM_THREADS 8
#define NUM_BLOCKS 1


__host__ void generate_rand_data(unsigned int * data, unsigned int num_elements)
{
	for(unsigned int i=0; i < num_elements; i++)
	{
		data[i] = rand() % 4; //PLACE YOUR CODE HERE
	}
}

__device__ void copy_data_to_shared(unsigned int * const data,
			unsigned int * const shared_tmp,
			const unsigned int tid)
{
	// Copy data into shared memory
	shared_tmp[tid] = data[tid];
	__syncthreads();
}

__device__ void simple_squaring_operation(unsigned int * const data,
				const unsigned int tid)
{
	//square the mem value and overwrite
	data[tid] = data[tid] * data[tid]; //PLACE YOUR CODE HERE
}

__global__ void gpu_register_array_operation(unsigned int * const data, const unsigned int num_elements)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	//perform some simple operation
	simple_squaring_operation(data, tid);
}

__global__ void gpu_shared_array_operation(unsigned int * const data, const unsigned int num_elements)
{
	const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	//allocate shared memory
	__shared__ unsigned int shared_tmp[NUM_THREADS];

	//make a copy of the global device data into the shared device memory
	copy_data_to_shared(data, shared_tmp, tid);

	//perform some simple operation
	simple_squaring_operation(shared_tmp, tid);

	//push updated shared mem back to the initial global data mem
	data[tid] = shared_tmp[tid];
}

//tier 2 method for printing a specific cuda device properties
//already called by the tier 1 method
void print_all_device_properties(int device_id){
	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, device_id);
	// printf("============Start Device %x============\n", device_id);
	// printf("Name:                          %s\n",  prop.name);
	// printf("Total global memory:           %lu\n",  prop.totalGlobalMem);
	// printf("Total shared memory per block: %lu\n",  prop.sharedMemPerBlock);
	// printf("Total registers per block:     %lu\n",  (unsigned long)prop.regsPerBlock);
	// printf("Warp size:                     %lu\n",  (unsigned long)prop.warpSize);
	// printf("Maximum memory pitch:          %lu\n",  prop.memPitch);
	// printf("Maximum threads per block:     %lu\n",  (unsigned long)prop.maxThreadsPerBlock);
	// for (int i = 0; i < 3; ++i)
	// printf("Maximum dimension %d of block: %lu\n", i, (unsigned long)prop.maxThreadsDim[i]);
	// for (int i = 0; i < 3; ++i)
	// printf("Maximum dimension %d of grid:  %lu\n", i, (unsigned long)prop.maxGridSize[i]);
	// printf("Total constant memory:         %lu\n",  prop.totalConstMem);
	// printf("Major revision number:         %lu\n",  (unsigned long)prop.major);
	// printf("Minor revision number:         %lu\n",  (unsigned long)prop.minor);
	// printf("Clock rate:                    %lu\n",  (unsigned long)prop.clockRate);
	// printf("Texture alignment:             %lu\n",  prop.textureAlignment);
	// printf("Concurrent copy and execution: %s\n",  (prop.deviceOverlap ? "Yes" : "No"));
	// printf("Number of multiprocessors:     %lu\n",  (unsigned long)prop.multiProcessorCount);
	// printf("Kernel execution timeout:      %s\n",  (prop.kernelExecTimeoutEnabled ? "Yes" : "No"));
	// printf("Integrated:                    %s\n",  (prop.integrated ? "Yes" : "No"));
	// printf("Mapable Host Memory:           %s\n",  (prop.canMapHostMemory ? "Yes" : "No"));
	// printf("Compute Mode:                  %d\n",  prop.computeMode);
	// printf("Concurrent Kernels:            %d\n",  prop.concurrentKernels);
	// printf("ECC Enabled:                   %s\n",  (prop.ECCEnabled ? "Yes" : "No"));
	// printf("pci Bus ID:                    %lu\n",  (unsigned long)prop.pciBusID);
	// printf("pci Device ID:                 %lu\n",  (unsigned long)prop.pciDeviceID);
	// printf("Using a tcc Driver:            %s\n",  (prop.tccDriver ? "Yes" : "No"));
	// printf("============End Device %x============\n", device_id);

	printf("============Start Device %x============\n", device_id);
	printf("Name:                          %s\n",  prop.name);
	printf("Total global memory:           %lu\n",  prop.totalGlobalMem);
	printf("Total shared memory per block: %lu\n",  prop.sharedMemPerBlock);
	printf("Total registers per block:     %d\n",  prop.regsPerBlock);
	printf("Warp size:                     %d\n",  prop.warpSize);
	printf("Maximum memory pitch:          %lu\n",  prop.memPitch);
	printf("Maximum threads per block:     %d\n",  prop.maxThreadsPerBlock);
	for (int i = 0; i < 3; ++i)
	printf("Maximum dimension %d of block: %d\n", i, prop.maxThreadsDim[i]);
	for (int i = 0; i < 3; ++i)
	printf("Maximum dimension %d of grid:  %d\n", i, prop.maxGridSize[i]);
	printf("Total constant memory:         %lu\n",  prop.totalConstMem);
	printf("Major revision number:         %d\n",  prop.major);
	printf("Minor revision number:         %d\n",  prop.minor);
	printf("Clock rate:                    %d\n",  prop.clockRate);
	printf("Texture alignment:             %lu\n",  prop.textureAlignment);
	printf("Concurrent copy and execution: %s\n",  (prop.deviceOverlap ? "Yes" : "No"));
	printf("Number of multiprocessors:     %d\n",  prop.multiProcessorCount);
	printf("Kernel execution timeout:      %s\n",  (prop.kernelExecTimeoutEnabled ? "Yes" : "No"));
	printf("Integrated:                    %s\n",  (prop.integrated ? "Yes" : "No"));
	printf("Mapable Host Memory:           %s\n",  (prop.canMapHostMemory ? "Yes" : "No"));
	printf("Compute Mode:                  %d\n",  prop.computeMode);
	printf("Concurrent Kernels:            %d\n",  prop.concurrentKernels);
	printf("ECC Enabled:                   %s\n",  (prop.ECCEnabled ? "Yes" : "No"));
	printf("pci Bus ID:                    %d\n",  prop.pciBusID);
	printf("pci Device ID:                 %d\n",  prop.pciDeviceID);
	printf("Using a tcc Driver:            %s\n",  (prop.tccDriver ? "Yes" : "No"));
	printf("============End Device %x============\n", device_id);
}

//tier 1 method for printing all cuda devices and their properties
void print_all_CUDA_devices_and_properties() {
  int device_id;
  cudaGetDeviceCount( &device_id);
  printf("Print of all CUDA devices and device properties\n");
	for (int i = 0; i < device_id; i++){
		//states that cudaDeviceProp returns a 25 data types in a struct
		print_all_device_properties(device_id);
	}
}

__host__ float execute_register_memory_operations(void)
{
	const unsigned int num_elements = NUM_THREADS;
	const unsigned int num_bytes = NUM_THREADS * sizeof(unsigned int);
	unsigned int * d_data;		//device data
	unsigned int hi_data[num_elements];		//initial host data
	unsigned int hf_data[num_elements];		//final host data

	/* Set timing Metrics */
	cudaEvent_t kernel_start, kernel_stop;
	float delta = 0.0F;
	cudaEventCreate(&kernel_start,0);
	cudaEventCreate(&kernel_stop,0);

	//set CUDA stream
  	cudaStream_t stream;
  	cudaStreamCreate(&stream);

	//start timing metric
	cudaEventRecord(kernel_start, 0);

	//device memory alloc
	cudaMalloc(&d_data, num_bytes);

	//populate the initial host array with random data
	generate_rand_data(hi_data, num_elements);

	//copy from host memory to device memory
	cudaMemcpy(d_data, hi_data, num_bytes, cudaMemcpyHostToDevice);

	//Call GPU kernel <<<BLOCK TOTAL, THREADS TOTAL>>>
	gpu_register_array_operation<<<NUM_BLOCKS, NUM_THREADS>>>(d_data, num_elements);

 	cudaStreamSynchronize(stream);	// Wait for the GPU launched work to complete
	cudaGetLastError();

	//copy from device to host memory
	cudaMemcpy(hf_data, d_data, num_bytes, cudaMemcpyDeviceToHost);

	//end timing metric
	cudaEventRecord(kernel_stop, 0);
	cudaEventSynchronize(kernel_stop);
	cudaEventElapsedTime(&delta, kernel_start, kernel_stop);
	cudaEventDestroy(kernel_start);
	cudaEventDestroy(kernel_stop);

	//console print the host data after the GPU kernal
	for (int i = 0; i < num_elements; i++){
		printf("Input value: %x, device output: %x\n", hi_data[i], hf_data[i]);
	}

	//free device and host memory allocations
	//PLACE YOUR CODE HERE

	return delta;
}


__host__ float execute_shared_memory_operations()
{
	const unsigned int num_elements = NUM_THREADS;
	const unsigned int num_bytes = NUM_THREADS * sizeof(unsigned int);
	unsigned int * d_data;		//device data
	unsigned int hi_data[num_elements];		//initial host data
	unsigned int hf_data[num_elements];		//final host data

	/* Set timing Metrics */
	cudaEvent_t kernel_start, kernel_stop;
	float delta = 0.0F;
	cudaEventCreate(&kernel_start,0);
	cudaEventCreate(&kernel_stop,0);

	//set CUDA stream
  cudaStream_t stream; //PLACE YOUR CODE HERE
	cudaStreamCreate(&stream);

	//start timing metric
	cudaEventRecord(kernel_start, 0);

	//device memory alloc
	cudaMalloc(&d_data, num_bytes);

	//populate the initial host array with random data
	generate_rand_data(hi_data, num_elements);

	//copy from host memory to device memory
	cudaMemcpy(d_data, hi_data, num_bytes, cudaMemcpyHostToDevice);

	//Call GPU kernels <<<BLOCK TOTAL, THREADS TOTAL>>>
	gpu_shared_array_operation<<<NUM_BLOCKS, NUM_THREADS>>>(d_data, num_elements);

	//sync the cuda stream
 	cudaStreamSynchronize(stream);	// Wait for the GPU launched work to complete
	cudaGetLastError();				//error handling

	//copy from device to host memory
	cudaMemcpy(hf_data, d_data, num_bytes, cudaMemcpyDeviceToHost);

	//end timing metric
	cudaEventRecord(kernel_stop, 0);
	cudaEventSynchronize(kernel_stop);
	cudaEventElapsedTime(&delta, kernel_start, kernel_stop);
	cudaEventDestroy(kernel_start);
	cudaEventDestroy(kernel_stop);

	//console print the host data after the GPU kernal
	for (int i = 0; i < num_elements; i++){
		printf("Input value: %x, device output: %x\n", hi_data[i], hf_data[i]);
	}

	//free device and host memory allocations
	cudaFree((void* ) d_data);		//free devide data
	cudaFreeHost(hi_data);	 			//free up the host memory
	cudaFreeHost(hf_data);	 			//free up the host memory
	cudaDeviceReset();

	return delta;
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {
	//print all cuda devices and device properties for kicks
  print_all_CUDA_devices_and_properties();

  //test harness for timing some kernels using streams and events
	float delta_shared = execute_shared_memory_operations(); //PLACE YOUR CODE HERE TO USE SHARED MEMORY FOR OPERATIONS
	float delta_register = execute_register_memory_operations();//PLACE YOUR CODE HERE TO USE REGISTER MEMORY FOR OPERATIONS

	//print out the results of the time executions returned by the prev methods
	printf("========================\n");
	printf("Summary\n");
	printf("Total Threads: %d\n", NUM_THREADS);
	printf("Total Blocks: %d\n", NUM_BLOCKS);
	printf("========================\n");
	printf("Time to copy global to shared mem, perform simple operation w/ shared memory, copy memory back to global\n");
	printf("duration: %fms\n",delta_shared);
	printf("========================\n");
	printf("Time to copy global to register mem, perform simple operation w/ register memory, copy memory back to global\n");
	printf("duration: %fms\n",delta_register);
	return 0;
}
