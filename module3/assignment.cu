//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h> //srand and rand
#include <math.h>

// add function
__global__ void add(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
	
    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}
    
__global__ void subtract(double *a, double *b, double *c, int n){
                // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
	
    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] - b[id];

}	
	
 __global__ void mult(double *a, double *b, double *c, int n){
            // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
	
    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] * b[id];

}	   
		
__global__ void mod(double *a, double *b, double *c, int n){
      
               // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
	
    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] % b[id];
               
}	


	
int main(int argc, char** argv)
{
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;
	
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}
    
    
    // Host input vectors
    double *h_a;
    double *h_b;
    //Host output vector
    double *h_c;
    
    
    // Device input vectors
    double *d_a;
    double *d_b;
    //Device output vector
    double *d_c;
    
    // Size, in bytes, of each vector
    size_t bytes = totalThreads*sizeof(double);
    
    
    // Allocate memory for each vector on host
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);
    
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
 
    //initialize the vectors
    for(int i = 0;i<totalThreads;i++){
         //first array is 0 through number of threads
		 h_a[i] = i;
		// second array is a random number between 0 and 3
			h_b[i] = rand() % 4;                               
    }
	
	
	//copy both arrays from host to device
	// Copy host vectors to device
    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);
	
	
	
	
	
	//performing add function
	//add<<<numBlocks, totalThreads>>>(d_a, d_b, d_c, totalThreads); 
	
	//performing subtract function
	//subtract<<<numBlocks, totalThreads>>>(d_a, d_b, d_c, totalThreads); 
	
	//performing mult function
	mult<<<numBlocks, totalThreads>>>(d_a, d_b, d_c, totalThreads); 
	
	//performing mod fuction
	
	//copy the array from device to host
	cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost); 
	
	for(int i = 0; i < 7; i++){
		printf("%f ",h_a[i]); 

	}
	printf("\n"); 
	for(int i = 0; i < 7; i++){
		printf("%f ",h_b[i]); 
	}
	printf("\n"); 
	for(int i = 0; i < 7; i++){
		printf("%f ",h_c[i]); 
	}
	
	
	cudaFree(d_a); 
	cudaFree(d_b); 
	cudaFree(d_c); 
                         
	
                                      
                                      
    return 0;  
    
}