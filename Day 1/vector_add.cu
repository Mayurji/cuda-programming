#include <iostream>
#include <cmath>

//host - cpu
//device - gpu
//cpu -> gpu -> cpu

__global__ void vector_add(const float* A, const float* B, float* C, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // n. of grid -> n. of blocks -> n. of threads. (256, 512) 
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main(){
    //Host initialization
    int N = 10;
    float A[N], B[N], C[N];

    //ARRAY values for A and B
    for (int i=0; i<N; ++i){
        A[i] = (float)i + 1.0f;
        B[i] = 2.0f;
    }

    // Declare variable used in Device memory
    float *d_a, *d_b, *d_c;

    // Allocation of memory in device (GPU)
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // Copy from Host (A and B) to device (d_a and d_b)
    cudaMemcpy(d_a, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, N * sizeof(float), cudaMemcpyHostToDevice);

    int blocksize = 256;
    int gridsize = (int)ceil((float)N/blocksize); // 1
    vector_add<<<gridsize, blocksize>>>(d_a, d_b, d_c, N);

    cudaMemcpy(C, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Vector A: ");
    for (int i=0; i<N; ++i){
        std::cout<<A[i]<<(", ");
    }

    printf("\n Vector B: ");
    for (int i=0; i<N; ++i){
        std::cout<<B[i]<<(", ");
    }

    printf("\n Vector C: ");
    for (int i=0; i<N; ++i){
        std::cout<<C[i]<<(", ");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}