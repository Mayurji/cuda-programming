#include <iostream>
#include <vector>
#include <cmath>

// Matrix dimensions 
int N = 10; // Matrix A is N x N, and Vector X is N.


/**
 * @brief CUDA kernel for parallel matrix-vector multiplication.
 * * The matrix-vector multiplication B = A * X, where A is M x N, X is N x 1, 
 * and B is M x 1.
 * * Each thread computes a single element B[row].
 * The row index (i) is calculated based on the global thread ID.
 * * @param d_A Pointer to the matrix A on the device (GPU).
 * @param d_X Pointer to the vector X on the device (GPU).
 * @param d_B Pointer to the result vector B on the device (GPU).
 */
__global__ void matrixVectorMulKernel(const float* d_A, const float* d_X, float* d_B, int N) {
    // Calculate the global thread ID along the X-axis (which corresponds to the row index i)
    // gridDim.x * blockDim.x is the total number of threads in the grid.
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the current thread ID is within the bounds of the matrix rows (M)
    if (row < N) {
        float sum = 0.0f;
        // The current thread calculates the dot product of matrix row 'row' with vector X
        // B[row] = Sum(A[row][col] * X[col]) for col = 0 to N-1
        for (int col = 0; col < N; ++col) {
            // A[row * N + col] is the element A[i][j] in row-major order
            sum += d_A[row * N + col] * d_X[col];
        }
        
        // Write the calculated result to the corresponding element in the result vector B
        d_B[row] = sum;
    }
}

/**
 * @brief Host code (CPU) for setting up data, launching the kernel, and verifying results.
 */
int main() {
    // --- 1. Host (CPU) Data Initialization ---
    // Define the host vectors and matrix
    const int N = 10;
    float *h_A, *h_X, *h_B;

    // Initialize A, X, B with simple values: 
    
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            {
                h_A[i*N + j] = 1.0; // index is generated based on row-major order of the matrix.
            }
        h_B[i] = 0.0f;
        h_X[i] = 2.0f;
    }

    
    std::cout << "Matrix A (" << M << "x" << N << "):" << std::endl;
    for (int i = 0; i < N * N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_A[i * N + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "\nVector X (" << N << "x1):" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_X[i] << std::endl;
    }

    // --- 2. Device (GPU) Memory Allocation ---
    float *d_A, *d_X, *d_B;
    
    // Allocate memory on the GPU for A, X, and B
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_X, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));

    // --- 3. Data Transfer: Host to Device ---
    // Copy initialized A and X from CPU RAM to GPU Global Memory
    cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, h_X, N*sizeof(float), cudaMemcpyHostToDevice);

    // --- 4. Kernel Configuration and Launch ---
    
    // Determine the number of threads per block. Typically a power of 2, often 256 or 512.
    // We choose a small number here since M (4) is very small.
    int threadsPerBlock = 256; 
    
    // Determine the number of blocks needed to cover all M rows.
    // Use ceiling division: (M + threadsPerBlock - 1) / threadsPerBlock
    int numBlocks = (M + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "\nLaunching CUDA Kernel with " << numBlocks << " block(s) and " 
              << threadsPerBlock << " threads per block." << std::endl;
    
    // Launch the kernel. The dimension is (numBlocks, threadsPerBlock).
    matrixVectorMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_X, d_B, N);

    // Check for any launch errors
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        std::cerr << "CUDA Kernel launch failed: " << cudaGetErrorString(kernelError) << std::endl;
        // Clean up resources before exiting
        cudaFree(d_A);
        cudaFree(d_X);
        cudaFree(d_B);
        return 1;
    }

    // Wait for the GPU to finish its work
    cudaDeviceSynchronize();

    // --- 5. Data Transfer: Device to Host ---
    // Copy the result vector B back from GPU Global Memory to CPU RAM
    cudaMemcpy(h_B, d_B, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "\nResult Vector B (M x 1):" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_B[i];
    }
    
    // Free device memory allocated on the GPU
    cudaFree(d_A);
    cudaFree(d_X);
    cudaFree(d_B);

    return 0;
}