#include <iostream>
#include <cmath> // For ceil and sqrt

// Define the matrix dimension N (10x10)
const int N = 10;

// Device kernel function for matrix addition
// It uses a 2D block and 2D grid structure.
__global__ void MatrixAdd_B(int *A, int *B, int *C, int size) {
    // Calculate the global 2D index (i, j) for the current thread
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    // Boundary check: Only process threads that fall within the NxN matrix bounds
    if (i < size && j < size) {
        // Calculate the 1D index
        int index = i * size + j;
        // Perform the addition
        C[index] = A[index] + B[index];
    }
}

// Host utility function to print the matrix
void printMatrix(const int *matrix, int size, const char* name) {
    std::cout << name << " (" << size << "x" << size << "):\n";
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            // Print the element at 1D index i * size + j
            std::cout << matrix[i * size + j] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    // 1. Setup Host Data
    const int matrix_size = N * N * sizeof(int);

    // Host matrices: h_a, h_b (inputs), h_c (output)
    int *h_a = new int[N * N];
    int *h_b = new int[N * N];
    int *h_c = new int[N * N];

    // Initialize host matrices with simple data: A[i, j] = 1, B[i, j] = 2
    for (int i = 0; i < N * N; ++i) {
        h_a[i] = 1; // All elements of A are 1
        h_b[i] = 2; // All elements of B are 2
        h_c[i] = 0; // Initialize result C to 0
    }

    // 2. Allocate Device Memory
    int *d_a, *d_b, *d_c;
    // cudaMalloc is used to allocate memory on the GPU (device)
    cudaMalloc((void**)&d_a, matrix_size);
    cudaMalloc((void**)&d_b, matrix_size);
    cudaMalloc((void**)&d_c, matrix_size);

    // 3. Copy Input Data from Host to Device
    // cudaMemcpy is used to transfer data from CPU (Host) to GPU (Device)
    cudaMemcpy(d_a, h_a, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, matrix_size, cudaMemcpyHostToDevice);

    // 4. Define Launch Configuration (Grid and Block)
    // The dimensions are chosen to ensure the total number of threads is >= N*N
    // N=10, Threads per block = 32x16 = 512
    dim3 dimBlock(32, 16);

    // Grid dimensions: ceil(N/blockDim.x) and ceil(N/blockDim.y)
    // GridX = ceil(10/32.0) = 1
    // GridY = ceil(10/16.0) = 1
    dim3 dimGrid(
        static_cast<int>(std::ceil(N / (float)dimBlock.x)),
        static_cast<int>(std::ceil(N / (float)dimBlock.y))
    );

    // Total Threads Launched = 1 * 1 * 512 = 512
    // Active Threads (doing useful work) = 10 * 10 = 100

    std::cout << "--- CUDA Matrix Addition (N=" << N << ") ---\n";
    std::cout << "Block Dimensions (Threads/Block): (" << dimBlock.x << ", " << dimBlock.y << ")\n";
    std::cout << "Grid Dimensions (Blocks/Grid): (" << dimGrid.x << ", " << dimGrid.y << ")\n";
    std::cout << "Total Threads Launched: " << dimBlock.x * dimBlock.y * dimGrid.x * dimGrid.y << "\n";
    std::cout << "Active Threads (Matrix Elements): " << N * N << "\n\n";

    // 5. Launch the Kernel
    // Syntax: kernelName<<<dimGrid, dimBlock>>>(arg1, arg2, ...);
    MatrixAdd_B<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);

    // Check for any launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Kernel launch failed: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    // Wait for the device to finish its work
    cudaDeviceSynchronize();

    // 6. Copy Result Data from Device to Host
    // cudaMemcpy is used to transfer data from GPU (Device) to CPU (Host)
    cudaMemcpy(h_c, d_c, matrix_size, cudaMemcpyDeviceToHost);

    // 7. Verification and Cleanup
    printMatrix(h_a, N, "Input A");
    printMatrix(h_b, N, "Input B");
    printMatrix(h_c, N, "Result C (A + B)"); // Should be all 3s (1+2)

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}