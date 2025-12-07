# Cuda Programming

Implementation of CUDA programs from the book: "Programming Massively Parallel Processors"

<!-- GitAds-Verify: 2YZIV9D1CW1GCHBR14RXRSFPURU7OVG1 -->

## Why CUDA?

- It leverages the parallel processing power of NVIDIA GPUs for high-performance computing.

### Watch on YouTube

[Master CUDA Programming: Zero to Hero](https://www.youtube.com/playlist?list=PLVVBQldz3m5v1VDhlCyB1DhPfjREsJWmf)

# Day 1

## Task - Parallel Vector Addition

![alt text](images/image-1.png)

## How to compile the CUDA program

```
nvcc vector_add.cu -i vector_add #generates executed output in vector_add
./vector_add
```

## How to run Cuda Program in Google Colab

**Steps**

1. Switch the Runtime type to GPU.
2. Check nvcc version and nvidia-smi.
3. Install and load nvcc4jupyter
4. Write your cuda Program.
5. Compile the CUDA program.
6. View the results.

# Day 2

## Write a CUDA program for Matrix Addition

Key Learning: How does a data structure gets mapped to threads in grid block?

Adding 2x2 matrix.

![alt text](images/image-3.png)

**2D Grid of Block**

We use the 2D properties of the CUDA grid and block to determine the row ($i$) and column ($j$) indices for the matrix:

![alt text](images/image-4.png)

**Matrix-Indices**

![alt text](images/image-2.png)

**Combining Thread to Matrix indices**

![alt text](images/image-6.png)

# Day 3

## Write a CUDA program for matrix-vector multiplication

Matrix-vector multiplication is an operation where a matrix and a vector are combined to produce a new vector. It's defined only when the number of columns in the matrix equals the number of rows (elements) in the vector. The resulting vector is a linear combination of the matrix's columns, where the components of the input vector act as the scalar coefficients.

![](images/image-7.png)

**Index-wise Computation**

![alt text](images/image-8.png)

# Day 4

## Write a CUDA Program for Simplified Block-wise Prefix Sum 

<img src="images/shared_mem.png" alt="Shared memory" width="400"/>


It computes the cumulative sum (or other associative binary operation) of elements in an array in parallel.

Given an input array:

```
[ a0, a1, a2, a3, ... , an ]
```

The inclusive prefix sum (scan) computes an output array as follows:

```
[ a0,
  a0+a1,
  a0+a1+a2,
  a0+a1+a2+a3, ... ]
```
So each output element i is the sum of all elements up to and including index i.

### **Why Parallelization Matters?**

A sequential prefix sum requires O(n) steps:

```
out[i] = out[i-1] + in[i]
```
Each element depends on the previous one → inherently sequential.

The magic of parallel scan is transforming this dependency into a form that can be computed in O(log n) time using n processors, by restructuring the computation as a tree.

### Implement a Simplified Block-wise Prefix Sum

It leverages CUDA capabilities to perform

     - Concurrent Execution: From O(n) to O(log n)
     - Shared Memory: Accessing results(partial sum) performed by other threads in previous iteration into current thread.
     - Coalesced Memory Access: Single thread reads multiple locations.

*Coalescing is a memory optimization technique where the GPU hardware automatically combines multiple small memory requests from a group of threads (a warp) into a single, larger, and highly efficient memory transaction.*

![alt text](images/memory_coalesce.png)

### Input and Kernel Configuration

**Input (h_input) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}**

**Block Size (BlockDim.x) = 8 threads**

**Grid Size: 2 Blocks(block 0 handles indices 0-7 and block 1 handles indices 8-15)**

**Step 1: Initial Loading and Combine(Coalescing)**

![alt text](images/load_coalesce.png)

We store the results in the shared memory after loading.

**Step 2: Parallel Inclusive Scan**

Iterate through block of threads in a logarithmic loop.

`for (int stride=1; stride < BlockDim.x; stride *= 2)`

Iteration 1: *stride = 1*

if Thread with tid >= 1 read from (tid - 1)

![alt text](images/scan.png)


Iteration 2: *stride = 2*

if Thread with tid >= 2 read from (tid - 2)

Iteration 3: *stride = 4*

if Thread with tid >= 4 read from (tid - 4)

**Shared memory after scan**

S = {10, 22, 36, 52, 70, 90, 112, 136}

Step 3: **Write the shared memory to global memory**
