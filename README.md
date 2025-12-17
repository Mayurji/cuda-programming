# Cuda Programming

Implementation of CUDA programs from the book: "Programming Massively Parallel Processors"

**Follow the implementation on YouTube Playlist**

[Master CUDA Programming: From Zero to Hero](https://www.youtube.com/playlist?list=PLVVBQldz3m5v1VDhlCyB1DhPfjREsJWmf)

<!-- GitAds-Verify: 2YZIV9D1CW1GCHBR14RXRSFPURU7OVG1 -->

## Why CUDA?

- It leverages the parallel processing power of NVIDIA GPUs for high-performance computing.

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

# Day 5

## Write a CUDA program for Layer Normalization In a Neural Network (Transformer Architecture)

**Layer Normalization is a technique commonly used in Neural Networks (especially Transformers) to normalize the inputs across the features (columns) of an input sample (row).**

During neural network training, we use layer normalization to stabilise the training process.

<img src="images/layer_norm.png" alt="Layer normalization" width="400"/>

**2D into Row-Major Order**

<img src="images/row-maj-ord.png" alt="2d-row-major-order" width="500"/>

## 🌟 Sample 3x3 Layer Normalization Walkthrough

Let's assume the input matrix **A** (3 rows, 3 columns) is the following:

$$
\mathbf{A} = \begin{pmatrix}
1 & 2 & 3 \\
10 & 11 & 12 \\
0 & 0.5 & 1
\end{pmatrix}
$$

The Layer Normalization is applied **independently to each row** (sample) of the matrix.

---

### Row 1: (1, 2, 3)

The normalization steps (Mean, Variance, Normalize) are applied to the elements $\mathbf{A}_{1} = (1, 2, 3)$.

1.  **Calculate Mean ($\mu$):**
    $$
    \mu = \frac{1 + 2 + 3}{3} = \frac{6}{3} = \mathbf{2.0}
    $$
2.  **Calculate Variance ($\sigma^2$):**
    $$
    \sigma^2 = \frac{(1-2)^2 + (2-2)^2 + (3-2)^2}{3} = \frac{(-1)^2 + 0^2 + 1^2}{3} = \frac{1 + 0 + 1}{3} = \frac{2}{3} \approx \mathbf{0.6667}
    $$
3.  **Calculate Standard Deviation ($\sigma$):** (Ignoring the small $\epsilon=1\mathrm{e}{-7}$ for simplicity)
    $$
    \sigma = \sqrt{\sigma^2} = \sqrt{\frac{2}{3}} \approx \mathbf{0.8165}
    $$
4.  **Normalize (Output Row $\mathbf{B}_{1}$):**
    $$
    \mathbf{B}_{1} = \left(\frac{1-2}{0.8165}, \frac{2-2}{0.8165}, \frac{3-2}{0.8165}\right) \approx (-1.22, 0, 1.22)
    $$

---

### Row 2: (10, 11, 12)

The normalization steps are applied to the elements $\mathbf{A}_{2} = (10, 11, 12)$.

1.  **Calculate Mean ($\mu$):**
    $$
    \mu = \frac{10 + 11 + 12}{3} = \frac{33}{3} = \mathbf{11.0}
    $$
2.  **Calculate Variance ($\sigma^2$):**
    $$
    \sigma^2 = \frac{(10-11)^2 + (11-11)^2 + (12-11)^2}{3} = \frac{(-1)^2 + 0^2 + 1^2}{3} = \frac{2}{3} \approx \mathbf{0.6667}
    $$
3.  **Calculate Standard Deviation ($\sigma$):**
    $$
    \sigma = \sqrt{\frac{2}{3}} \approx \mathbf{0.8165}
    $$
4.  **Normalize (Output Row $\mathbf{B}_{2}$):**
    $$
    \mathbf{B}_{2} = \left(\frac{10-11}{0.8165}, \frac{11-11}{0.8165}, \frac{12-11}{0.8165}\right) \approx (-1.22, 0, 1.22)
    $$

---

### Row 3: (0, 0.5, 1)

The normalization steps are applied to the elements $\mathbf{A}_{3} = (0, 0.5, 1)$.

1.  **Calculate Mean ($\mu$):**
    $$
    \mu = \frac{0 + 0.5 + 1}{3} = \frac{1.5}{3} = \mathbf{0.5}
    $$
2.  **Calculate Variance ($\sigma^2$):**
    $$
    \sigma^2 = \frac{(0-0.5)^2 + (0.5-0.5)^2 + (1-0.5)^2}{3} = \frac{0.25 + 0 + 0.25}{3} = \frac{0.5}{3} \approx \mathbf{0.1667}
    $$
3.  **Calculate Standard Deviation ($\sigma$):**
    $$
    \sigma = \sqrt{\sigma^2} = \sqrt{\frac{0.5}{3}} \approx \mathbf{0.4082}
    $$
4.  **Normalize (Output Row $\mathbf{B}_{3}$):**
    $$
    \mathbf{B}_{3} = \left(\frac{0-0.5}{0.4082}, \frac{0.5-0.5}{0.4082}, \frac{1-0.5}{0.4082}\right) \approx (-1.22, 0, 1.22)
    $$

---

## Final Input and Output

### Sample Input Matrix $\mathbf{A}$ (Host Array `A` and Device Array `d_a`)
$$
\mathbf{A} = \begin{pmatrix}
1.00 & 2.00 & 3.00 \\
10.00 & 11.00 & 12.00 \\
0.00 & 0.50 & 1.00
\end{pmatrix}
$$

### Expected Output Matrix $\mathbf{B}$ (Host Array `B` and Device Array `d_b`)
$$
\mathbf{B} \approx \begin{pmatrix}
-1.22 & 0.00 & 1.22 \\
-1.22 & 0.00 & 1.22 \\
-1.22 & 0.00 & 1.22
\end{pmatrix}
$$

**Observation:** Notice that although the input rows had vastly different magnitudes (1 to 3 vs. 10 to 12), the output rows have the **same mean (0) and standard deviation (1)**, demonstrating how Layer Normalization scales the data to a uniform distribution within each row.

# Day 6

## Matrix Transpose

<img src="images/matrix_transpose.png" alt="matrix-transpose" width="850"/>

### **Mapping Thread to Matrix indices**

![alt text](images/image-6.png)