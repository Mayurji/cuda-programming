#include <iostream>
#include <cmath>

__global__ void matrixTranspose(const float *input, float *output, int width, int height){

    int x = blockIdx.x * blockDim.x + threadIdx.x; // Column index of the matrix
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Row index of the matrix

    if (x < width && y < height){
        int inputIndex = y * width + x;
        int outputIndex = x * height + y;
        output[outputIndex] = input[inputIndex];
    }
}

int main(){
    int width = 10;
    int height = 10;

    // float *h_input, *h_output;

    float *h_input = (float *)malloc(width * height * sizeof(float));
    float *h_output = (float *)malloc(width * height * sizeof(float));

    for (int i=0; i<width * height; i++){
        h_input[i] = (float)i;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));

    cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocksize(10, 10);
    dim3 gridsize((width + blocksize.x - 1) / blocksize.x, (height + blocksize.x - 1) / blocksize.y);
    matrixTranspose<<<gridsize, blocksize>>>(d_input, d_output, width, height);

    cudaMemcpy(h_output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Input Matrix: \n");
    for (int i=0; i<width * height; i++){
        std::cout<< h_input[i] <<", ";
    }

    printf("Output Matrix: \n");
    for (int i=0; i<width * height; i++){
        std::cout<< h_output[i] <<", ";
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}