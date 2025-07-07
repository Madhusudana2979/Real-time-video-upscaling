#include<stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
//#include <device_atomic_functions.hpp>
#include<device_functions.h>
#include<stdlib.h>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include<device_launch_parameters.h>//useful
#include<thread>
#include<chrono>
#include <cuda_fp16.h>  // For half-precision types
#include <algorithm>
#include <functional>
#include <vector>
#include<cuda.h>
#include<math.h>
#include <cmath>
#include <limits>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <numeric>  // Required for std::accumulate
#define BLOCK_SIZE 16
#define WINDOW_SIZE 3
#define HALF_WINDOW (WINDOW_SIZE / 2)



#include "opencv2/opencv.hpp"
#include <stdio.h>
using namespace std;
using namespace cv;


__global__ void OFA_kernel(
    const uint8_t* d_frame1,
    const uint8_t* d_frame2,
    float* d_vx,
    float* d_vy,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Define shared memory for images and gradients
    __shared__ float shared_Ix[BLOCK_SIZE][BLOCK_SIZE];//16x16=256*4 bytes (1024 bytes)
    __shared__ float shared_Iy[BLOCK_SIZE][BLOCK_SIZE];//1024 bytes 
    __shared__ float shared_It[BLOCK_SIZE][BLOCK_SIZE];//1024 bytes 

    // Compute gradients using Sobel operator
    int sobel_x[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    int sobel_y[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

    float Ix = 0.0f, Iy = 0.0f, It = 0.0f;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                uint8_t pixel = d_frame1[(y + i) * width + (x + j)];
                Ix += pixel * sobel_x[i + 1][j + 1];
                Iy += pixel * sobel_y[i + 1][j + 1];
            }
        }
        It = d_frame2[y * width + x] - d_frame1[y * width + x];
    }

    // Store gradients in shared memory
    shared_Ix[threadIdx.x][threadIdx.y] = Ix;
    shared_Iy[threadIdx.x][threadIdx.y] = Iy;
    shared_It[threadIdx.x][threadIdx.y] = It;

    __syncthreads();

    // Compute structure tensor components
    float S_Ix2 = 0.0f, S_Iy2 = 0.0f, S_IxIy = 0.0f;
    float S_IxIt = 0.0f, S_IyIt = 0.0f;
#pragma unroll 
    for (int i = -HALF_WINDOW; i <= HALF_WINDOW; i++) {
        for (int j = -HALF_WINDOW; j <= HALF_WINDOW; j++) {
            int nx = min(max(threadIdx.x + i, 0), BLOCK_SIZE - 1);
            int ny = min(max(threadIdx.y + j, 0), BLOCK_SIZE - 1);

            float Ix_n = shared_Ix[nx][ny];
            float Iy_n = shared_Iy[nx][ny];
            float It_n = shared_It[nx][ny];

            S_Ix2 += Ix_n * Ix_n;
            S_Iy2 += Iy_n * Iy_n;
            S_IxIy += Ix_n * Iy_n;
            S_IxIt += Ix_n * It_n;
            S_IyIt += Iy_n * It_n;
        }
    }

    // Store structure tensor components (for debugging, replace d_vx and d_vy later)
    d_vx[y * width + x] = S_Ix2;  // sum(Ix^2)
    d_vy[y * width + x] = S_Iy2;  // sum(Iy^2)
}


__global__ void warp_frame(
    const uint8_t* d_frame1,   // Reference frame F_t
    const float* d_vx,         // Flow field Vx
    const float* d_vy,         // Flow field Vy
    uint8_t* d_fake_frame,     // Output intermediate frame
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Read motion vector (half-step)
    float vx = d_vx[y * width + x] * 0.5f;
    float vy = d_vy[y * width + x] * 0.5f;

    // Compute new position
    float new_x = x + vx;
    float new_y = y + vy;

    // Bilinear interpolation
    int x0 = floorf(new_x), x1 = x0 + 1;
    int y0 = floorf(new_y), y1 = y0 + 1;

    float wx1 = new_x - x0, wx0 = 1.0f - wx1;
    float wy1 = new_y - y0, wy0 = 1.0f - wy1;

    // Clamp indices to valid range
    x0 = max(0, min(x0, width - 1));
    x1 = max(0, min(x1, width - 1));
    y0 = max(0, min(y0, height - 1));
    y1 = max(0, min(y1, height - 1));

    // Fetch pixel values
    float p00 = d_frame1[y0 * width + x0];
    float p01 = d_frame1[y0 * width + x1];
    float p10 = d_frame1[y1 * width + x0];
    float p11 = d_frame1[y1 * width + x1];

    // Bilinear interpolation
    float interp_val = (p00 * wx0 + p01 * wx1) * wy0 +
        (p10 * wx0 + p11 * wx1) * wy1;

    // Store output
    d_fake_frame[y * width + x] = static_cast<uint8_t>(interp_val);
}

__global__ void bilinear_interpolation(const float* input, float* output,
    int input_width, int input_height,
    int output_width, int output_height,
    float scaling_factor, int blocksize) {

    // Compute output pixel coordinates
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Early bounds check for output coordinates
    if (out_x >= output_width || out_y >= output_height) {
        return;
    }

    // Compute corresponding input pixel coordinates using floating-point arithmetic
    float in_x_f = out_x / scaling_factor;
    float in_y_f = out_y / scaling_factor;

    // Get the integer coordinates for the top-left input pixel
    int in_x = (int)floorf(in_x_f);
    int in_y = (int)floorf(in_y_f);

    // Calculate the fractional parts for bilinear interpolation
    float frac_x = in_x_f - in_x;
    float frac_y = in_y_f - in_y;

    // Ensure we're within bounds for the input image
    if (in_x >= 0 && in_x < input_width - 1 && in_y >= 0 && in_y < input_height - 1) {
        // Get the four pixel values for bilinear interpolation
        float top_left = input[in_y * input_width + in_x];
        float top_right = input[in_y * input_width + (in_x + 1)];
        float bottom_left = input[(in_y + 1) * input_width + in_x];
        float bottom_right = input[(in_y + 1) * input_width + (in_x + 1)];

        // Perform bilinear interpolation
        float top = (1.0f - frac_x) * top_left + frac_x * top_right;
        float bottom = (1.0f - frac_x) * bottom_left + frac_x * bottom_right;
        float result = (1.0f - frac_y) * top + frac_y * bottom;

        // Write the interpolated result to the output
        output[out_y * output_width + out_x] = result;
    }
    else {
        // Handle edge cases - use nearest neighbor for pixels at the image edge
        int safe_x = min(max(0, in_x), input_width - 1);
        int safe_y = min(max(0, in_y), input_height - 1);
        output[out_y * output_width + out_x] = input[safe_y * input_width + safe_x];
    }
}


// Error checking helper function
#define checkCudaErrors(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Utility function to save image data to file
void saveImageData(const std::string& filename, uint8_t* data, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Write PGM header
    file << "P5\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<char*>(data), width * height);
    file.close();

    std::cout << "Image saved to " << filename << std::endl;
}

// Utility function to convert float array to uint8_t
void floatToUint8(float* src, uint8_t* dst, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, src[i])));
    }
}

// Utility function to convert uint8_t array to float
void uint8ToFloat(uint8_t* src, float* dst, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = static_cast<float>(src[i]);
    }
}

// Helper function for asynchronous uint8 to float conversion
void uint8ToFloatAsync(const uint8_t* input, float* output, int size, cudaStream_t stream) {
    // This would ideally be a CUDA kernel, but for simplicity using host code
    // In a real implementation, this should be a CUDA kernel launched with the stream
    for (int i = 0; i < size; i++) {
        output[i] = static_cast<float>(input[i]) / 255.0f;
    }
}

#define NUM_CHANNELS 3    // RGB channels
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

// Structure to hold frame data for processing pipeline
#include <cuda_runtime.h>
#include <iostream>

// Increased number of streams for better overlapping of operations

#define NUM_STREAMS 4
#define BLOCK_SIZE 16      


// Add this to your kernel.cu file
__global__ void merge_and_convert_channels(
    float* r_channel, float* g_channel, float* b_channel,
    unsigned char* output, int width, int height) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        int out_idx = 3 * idx;

        // Convert float (0-255) to uchar with clamping
        output[out_idx] = static_cast<unsigned char>(max(0.0f, min(255.0f, r_channel[idx])));
        output[out_idx + 1] = static_cast<unsigned char>(max(0.0f, min(255.0f, g_channel[idx])));
        output[out_idx + 2] = static_cast<unsigned char>(max(0.0f, min(255.0f, b_channel[idx])));
    }
}

__global__ void merge_and_convert_channels(
    const float* d_r_channel,
    const float* d_g_channel,
    const float* d_b_channel,
    unsigned char* d_output_rgb,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    // Calculate input and output indices
    int pixel_idx = y * width + x;
    int output_idx = 3 * pixel_idx; // RGB has 3 components per pixel

    // Read float values from separate channels
    float r_val = d_r_channel[pixel_idx];
    float g_val = d_g_channel[pixel_idx];
    float b_val = d_b_channel[pixel_idx];

    // Clamp values to [0, 255] range and convert to unsigned char
    unsigned char r = static_cast<unsigned char>(min(max(r_val, 0.0f), 255.0f));
    unsigned char g = static_cast<unsigned char>(min(max(g_val, 0.0f), 255.0f));
    unsigned char b = static_cast<unsigned char>(min(max(b_val, 0.0f), 255.0f));

    // Write to interleaved output array (RGB format)
    d_output_rgb[output_idx] = r;
    d_output_rgb[output_idx + 1] = g;
    d_output_rgb[output_idx + 2] = b;
}


// Convert uint8 data to float data for processing
__global__ void convert_uint8_to_float(
    const uint8_t* input,
    float* output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx] = static_cast<float>(input[idx]);
    }
}


//----------------------------------------------------------------------------------------No optoc flow (180+ FPS)  
int main() {
    // std::string input_video_path = "C:\\Users\\madhu\\Downloads\\resized_video.mp4";
//  std::string input_video_path = "C:\\Users\\madhu\\Downloads\\bike_ride_test_HDR.mp4";

    //std::string input_video_path = "C:\\Users\\madhu\\Downloads\\doom_eternal_resized.mp4";
    std::string input_video_path = "/home/madhu/tensorrt_Test/output_video.webm";
    cv::VideoCapture cap(input_video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return -1;
    }

    int input_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int input_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    float scaling_factor = 4.0f;
    int output_width = input_width * scaling_factor;
    int output_height = input_height * scaling_factor;
    double fps = cap.get(cv::CAP_PROP_FPS);
    const int num_channels = 3;

    // Allocate pinned memory and CUDA resources
    float* h_input[NUM_STREAMS][num_channels];
    float* d_input[NUM_STREAMS][num_channels], * d_output[NUM_STREAMS][num_channels];
    unsigned char* d_display_buffer[NUM_STREAMS];
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t events[NUM_STREAMS];

    for (int i = 0; i < NUM_STREAMS; ++i) {
        for (int c = 0; c < num_channels; ++c) {
            cudaMallocHost(&h_input[i][c], input_width * input_height * sizeof(float));
            cudaMalloc(&d_input[i][c], input_width * input_height * sizeof(float));
            cudaMalloc(&d_output[i][c], output_width * output_height * sizeof(float));
        }

        // Allocate buffer for merged display data on device
        cudaMalloc(&d_display_buffer[i], output_width * output_height * 3 * sizeof(unsigned char));

        cudaStreamCreate(&streams[i]);
        cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);
    }

    // Create Mat for output display (reused for each frame)
    cv::Mat output_frame(output_height, output_width, CV_8UC3);

    cv::Mat temp_frame;
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0, display_count = 0;
    int stream_idx = 0;
    bool stop_processing = false;

    while (cap.read(temp_frame) && !stop_processing) {
        int active_streams = std::min(NUM_STREAMS, 4);  // Dynamically limit streams to avoid bottlenecks

        // Synchronize before using a stream
        if (cudaEventQuery(events[stream_idx]) != cudaSuccess) {
            cudaEventSynchronize(events[stream_idx]);
        }

        std::vector<cv::Mat> channels(3);
        cv::split(temp_frame, channels);

        // Process each channel in parallel
        for (int c = 0; c < num_channels; ++c) {
            cv::Mat float_frame(input_height, input_width, CV_32F, h_input[stream_idx][c]);
            channels[c].convertTo(float_frame, CV_32F);

            // H2D Transfer
            cudaMemcpyAsync(d_input[stream_idx][c], h_input[stream_idx][c],
                input_width * input_height * sizeof(float),
                cudaMemcpyHostToDevice, streams[stream_idx]);

            // Kernel launch
            dim3 block(16, 16);
            dim3 grid((output_width + block.x - 1) / block.x, (output_height + block.y - 1) / block.y);
             bilinear_interpolation << <grid, block, 0, streams[stream_idx] >> > (
          // merged_kernel_with_loops << <grid, block, 0, streams[stream_idx] >> > (
            //<< <grid, block, 0, streams[stream_idx] >> > (
                d_input[stream_idx][c], d_output[stream_idx][c],
                input_width, input_height, output_width, output_height, scaling_factor, BLOCK_SIZE);
        }

        // Convert and merge channels directly on GPU
        dim3 block(16, 16);
        dim3 grid((output_width + block.x - 1) / block.x, (output_height + block.y - 1) / block.y);
        merge_and_convert_channels << <grid, block, 0, streams[stream_idx] >> > (
            d_output[stream_idx][0], d_output[stream_idx][1], d_output[stream_idx][2],
            d_display_buffer[stream_idx], output_width, output_height);

        // Record event
        cudaEventRecord(events[stream_idx], streams[stream_idx]);

        // Display frames after processing
        for (int i = 0; i < active_streams; ++i) {
            if (cudaEventQuery(events[i]) == cudaSuccess) {
                // Show every 4th frame
                if (++display_count % 4 == 0) {
                    // Download only for display - still necessary with OpenCV
                    cudaMemcpy(output_frame.data, d_display_buffer[i],
                        output_width * output_height * 3 * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost);

                    cv::imshow("Upscaled Video", output_frame);
                    int key = cv::waitKey(5);
                    if (key == 'q' || key == 'Q') {
                        stop_processing = true;
                        break;
                    }
                }

                frame_count++;
            }
        }

        auto current_time = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration<double>(current_time - start_time).count() >= 1.0) {
            std::cout << "FPS: " << frame_count << std::endl;
            frame_count = 0;
            start_time = current_time;
        }

        stream_idx = (stream_idx + 1) % active_streams;
    }

    // Cleanup
    cap.release();
    for (int i = 0; i < NUM_STREAMS; ++i) {
        for (int c = 0; c < num_channels; ++c) {
            cudaFreeHost(h_input[i][c]);
            cudaFree(d_input[i][c]);
            cudaFree(d_output[i][c]);
        }
        cudaFree(d_display_buffer[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }
    cv::destroyAllWindows();
    return 0;
}
