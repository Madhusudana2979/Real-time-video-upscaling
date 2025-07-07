#include<stdio.h> 
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



int main() {
   // std::string input_video_path = "C:\\Users\\madhu\\Downloads\\doom_eternal_resized.mp4";
   // std::string input_video_path = "C:\\Users\\madhu\\Downloads\\bike_ride_test_HDR.mp4";
    std::string input_video_path = "/home/madhu/tensorrt_Test/output_video.webm";
  //  std::string input_video_path = "C:\\Users\\madhu\\Downloads\\one_piece_kaido_defeat_test.mp4";

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

    // Performance metrics tracking
    int total_frames = 0;
    double total_processing_time = 0.0;
    double min_frame_time = std::numeric_limits<double>::max();
    double max_frame_time = 0.0;

    // Create CUDA events for detailed timing
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    // Create events for measuring different stages
    cudaEvent_t flow_start, flow_stop, warp_start, warp_stop, upscale_start, upscale_stop;
    cudaEventCreate(&flow_start);
    cudaEventCreate(&flow_stop);
    cudaEventCreate(&warp_start);
    cudaEventCreate(&warp_stop);
    cudaEventCreate(&upscale_start);
    cudaEventCreate(&upscale_stop);

    // Performance statistics
    double total_flow_time = 0.0, total_warp_time = 0.0, total_upscale_time = 0.0;

    // Allocate pinned memory and CUDA resources
    float* h_input[NUM_STREAMS][num_channels];
    float* d_input[NUM_STREAMS][num_channels], * d_output[NUM_STREAMS][num_channels];
    unsigned char* d_display_buffer[NUM_STREAMS];
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t events[NUM_STREAMS];

    // Allocate memory for optical flow and frame warping
    cv::Mat prev_frame, curr_frame, frame_gray, prev_gray;
    uint8_t* d_frame1[NUM_STREAMS], * d_frame2[NUM_STREAMS]; // For grayscale frames
    float* d_vx[NUM_STREAMS], * d_vy[NUM_STREAMS]; // For flow fields

    // For generating intermediate frames (per channel)
    uint8_t* d_prev_color[NUM_STREAMS][num_channels];
    uint8_t* d_curr_color[NUM_STREAMS][num_channels];
    uint8_t* d_fake_color[NUM_STREAMS][num_channels];

    // For upscaled fake frames
    float* d_fake_float[NUM_STREAMS][num_channels];
    float* d_fake_upscaled[NUM_STREAMS][num_channels];

    // Host memory for displaying the results
    float* h_vx, * h_vy; // For flow visualization

    // Allocate host memory for flow visualization
    cudaMallocHost(&h_vx, input_width * input_height * sizeof(float));
    cudaMallocHost(&h_vy, input_width * input_height * sizeof(float));

    for (int i = 0; i < NUM_STREAMS; ++i) {
        // Allocate memory for regular video processing
        for (int c = 0; c < num_channels; ++c) {
            cudaMallocHost(&h_input[i][c], input_width * input_height * sizeof(float));
            cudaMalloc(&d_input[i][c], input_width * input_height * sizeof(float));
            cudaMalloc(&d_output[i][c], output_width * output_height * sizeof(float));

            // Allocate memory for RGB channel warping
            cudaMalloc(&d_prev_color[i][c], input_width * input_height * sizeof(uint8_t));
            cudaMalloc(&d_curr_color[i][c], input_width * input_height * sizeof(uint8_t));
            cudaMalloc(&d_fake_color[i][c], input_width * input_height * sizeof(uint8_t));

            // Allocate memory for upscaling fake frames
            cudaMalloc(&d_fake_float[i][c], input_width * input_height * sizeof(float));
            cudaMalloc(&d_fake_upscaled[i][c], output_width * output_height * sizeof(float));
        }

        // Allocate memory for optical flow computation
        cudaMalloc(&d_frame1[i], input_width * input_height * sizeof(uint8_t));
        cudaMalloc(&d_frame2[i], input_width * input_height * sizeof(uint8_t));
        cudaMalloc(&d_vx[i], input_width * input_height * sizeof(float));
        cudaMalloc(&d_vy[i], input_width * input_height * sizeof(float));

        // Allocate buffer for merged display data on device
        cudaMalloc(&d_display_buffer[i], output_width * output_height * 3 * sizeof(unsigned char));

        cudaStreamCreate(&streams[i]);
        cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);
    }

    // Create Mat for output display
    cv::Mat output_frame(output_height, output_width, CV_8UC3);
    cv::Mat flow_vis(input_height, input_width, CV_8UC3); // For flow visualization

    // For original and fake frames
    cv::Mat upscaled_original(output_height, output_width, CV_8UC3);
    cv::Mat fake_frame(input_height, input_width, CV_8UC3);  // For intermediate color frame
    cv::Mat upscaled_fake_frame(output_height, output_width, CV_8UC3); // For upscaled fake frame

    // Get first frame to initialize
    cap.read(prev_frame);
    cv::cvtColor(prev_frame, prev_gray, cv::COLOR_BGR2GRAY);

    // Timer for total operation time and FPS calculation
    auto total_start_time = std::chrono::high_resolution_clock::now();

    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0, display_count = 0;
    int stream_idx = 0;
    bool stop_processing = false;

    while (cap.read(curr_frame) && !stop_processing) {
        int active_streams = std::min(NUM_STREAMS, 4);  // Dynamically limit streams to avoid bottlenecks

        // Start timing this frame
        cudaEventRecord(start_event, streams[stream_idx]);

        // Synchronize before using a stream
        if (cudaEventQuery(events[stream_idx]) != cudaSuccess) {
            cudaEventSynchronize(events[stream_idx]);
        }

        // Convert current frame to grayscale for optical flow
        cv::cvtColor(curr_frame, frame_gray, cv::COLOR_BGR2GRAY);

        // Upload grayscale frames to device
        cudaMemcpyAsync(d_frame1[stream_idx], prev_gray.data,
            input_width * input_height * sizeof(uint8_t),
            cudaMemcpyHostToDevice, streams[stream_idx]);
        cudaMemcpyAsync(d_frame2[stream_idx], frame_gray.data,
            input_width * input_height * sizeof(uint8_t),
            cudaMemcpyHostToDevice, streams[stream_idx]);

        // Begin timing optical flow computation
        cudaEventRecord(flow_start, streams[stream_idx]);

        // Compute optical flow
        dim3 flow_block(16, 16);
        dim3 flow_grid((input_width + flow_block.x - 1) / flow_block.x,
            (input_height + flow_block.y - 1) / flow_block.y);
        OFA_kernel << <flow_grid, flow_block, 0, streams[stream_idx] >> > (
            d_frame1[stream_idx], d_frame2[stream_idx],
            d_vx[stream_idx], d_vy[stream_idx],
            input_width, input_height
            );

        // End timing optical flow computation
        cudaEventRecord(flow_stop, streams[stream_idx]);

        // Split previous and current frames into channels and upload
        std::vector<cv::Mat> prev_channels(3), curr_channels(3);
        cv::split(prev_frame, prev_channels);
        cv::split(curr_frame, curr_channels);

        // Begin timing frame warping
        cudaEventRecord(warp_start, streams[stream_idx]);

        // Process each color channel
        for (int c = 0; c < num_channels; ++c) {
            // Upload previous and current color channels
            cudaMemcpyAsync(d_prev_color[stream_idx][c], prev_channels[c].data,
                input_width * input_height * sizeof(uint8_t),
                cudaMemcpyHostToDevice, streams[stream_idx]);

            cudaMemcpyAsync(d_curr_color[stream_idx][c], curr_channels[c].data,
                input_width * input_height * sizeof(uint8_t),
                cudaMemcpyHostToDevice, streams[stream_idx]);

            // Generate intermediate frame using warping for each color channel
            warp_frame << <flow_grid, flow_block, 0, streams[stream_idx] >> > (
                d_prev_color[stream_idx][c],
                d_vx[stream_idx],
                d_vy[stream_idx],
                d_fake_color[stream_idx][c],
                input_width, input_height
                );

            // Convert fake frame to float for upscaling
            convert_uint8_to_float << <flow_grid, flow_block, 0, streams[stream_idx] >> > (
                d_fake_color[stream_idx][c],
                d_fake_float[stream_idx][c],
                input_width, input_height
                );
        }

        // End timing frame warping
        cudaEventRecord(warp_stop, streams[stream_idx]);

        // Begin timing upscaling
        cudaEventRecord(upscale_start, streams[stream_idx]);

        // Upscaling for each channel
        for (int c = 0; c < num_channels; ++c) {
            dim3 upscale_block(16, 16);
            dim3 upscale_grid((output_width + upscale_block.x - 1) / upscale_block.x,
                (output_height + upscale_block.y - 1) / upscale_block.y);

            // Upscale the fake frame
          //  merged_kernel_with_loops << <upscale_grid, upscale_block, 0, streams[stream_idx] >> > (
            bilinear_interpolation << <upscale_grid, upscale_block, 0, streams[stream_idx] >> > (
                d_fake_float[stream_idx][c], d_fake_upscaled[stream_idx][c],
                input_width, input_height, output_width, output_height, scaling_factor, BLOCK_SIZE
                );

            // Also upscale the original frame
            cv::Mat float_frame(input_height, input_width, CV_32F, h_input[stream_idx][c]);
            curr_channels[c].convertTo(float_frame, CV_32F);

            // H2D Transfer
            cudaMemcpyAsync(d_input[stream_idx][c], h_input[stream_idx][c],
                input_width * input_height * sizeof(float),
                cudaMemcpyHostToDevice, streams[stream_idx]);

            // Kernel launch for upscaling original
          //  merged_kernel_with_loops << <upscale_grid, upscale_block, 0, streams[stream_idx] >> > (
            bilinear_interpolation << <upscale_grid, upscale_block, 0, streams[stream_idx] >> > (
                d_input[stream_idx][c], d_output[stream_idx][c],
                input_width, input_height, output_width, output_height, scaling_factor, BLOCK_SIZE
                );
        }

        // End timing upscaling
        cudaEventRecord(upscale_stop, streams[stream_idx]);

        // Convert and merge channels for displaying the original upscaled frame
        dim3 merge_block(16, 16);
        dim3 merge_grid((output_width + merge_block.x - 1) / merge_block.x,
            (output_height + merge_block.y - 1) / merge_block.y);
        merge_and_convert_channels << <merge_grid, merge_block, 0, streams[stream_idx] >> > (
            d_output[stream_idx][0], d_output[stream_idx][1], d_output[stream_idx][2],
            d_display_buffer[stream_idx], output_width, output_height
            );

        // Also merge and convert the upscaled fake frame channels
        unsigned char* d_fake_display_buffer;
        cudaMalloc(&d_fake_display_buffer, output_width * output_height * 3 * sizeof(unsigned char));

        merge_and_convert_channels << <merge_grid, merge_block, 0, streams[stream_idx] >> > (
            d_fake_upscaled[stream_idx][0], d_fake_upscaled[stream_idx][1], d_fake_upscaled[stream_idx][2],
            d_fake_display_buffer, output_width, output_height
            );

        // End timing for the whole frame
        cudaEventRecord(stop_event, streams[stream_idx]);

        // Record event
        cudaEventRecord(events[stream_idx], streams[stream_idx]);

        // Display frames after processing
        for (int i = 0; i < active_streams; ++i) {
            if (cudaEventQuery(events[i]) == cudaSuccess) {
                // Get frame processing time
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start_event, stop_event);
                double seconds = milliseconds / 1000.0;

                // Gather timing for individual operations
                float flow_ms = 0, warp_ms = 0, upscale_ms = 0;
                cudaEventElapsedTime(&flow_ms, flow_start, flow_stop);
                cudaEventElapsedTime(&warp_ms, warp_start, warp_stop);
                cudaEventElapsedTime(&upscale_ms, upscale_start, upscale_stop);

                // Update performance metrics
                total_frames++;
                total_processing_time += seconds;
                min_frame_time = std::min(min_frame_time, static_cast<double>(seconds));
                max_frame_time = std::max(max_frame_time, static_cast<double>(seconds));

                // Update stage timing totals
                total_flow_time += flow_ms / 1000.0;
                total_warp_time += warp_ms / 1000.0;
                total_upscale_time += upscale_ms / 1000.0;

                // Show every 4th frame
                if (++display_count % 4 == 0) {
                    // Download upscaled original frame
                    cudaMemcpy(upscaled_original.data, d_display_buffer[i],
                        output_width * output_height * 3 * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost);

                    // Download upscaled fake frame
                    cudaMemcpy(upscaled_fake_frame.data, d_fake_display_buffer,
                        output_width * output_height * 3 * sizeof(unsigned char),
                        cudaMemcpyDeviceToHost);

                    // Show both upscaled original and intermediate frame
                    cv::imshow("Upscaled Original", upscaled_original);
                    cv::imshow("Upscaled Interpolated Frame", upscaled_fake_frame);

                    // Download flow vectors for visualization
                    cudaMemcpy(h_vx, d_vx[i], input_width * input_height * sizeof(float), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_vy, d_vy[i], input_width * input_height * sizeof(float), cudaMemcpyDeviceToHost);

                    // Create a flow visualization image
                    cv::Mat flow_vis(input_height, input_width, CV_8UC3, cv::Scalar(0, 0, 0));

                    // Visualize optical flow
                    for (int y = 0; y < input_height; y++) {
                        for (int x = 0; x < input_width; x++) {
                            float fx = h_vx[y * input_width + x];
                            float fy = h_vy[y * input_width + x];

                            // Calculate magnitude and angle
                            float magnitude = sqrt(fx * fx + fy * fy);
                            float angle = atan2(fy, fx) * 180.0 / CV_PI;
                            if (angle < 0) angle += 360.0;

                            // Map to HSV (hue based on angle, saturation based on magnitude)
                            uchar h = static_cast<uchar>(angle / 2.0); // H: 0-180 for OpenCV
                            uchar s = std::min(255.0f, magnitude * 10.0f); // S: scale magnitude
                            uchar v = 255; // V: full brightness

                            flow_vis.at<cv::Vec3b>(y, x) = cv::Vec3b(h, s, v);
                        }
                    }

                    // Convert from HSV to BGR for display
                    cv::cvtColor(flow_vis, flow_vis, cv::COLOR_HSV2BGR);
                    cv::imshow("Optical Flow", flow_vis);

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
            std::cout << "Current FPS: " << frame_count << std::endl;
            frame_count = 0;
            start_time = current_time;
        }

        // Free the temporary buffer
        cudaFree(d_fake_display_buffer);

        // Update previous frame for next iteration
        prev_gray = frame_gray.clone();
        prev_frame = curr_frame.clone();
        stream_idx = (stream_idx + 1) % active_streams;
    }

    // Calculate total elapsed time
    auto total_end_time = std::chrono::high_resolution_clock::now();
    double total_elapsed_time = std::chrono::duration<double>(total_end_time - total_start_time).count();

    // Print detailed performance metrics
    if (total_frames > 0) {
        // Use total elapsed time for more accurate average FPS
        double avg_fps = total_frames / total_elapsed_time;
        double avg_ms_per_frame = (total_elapsed_time * 1000.0) / total_frames;

        std::cout << "\n===== Performance Metrics =====" << std::endl;
        std::cout << "Total frames processed: " << total_frames << std::endl;
        std::cout << "Total elapsed time: " << total_elapsed_time << " seconds" << std::endl;
        std::cout << "Average FPS: " << avg_fps << std::endl;
        std::cout << "Average ms per frame: " << avg_ms_per_frame << " ms" << std::endl;
        std::cout << "Min frame time: " << min_frame_time * 1000.0 << " ms" << std::endl;
        std::cout << "Max frame time: " << max_frame_time * 1000.0 << " ms" << std::endl;

        // Print stage-specific metrics
        double avg_flow_time = total_flow_time / total_frames * 1000.0;
        double avg_warp_time = total_warp_time / total_frames * 1000.0;
        double avg_upscale_time = total_upscale_time / total_frames * 1000.0;

        std::cout << "\n===== Processing Stage Metrics =====" << std::endl;
        std::cout << "Optical Flow: " << avg_flow_time << " ms per frame ("
            << (avg_flow_time / avg_ms_per_frame) * 100.0 << "% of total)" << std::endl;
        std::cout << "Frame Warping: " << avg_warp_time << " ms per frame ("
            << (avg_warp_time / avg_ms_per_frame) * 100.0 << "% of total)" << std::endl;
        std::cout << "Upscaling: " << avg_upscale_time << " ms per frame ("
            << (avg_upscale_time / avg_ms_per_frame) * 100.0 << "% of total)" << std::endl;

        // Video resolution info
        std::cout << "\n===== Video Information =====" << std::endl;
        std::cout << "Input resolution: " << input_width << "x" << input_height << std::endl;
        std::cout << "Output resolution: " << output_width << "x" << output_height << std::endl;
        std::cout << "Scaling factor: " << scaling_factor << "x" << std::endl;
        std::cout << "Original video FPS: " << fps << std::endl;
    }

    // Cleanup
    cap.release();
    cudaFreeHost(h_vx);
    cudaFreeHost(h_vy);

    // Destroy timing events
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaEventDestroy(flow_start);
    cudaEventDestroy(flow_stop);
    cudaEventDestroy(warp_start);
    cudaEventDestroy(warp_stop);
    cudaEventDestroy(upscale_start);
    cudaEventDestroy(upscale_stop);

    for (int i = 0; i < NUM_STREAMS; ++i) {
        for (int c = 0; c < num_channels; ++c) {
            cudaFreeHost(h_input[i][c]);
            cudaFree(d_input[i][c]);
            cudaFree(d_output[i][c]);

            // Free color channel warping resources
            cudaFree(d_prev_color[i][c]);
            cudaFree(d_curr_color[i][c]);
            cudaFree(d_fake_color[i][c]);

            // Free upscaled fake frame resources
            cudaFree(d_fake_float[i][c]);
            cudaFree(d_fake_upscaled[i][c]);
        }

        // Free optical flow resources
        cudaFree(d_frame1[i]);
        cudaFree(d_frame2[i]);
        cudaFree(d_vx[i]);
        cudaFree(d_vy[i]);

        cudaFree(d_display_buffer[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    cv::destroyAllWindows();
    return 0;
}