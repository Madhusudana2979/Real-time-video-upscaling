CUDA-Based Video Upscaling with Optical Flow
This project implements a CUDA-based video processing pipeline for upscaling video frames, with an optional optical flow-based frame interpolation for smoother playback. Two implementations are provided: one with optical flow (with_OF.cu) for generating intermediate frames, and one without optical flow (without_OF.cu) for direct upscaling. Both versions leverage CUDA for GPU acceleration and OpenCV for video handling and visualization.
Overview
The project processes input videos by upscaling their resolution using bilinear interpolation on the GPU. The with_OF.cu version additionally computes optical flow between consecutive frames to generate intermediate frames, effectively increasing the frame rate while maintaining smooth motion. Both implementations use CUDA streams for asynchronous processing to optimize performance.
Device: GeForce RTX 3050
Key Features

Upscaling: Scales video frames by a specified factor (default: 4x) using bilinear interpolation.
Optical Flow (with_OF version): Computes optical flow using the Lucas-Kanade method and generates intermediate frames via frame warping.
Multi-Stream Processing: Utilizes multiple CUDA streams (default: 4) for overlapping data transfers and kernel executions.
Performance Metrics: The with_OF version includes detailed timing metrics for optical flow computation, frame warping, and upscaling.
Real-Time Visualization: Displays upscaled frames (and optical flow visualization in with_OF) using OpenCV.

Performance

Without Optical Flow (without_OF.cu): Achieves high frame rates (450+ FPS) for direct upscaling(linux), suitable for real-time applications where frame interpolation is not needed.
With Optical Flow (with_OF.cu): Slower due to additional optical flow computation and frame warping but provides smoother playback by interpolating frames capable of achieveing around ~250FPS. Includes detailed performance metrics for each processing stage.

Prerequisites
To compile and run the code, ensure you have the following installed:

CUDA Toolkit: Version compatible with your GPU (e.g., CUDA 11.x or later).
OpenCV: Version with CUDA support (e.g., OpenCV 4.x compiled with CUDA).
C++ Compiler: Compatible with CUDA (e.g., g++ or Visual Studio).
NVIDIA GPU: With sufficient memory to handle video frame buffers.
Operating System: Tested on Linux (e.g., Ubuntu).

Dependencies

CUDA runtime libraries (cuda_runtime.h, cuda_runtime_api.h, etc.).
OpenCV libraries (opencv2/opencv.hpp).
Standard C++ libraries (iostream, vector, chrono, etc.).

Installation

Install CUDA Toolkit:

Download and install the CUDA Toolkit from the NVIDIA Developer website.
Ensure your GPU drivers are up to date.

Install OpenCV with CUDA Support:

Follow instructions to build OpenCV with CUDA support. For example, on Ubuntu:sudo apt-get install libopencv-dev

For CUDA support, you may need to build OpenCV from source with the -D WITH_CUDA=ON flag in CMake.


Set Up Project:

Copy the provided source files (with_OF.txt or without_OF.txt) and rename them to .cu (e.g., with_OF.cu).
Ensure your development environment includes paths to CUDA and OpenCV headers and libraries.


Compile the Code:Use nvcc to compile the CUDA code. Example command for Linux:
nvcc -o video_upscale with_OF.cu `pkg-config --cflags --libs opencv4` -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart

For Windows, use a compatible compiler (e.g., Visual Studio) with CUDA and OpenCV configured. The driver support in windows is not as efficient as it is in Ubuntu, so there might be performance degradation wrt FPS achieved.

Usage

Set Input and Output Video Paths:

Input Video Path: Modify the input_video_path variable in the main() function to point to your input video file. The default path is:std::string input_video_path = "/home/madhu/tensorrt_Test/output_video.webm";

Replace it with the path to your video file (e.g., /path/to/your/video.mp4 on Linux or C:\\path\\to\\your\\video.mp4 on Windows).
Output Video Path: The current implementation does not save the output video to a file but displays it in real-time using OpenCV. To save the output, you can modify the code to use cv::VideoWriter. Example:cv::VideoWriter writer("output_video.mp4", cv::VideoWriter::fourcc('M','J','P','G'), fps, cv::Size(output_width, output_height));
writer.write(output_frame);

Add this in the display loop where output_frame is shown.


Run the Program:

Execute the compiled binary:./video_upscale


The program will process the video, display upscaled frames, and (for with_OF) show optical flow visualization and interpolated frames.
Press q or Q to exit the program during playback.


Configuration Parameters:

Scaling Factor: Adjust scaling_factor (default: 4.0f) to change the upscaling ratio.
Number of Streams: Modify NUM_STREAMS to adjust concurrency.
Block Size: The BLOCK_SIZE (default: 16) defines the CUDA thread block size.
Window Size (with_OF): The WINDOW_SIZE (default: 3) and HALF_WINDOW control the optical flow computation window.


Code Structure
Common Components
Both versions share the following components:

CUDA Kernels:
bilinear_interpolation: Performs bilinear interpolation for upscaling frames.
convert_uint8_to_float: Converts uint8_t frame data to float for processing.
merge_and_convert_channels: Merges RGB channels and converts them to uint8_t for display.


Utility Functions:
saveImageData: Saves grayscale images as PGM files (for debugging).
floatToUint8 and uint8ToFloat: Convert between float and uint8_t data types.
checkCudaErrors: Macro for CUDA error checking.


Main Pipeline:
Reads video frames using OpenCV.
Processes frames in multiple CUDA streams for parallel execution.
Displays results using OpenCV windows.



With Optical Flow (with_OF.cu)

Additional Kernels:
OFA_kernel: Computes optical flow using the Lucas-Kanade method with Sobel gradients and structure tensor.
warp_frame: Generates intermediate frames by warping based on optical flow vectors.


Additional Features:
Computes optical flow between consecutive frames.
Generates intermediate frames for smoother playback.
Visualizes optical flow as an HSV image (hue for direction, saturation for magnitude).
Tracks detailed performance metrics (optical flow, warping, upscaling times).


Memory Management:
Allocates additional memory for grayscale frames, flow fields (d_vx, d_vy), and intermediate frames.
Uses pinned memory for host-device transfers.



Without Optical Flow (without_OF.cu)

Simplified Pipeline:
Directly upscales input frames without computing optical flow or interpolating frames.
Faster processing due to fewer computations.


Memory Usage:
Requires less GPU memory compared to with_OF since it skips optical flow and warping.



Performance Considerations

Without Optical Flow:

Optimized for speed, achieving 180+ FPS on modern NVIDIA GPUs.
Suitable for applications where high frame rates are critical, and frame interpolation is not needed.
Memory usage is lower due to fewer buffers.


With Optical Flow:

Slower due to additional computations (optical flow and frame warping).
Provides smoother playback by interpolating frames, ideal for low-frame-rate videos.
Includes detailed timing metrics to analyze performance bottlenecks:
Optical flow computation time.
Frame warping time.
Upscaling time.


Higher memory usage due to additional buffers for flow fields and intermediate frames.



Limitations

Optical Flow Accuracy: The Lucas-Kanade method in OFA_kernel assumes small motions and may not handle large displacements or complex scenes well.
Memory Usage: The with_OF version requires significant GPU memory for flow fields and intermediate frames, which may be a bottleneck on low-memory GPUs.
Output Saving: The current implementation displays frames but does not save the output video. Users must add cv::VideoWriter for saving.
Error Handling: Limited error handling for video format compatibility or GPU memory allocation failures.

Future Improvements

Enhanced Optical Flow: Implement more robust optical flow algorithms (e.g., Farneback or deep learning-based methods).
Output Video Saving: Add support for saving upscaled and interpolated videos using cv::VideoWriter.
Dynamic Stream Adjustment: Automatically adjust the number of streams based on GPU capabilities.
Optimizations:
Optimize memory access patterns in CUDA kernels.
Use shared memory for bilinear interpolation.
Implement half-precision (cuda_fp16.h) for faster computations.


Error Handling: Add robust checks for video format compatibility and memory allocation.

Troubleshooting

Video File Not Found:
Ensure the input_video_path is correct and the video file is accessible.
Supported formats depend on OpenCV's backend (e.g., FFmpeg for .mp4, .webm).


CUDA Errors:
Check GPU compatibility and CUDA Toolkit installation.
Ensure sufficient GPU memory is available.


OpenCV Display Issues:
Verify OpenCV is built with GUI support (e.g., Qt or GTK).
Check that the display window is not blocked by other applications.


Low FPS:
Reduce scaling_factor or NUM_STREAMS to lower GPU load.
Use a more powerful GPU for the with_OF version.



Example Configuration
To process a video at /path/to/video.mp4 with a scaling factor of 2x, modify the main() function:
std::string input_video_path = "/path/to/video.mp4";
float scaling_factor = 2.0f;

To save the output video, add after the display loop:
cv::VideoWriter writer("output_video.mp4", cv::VideoWriter::fourcc('M','J','P','G'), fps, cv::Size(output_width, output_height));
writer.write(output_frame); // For without_OF
// For with_OF, write both upscaled_original and upscaled_fake_frame as needed

License
This project is provided as-is for educational purposes. Users are responsible for ensuring compliance with licenses for dependencies (CUDA, OpenCV).
