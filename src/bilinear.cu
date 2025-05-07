// bilinear.cu
#include "bilinear.h" // Include the header declaration

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp> // For cv::cvtColor and cv::convertTo
#include <stdio.h>            // For error messages
#include <math.h>             // For sqrtf, atan2f
// Define PI if not available in math_constants.h implicitly
#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

// --- CUDA Error Handling Utility (Same as before) ---
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code); // Or handle error differently
    }
}

// --- Device Helper Function (Same as before) ---
__device__ inline float getValueDevice(const float* I, int r, int c, int W_orig_rows, int H_orig_cols) {
    // W_orig_rows: total number of rows in the image
    // H_orig_cols: total number of columns in the image (stride for row-major access)
    int clamped_r = max(0, min(W_orig_rows - 1, r));
    int clamped_c = max(0, min(H_orig_cols - 1, c));
    return I[clamped_r * H_orig_cols + clamped_c];
}

// --- Updated CUDA Kernel ---
// Now also calculates and outputs S0, S1, S2
__global__ void bilinearInterpolationPolarizationKernel(
    const float* I,
    float* I0, float* I45, float* I90, float* I135,
    float* S0_out, float* S1_out, float* S2_out, // Add S0, S1, S2 output pointers
    float* DoLP, float* AoLP,
    int W_orig, int H_orig) // W_orig = rows, H_orig = cols
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Image column index (x-coordinate)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Image row index (y-coordinate)

    if (row < W_orig && col < H_orig) { // Check bounds: row < num_rows, col < num_cols
        int i_padded = row + 1; // Parity check based on 1-based indexing as in original logic
        int j_padded = col + 1;
        int x_parity = i_padded % 2;
        int y_parity = j_padded % 2;

        // Get neighbor values using W_orig (total rows) and H_orig (total cols)
        float center = getValueDevice(I, row,     col,     W_orig, H_orig);
        float N  = getValueDevice(I, row - 1, col,     W_orig, H_orig);
        float S  = getValueDevice(I, row + 1, col,     W_orig, H_orig);
        float W_ = getValueDevice(I, row,     col - 1, W_orig, H_orig);
        float E  = getValueDevice(I, row,     col + 1, W_orig, H_orig);
        float NW = getValueDevice(I, row - 1, col - 1, W_orig, H_orig);
        float SE = getValueDevice(I, row + 1, col + 1, W_orig, H_orig);
        float NE = getValueDevice(I, row - 1, col + 1, W_orig, H_orig);
        float SW = getValueDevice(I, row + 1, col - 1, W_orig, H_orig);

        float avg_diag = (NW + SE + NE + SW) / 4.0f;
        float avg_horiz = (E + W_) / 2.0f;
        float avg_vert = (S + N) / 2.0f;

        int idx = row * H_orig + col; // Linear index for output arrays (row-major, H_orig is width/cols)

        // Temporary variables to store intensity results for this thread
        float i0_val, i45_val, i90_val, i135_val;

        // --- Calculate I0, I45, I90, I135 (Same logic as before) ---
        if (x_parity == 1) { // Odd row (1-based)
            if (y_parity == 1) { // Odd col (1-based) -> I0 is center
                i90_val = avg_diag; i0_val = center; i135_val = avg_horiz; i45_val = avg_vert;
            } else {             // Odd row, Even col (1-based) -> I135 is center
                i45_val = avg_diag; i135_val = center; i0_val = avg_horiz; i90_val = avg_vert;
            }
        } else { // Even row (1-based)
            if (y_parity == 1) { // Even row, Odd col (1-based) -> I45 is center
                i135_val = avg_diag; i45_val = center; i90_val = avg_horiz; i0_val = avg_vert;
            } else {             // Even row, Even col (1-based) -> I90 is center
                i0_val = avg_diag; i90_val = center; i45_val = avg_horiz; i135_val = avg_vert;
            }
        }

        // --- Store Intensity Results ---
        I0[idx] = i0_val;
        I45[idx] = i45_val;
        I90[idx] = i90_val;
        I135[idx] = i135_val;

        // --- Calculate Stokes Parameters ---
        float s0 = i0_val + i90_val;
        float s1 = i0_val - i90_val;
        float s2 = i45_val - i135_val;

        // --- Store Stokes Parameters ---
        S0_out[idx] = s0;
        S1_out[idx] = s1;
        S2_out[idx] = s2;

        // --- Calculate Polarization Parameters (DoLP and AoLP) ---
        // Calculate DoLP (add epsilon for stability, clamp to [0, 1])
        float dolp_val = 0.0f;
        float s1s1_s2s2 = s1 * s1 + s2 * s2;
        if (s0 > 1e-6f) { // Avoid division by zero or near-zero
             dolp_val = sqrtf(s1s1_s2s2) / s0;
        }
         // Clamp DoLP to [0, 1] range as noise might slightly exceed 1
        DoLP[idx] = fminf(fmaxf(dolp_val, 0.0f), 1.0f);

        // Calculate AoLP (range [-PI/2, PI/2])
        AoLP[idx] = 0.5f * atan2f(s2, s1);
    }
}


// --- Updated Host Wrapper Function Implementation ---
// Use extern "C" to match the header declaration
extern "C" bool Bilinear_Interpolation_And_Polarization_CUDA(const cv::Mat& inputImage,
                                                            cv::Mat& outputI0,
                                                            cv::Mat& outputI45,
                                                            cv::Mat& outputI90,
                                                            cv::Mat& outputI135,
                                                            cv::Mat& outputS0,   // Added
                                                            cv::Mat& outputS1,   // Added
                                                            cv::Mat& outputS2,   // Added
                                                            cv::Mat& outputDoLP,
                                                            cv::Mat& outputAoLP)
{
    // --- Input Validation (Same as before) ---
    if (inputImage.empty()) {
        fprintf(stderr, "Error: Input image is empty.\n");
        return false;
    }
    if (inputImage.channels() != 1) {
         fprintf(stderr, "Error: Input image must be single-channel (grayscale).\n");
        return false;
    }

    // --- Prepare Input Data (Same as before) ---
    cv::Mat inputFloat;
    if (inputImage.type() != CV_32F) {
        inputImage.convertTo(inputFloat, CV_32F, 1.0 / 255.0); // Scale to [0,1]
    } else {
        inputFloat = inputImage.clone(); // Clone to ensure continuity or if it's a ROI
    }
    if (!inputFloat.isContinuous()) {
        inputFloat = inputFloat.clone(); // Ensure it's continuous
    }

    int W = inputFloat.rows; // Number of rows (height)
    int H = inputFloat.cols; // Number of columns (width)
    size_t image_size_bytes = (size_t)W * H * sizeof(float);

    // --- Allocate Output cv::Mat ---
    outputI0.create(W, H, CV_32F);
    outputI45.create(W, H, CV_32F);
    outputI90.create(W, H, CV_32F);
    outputI135.create(W, H, CV_32F);
    outputS0.create(W, H, CV_32F);   // Allocate S0 Mat
    outputS1.create(W, H, CV_32F);   // Allocate S1 Mat
    outputS2.create(W, H, CV_32F);   // Allocate S2 Mat
    outputDoLP.create(W, H, CV_32F);
    outputAoLP.create(W, H, CV_32F);

    // --- GPU Memory Management ---
    float *d_I = nullptr, *d_I0 = nullptr, *d_I45 = nullptr, *d_I90 = nullptr, *d_I135 = nullptr;
    float *d_S0 = nullptr, *d_S1 = nullptr, *d_S2 = nullptr; // Added d_S0, d_S1, d_S2
    float *d_DoLP = nullptr, *d_AoLP = nullptr;
    cudaError_t err;

    // Lambda for cleanup
    auto cleanup_gpu_memory = [&]() {
        cudaFree(d_I); cudaFree(d_I0); cudaFree(d_I45); cudaFree(d_I90); cudaFree(d_I135);
        cudaFree(d_S0); cudaFree(d_S1); cudaFree(d_S2);
        cudaFree(d_DoLP); cudaFree(d_AoLP);
    };

    // Allocate all device memory
    // If any gpuErrchk aborts, program exits. If it doesn't abort, subsequent checks are needed.
    // Assuming gpuErrchk aborts on error as per its definition.
    err = cudaMalloc((void**)&d_I, image_size_bytes);    gpuErrchk(err);
    err = cudaMalloc((void**)&d_I0, image_size_bytes);   gpuErrchk(err);
    err = cudaMalloc((void**)&d_I45, image_size_bytes);  gpuErrchk(err);
    err = cudaMalloc((void**)&d_I90, image_size_bytes);  gpuErrchk(err);
    err = cudaMalloc((void**)&d_I135, image_size_bytes); gpuErrchk(err);
    err = cudaMalloc((void**)&d_S0, image_size_bytes);   gpuErrchk(err); // Allocate S0
    err = cudaMalloc((void**)&d_S1, image_size_bytes);   gpuErrchk(err); // Allocate S1
    err = cudaMalloc((void**)&d_S2, image_size_bytes);   gpuErrchk(err); // Allocate S2
    err = cudaMalloc((void**)&d_DoLP, image_size_bytes); gpuErrchk(err);
    err = cudaMalloc((void**)&d_AoLP, image_size_bytes); gpuErrchk(err);


    // --- Data Transfer: Host -> Device ---
    // Note: This cudaMemcpy is NOT wrapped in gpuErrchk in the original code.
    err = cudaMemcpy(d_I, inputFloat.ptr<float>(), image_size_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error memcpy H->D (d_I): %s\n", cudaGetErrorString(err));
        cleanup_gpu_memory();
        return false;
    }

    // --- Kernel Launch ---
    // Grid dimensions: H is cols (width, x-dim for kernel), W is rows (height, y-dim for kernel)
    dim3 blockSize(16, 16);
    dim3 gridSize((H + blockSize.x - 1) / blockSize.x, (W + blockSize.y - 1) / blockSize.y);

    // Call the updated kernel
    bilinearInterpolationPolarizationKernel<<<gridSize, blockSize>>>(
        d_I, d_I0, d_I45, d_I90, d_I135,
        d_S0, d_S1, d_S2, // Pass S0/S1/S2 device pointers
        d_DoLP, d_AoLP,
        W, H); // W is rows, H is cols

    gpuErrchk(cudaPeekAtLastError()); // Check for errors during kernel launch
    gpuErrchk(cudaDeviceSynchronize()); // Wait for kernel to complete and check for runtime errors

    // --- Data Transfer: Device -> Host ---
    // These are wrapped in gpuErrchk, so they will abort on failure.
    err = cudaMemcpy(outputI0.ptr<float>(), d_I0, image_size_bytes, cudaMemcpyDeviceToHost);       gpuErrchk(err);
    err = cudaMemcpy(outputI45.ptr<float>(), d_I45, image_size_bytes, cudaMemcpyDeviceToHost);     gpuErrchk(err);
    err = cudaMemcpy(outputI90.ptr<float>(), d_I90, image_size_bytes, cudaMemcpyDeviceToHost);     gpuErrchk(err);
    err = cudaMemcpy(outputI135.ptr<float>(), d_I135, image_size_bytes, cudaMemcpyDeviceToHost);   gpuErrchk(err);
    err = cudaMemcpy(outputS0.ptr<float>(), d_S0, image_size_bytes, cudaMemcpyDeviceToHost);       gpuErrchk(err); // Copy S0
    err = cudaMemcpy(outputS1.ptr<float>(), d_S1, image_size_bytes, cudaMemcpyDeviceToHost);       gpuErrchk(err); // Copy S1
    err = cudaMemcpy(outputS2.ptr<float>(), d_S2, image_size_bytes, cudaMemcpyDeviceToHost);       gpuErrchk(err); // Copy S2
    err = cudaMemcpy(outputDoLP.ptr<float>(), d_DoLP, image_size_bytes, cudaMemcpyDeviceToHost);   gpuErrchk(err);
    err = cudaMemcpy(outputAoLP.ptr<float>(), d_AoLP, image_size_bytes, cudaMemcpyDeviceToHost);   gpuErrchk(err);

    // --- Cleanup GPU Memory ---
    // Using the lambda here ensures all listed pointers are attempted to be freed.
    // cudaFree is safe with nullptr.
    cleanup_gpu_memory();

    return true; // Success
}