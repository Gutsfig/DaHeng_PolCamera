#ifndef POLARIZATION_KERNELS_H
#define POLARIZATION_KERNELS_H

#include "opencv2/opencv.hpp"
#include <cuda_runtime.h> // For cudaStream_t

// --- 添加这部分代码 ---
#if defined(_MSC_VER) // 仅在 MSVC 编译器下
    #ifdef POLAR_DLL_EXPORTS // 如果我们正在构建这个DLL
        #define POLAR_API __declspec(dllexport)
    #else // 如果我们正在使用这个DLL
        #define POLAR_API __declspec(dllimport)
    #endif
#else // 其他编译器（如GCC）
    #define POLAR_API
#endif
// --- 结束添加 ---

// CUDA_CHECK macro (保持不变)
#define CUDA_CHECK(call)                                                      \
do {                                                                          \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));                 \
        /* Consider exiting or throwing an exception here for critical errors */ \
    }                                                                         \
} while(0)

class POLAR_API PolarizationProcessor {
public:
    /**
     * @brief Constructor for PolarizationProcessor.
     *        Allocates necessary GPU and pinned host memory.
     * @param raw_image_width Width of the input raw polarization image.
     * @param raw_image_height Height of the input raw polarization image.
     * @param use_pinned_memory If true, uses pinned host memory for H2D and D2H transfers.
     *                          This is generally recommended for asynchronous operations.
     */
    PolarizationProcessor(int raw_image_width, int raw_image_height, bool use_pinned_memory = true);
    ~PolarizationProcessor();

    // Disable copy constructor and assignment operator
    PolarizationProcessor(const PolarizationProcessor&) = delete;
    PolarizationProcessor& operator=(const PolarizationProcessor&) = delete;

    /**
     * @brief Processes a single raw polarization frame.
     *        Uses pre-allocated GPU memory and optionally pinned host memory.
     *
     * @param raw_cv_image Input cv::Mat (CV_8UC1) representing the raw mosaic image.
     * @param i0_cv_image Output cv::Mat (CV_32FC1) for the 0-degree intensity component.
     * @param i45_cv_image Output cv::Mat (CV_32FC1) for the 45-degree intensity component.
     * @param i90_cv_image Output cv::Mat (CV_32FC1) for the 90-degree intensity component.
     * @param i135_cv_image Output cv::Mat (CV_32FC1) for the 135-degree intensity component.
     * @param s0_cv_image Output cv::Mat (CV_32FC1) for the S0 Stokes parameter.
     * @param s1_cv_image Output cv::Mat (CV_32FC1) for the S1 Stokes parameter.
     * @param s2_cv_image Output cv::Mat (CV_32FC1) for the S2 Stokes parameter.
     * @param dolp_cv_image Output cv::Mat (CV_32FC1) for the Degree of Linear Polarization.
     * @param aolp_cv_image Output cv::Mat (CV_32FC1) for the Angle of Linear Polarization.
     * @return True if processing was successful, false otherwise.
     */
    bool process_frame(
        const cv::Mat& raw_cv_image,
        cv::Mat& i0_cv_image,
        cv::Mat& i45_cv_image,
        cv::Mat& i90_cv_image,
        cv::Mat& i135_cv_image,
        cv::Mat& s0_cv_image,
        cv::Mat& s1_cv_image,
        cv::Mat& s2_cv_image,
        cv::Mat& dolp_cv_image,
        cv::Mat& aolp_cv_image
    );

private:
    bool initialized_ = false;
    bool use_pinned_memory_;

    int width_;
    int height_;
    int out_width_;
    int out_height_;
    size_t raw_size_bytes_;
    size_t out_float_size_bytes_;

    // CUDA Stream
    cudaStream_t stream_ = 0;

    // Pinned host memory buffers (optional)
    unsigned char* h_pinned_raw_ = nullptr;
    float* h_pinned_i0_ = nullptr; // Example for one output, can extend for all 9
    // For simplicity in this example, we'll only pin the input.
    // Output pinning requires more buffers or careful management if cv::Mat.data is used directly.

    // GPU device memory buffers
    unsigned char* d_raw_image_ = nullptr;
    float *d_i0_ = nullptr, *d_i45_ = nullptr, *d_i90_ = nullptr, *d_i135_ = nullptr;
    float *d_s0_ = nullptr, *d_s1_ = nullptr, *d_s2_ = nullptr;
    float *d_dolp_ = nullptr, *d_aolp_ = nullptr;

    // Kernel launch configuration
    dim3 threads_per_block_;
    dim3 num_blocks_;
};

#endif // POLARIZATION_KERNELS_H