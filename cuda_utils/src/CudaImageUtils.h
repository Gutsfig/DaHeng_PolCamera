#ifndef CUDA_IMAGE_UTILS_H
#define CUDA_IMAGE_UTILS_H

#include <opencv2/core.hpp>

// --- 1. 定义统一的API导出/导入宏 ---
#if defined(_MSC_VER)
    #ifdef CUDA_UTILS_DLL_EXPORTS // 如果我们正在构建这个DLL
        #define CUDA_UTILS_API __declspec(dllexport)
    #else // 如果我们正在使用这个DLL
        #define CUDA_UTILS_API __declspec(dllimport)
    #endif
#else
    #define CUDA_UTILS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// --- 2. 声明所有要导出的函数，并在前面加上宏 ---

/**
 * @brief (来自bilinear.h) Performs bilinear interpolation and polarization calculations.
 * ... (完整的Doxygen注释) ...
 */
CUDA_UTILS_API bool Bilinear_Interpolation_And_Polarization_CUDA(
    const cv::Mat& inputImage,
    cv::Mat& outputI0, cv::Mat& outputI45, cv::Mat& outputI90, cv::Mat& outputI135,
    cv::Mat& outputS0, cv::Mat& outputS1, cv::Mat& outputS2,
    cv::Mat& outputDoLP, cv::Mat& outputAoLP
);

/**
 * @brief (来自normalize_gpu.h) Normalizes a CV_32FC1 image to CV_8UC1 on the GPU.
 * ... (完整的Doxygen注释) ...
 */
CUDA_UTILS_API bool normalize_minmax_cuda_32f_to_8u(
    const cv::Mat& src_float_mat,
    cv::Mat& dst_uchar_mat
);


#ifdef __cplusplus
} // extern "C"
#endif

#endif // CUDA_IMAGE_UTILS_H