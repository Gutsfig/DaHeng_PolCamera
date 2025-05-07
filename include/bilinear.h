// bilinear.h
#ifndef BILINEAR_CUDA_H
#define BILINEAR_CUDA_H

#include <opencv2/core.hpp> // Include OpenCV core for cv::Mat

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Performs directional bilinear interpolation and calculates polarization parameters
 *        (DoLP, AoLP, S0, S1, S2) on a single-channel image using CUDA.
 *
 * Calculates four interpolated images (I0, I45, I90, I135), the Stokes parameters
 * (S0, S1, S2), the Degree of Linear Polarization (DoLP), and the Angle of
 * Linear Polarization (AoLP) based on neighbor pixels and pixel grid parity.
 *
 * @param inputImage The input single-channel image (e.g., grayscale CV_8U or CV_32F).
 *                   It will be converted to CV_32F internally if needed.
 * @param outputI0   Output cv::Mat (will be allocated as CV_32F) for 0-degree interpolation.
 * @param outputI45   Output cv::Mat (will be allocated as CV_32F) for 45-degree interpolation.
 * @param outputI90   Output cv::Mat (will be allocated as CV_32F) for 90-degree interpolation.
 * @param outputI135  Output cv::Mat (will be allocated as CV_32F) for 135-degree interpolation.
 * @param outputS0    Output cv::Mat (will be allocated as CV_32F) for Stokes parameter S0.
 * @param outputS1    Output cv::Mat (will be allocated as CV_32F) for Stokes parameter S1.
 * @param outputS2    Output cv::Mat (will be allocated as CV_32F) for Stokes parameter S2.
 * @param outputDoLP  Output cv::Mat (will be allocated as CV_32F) for Degree of Linear Polarization [0, 1].
 * @param outputAoLP  Output cv::Mat (will be allocated as CV_32F) for Angle of Linear Polarization [-PI/2, PI/2] radians.
 * @return true if successful, false otherwise (e.g., input error, CUDA error).
 */
bool Bilinear_Interpolation_And_Polarization_CUDA(const cv::Mat& inputImage,
                                                  cv::Mat& outputI0,
                                                  cv::Mat& outputI45,
                                                  cv::Mat& outputI90,
                                                  cv::Mat& outputI135,
                                                  cv::Mat& outputS0,   // Added S0
                                                  cv::Mat& outputS1,   // Added S1
                                                  cv::Mat& outputS2,   // Added S2
                                                  cv::Mat& outputDoLP,
                                                  cv::Mat& outputAoLP);


#ifdef __cplusplus
} // extern "C"
#endif

#endif // BILINEAR_CUDA_H