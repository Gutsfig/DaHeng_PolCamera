cmake_minimum_required(VERSION 3.20)
project(DaHeng_PolCamera LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
#89-40 series, 86-30 series
set(CMAKE_CUDA_ARCHITECTURES 89;86)
# set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe")
#-------------------------------------------------------------------------------
#修改为自己对应的目录
#-------------------------------------------------------------------------------
# set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
set(OpenCV_DIR "D:/Tools/opencv4.9/opencv/build")
set(GALAXY_SDK_PATH "D:/Tools/camera/Galaxy/GalaxySDK/Development/C++ SDK")
set(POLAR_LIB_PATH "D:/project/camera/DaHeng_PolCamera2")
#查找依赖项
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS 
    core 
    imgproc 
    highgui 
    #opencv_cuda
    #cudaimgproc
)
#-------------------------------------------------------------------------------
# File organization
#-------------------------------------------------------------------------------
set(SOURCES  
    src/main.cpp
    src/ThreadSafeQuene.cpp
    # src/cuda_utils/bilinear.cu
    # src/cuda_utils/normalize_gpu.cu
    # src/cuda_utils/custom_polarization_cuda.cu
    # src/cuda_utils/polarization_kernels.cu
    # src/cuda_utils/hsv_fusion_cuda.cu
)
#-------------------------------------------------------------------------------
# Executable target
#-------------------------------------------------------------------------------
add_executable(${PROJECT_NAME} 
    ${SOURCES} 
)


# Windows-specific libraries
if(WIN32)
    set(WIN_LIBS winmm)
endif()

#-------------------------------------------------------------------------------
# Target configurations
#-------------------------------------------------------------------------------
target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include 
    # ${PROJECT_SOURCE_DIR}/cuda_utils/src

    # "D:\\Tools\\opencv4.9\\opencv\\build\\include\\opencv2"
    # "D:\\Tools\\camera\\Galaxy\\GalaxySDK\\Development\\C++ SDK\\inc"
    "D:/project/camera/DaHeng_PolCamera2/cuda_utils/src"
    "${GALAXY_SDK_PATH}/inc"
    "${OpenCV_DIR}/include/opencv2"   
    "${POLAR_LIB_PATH}/cuda_utils/src" 
    
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    CUDA::cublas
    CUDA::cufft 
    CUDA::curand 
    CUDA::cusolver 
    CUDA::cusparse
    ${OpenCV_LIBS}
    ${WIN_LIBS}
    # Specific OpenCV libraries (adjust paths as needed)
    "${OpenCV_DIR}/x64/vc16/lib/opencv_world490.lib"   
    # "D:\\Tools\\opencv4.9\\opencv\\build\\x64\\vc16\\lib\\opencv_world490.lib"
    #GxIAPICPPEx.dll：封装过后的通用并且统一的编程接口
    # "D:\\Tools\\camera\\Galaxy\\GalaxySDK\\Development\\C++ SDK\\lib\\x64\\GxIAPICPPEx.lib"
    "${GALAXY_SDK_PATH}/lib/x64/GxIAPICPPEx.lib"
    #自定偏振解马赛克库
    "${POLAR_LIB_PATH}/lib/PolarizationProcessor.lib" 
    "${POLAR_LIB_PATH}/lib/CudaImageUtils.lib" 
)



#-------------------------------------------------------------------------------
# Output configuration
#-------------------------------------------------------------------------------
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/)
