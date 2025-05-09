cmake_minimum_required(VERSION 3.20)
# Project configuration
project(DaHeng_PolCamera)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89;86)
# set(XXX_ROOT "              ")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe")
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
set(OpenCV_DIR "D:/Tools/opencv4.9/opencv/build")
#-------------------------------------------------------------------------------
# File organization
#-------------------------------------------------------------------------------
set(SOURCES  
    # src/main.cpp
    src/main.cpp
    src/ThreadSafeQuene.cpp
    src/bilinear.cu

)

set(HEADERS
    include/bilinear.h
    include/ThreadSafeQuene.h
)
enable_language(CUDA)
#-------------------------------------------------------------------------------
# Output configuration
#-------------------------------------------------------------------------------
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/)


#-------------------------------------------------------------------------------
link_directories(
    ${TENSORRT_LIB_DIR}
    "${OpenCV_DIR}/x64/vc16/lib"  # OpenCV 库路径
)
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Dependency configuration
#-------------------------------------------------------------------------------
# OpenCV configuration
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS 
    core 
    imgproc 
    highgui 
    #cudaimgproc
)

# Windows-specific libraries
if(WIN32)
    set(WIN_LIBS winmm)
endif()
#-------------------------------------------------------------------------------
# Executable target
#-------------------------------------------------------------------------------
add_executable(${PROJECT_NAME} 
    ${SOURCES} 
    ${HEADERS}
)
#-------------------------------------------------------------------------------
# Target configurations
#-------------------------------------------------------------------------------
target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include 
    ${TENSORRT_INCLUDE_DIR}
    "D:\\Tools\\opencv4.9\\opencv\\build\\include\\opencv2"
    "D:\\Tools\\camera\\Galaxy\\GalaxySDK\\Development\\C++ SDK\\inc"
    "D:\\Tools\\camera\\Galaxy\\GalaxySDK\\Development\\VC SDK\\inc"
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
    #"D:/Tools/opencv470/opencv-4.7.0/build/install/x64/vc16/lib/opencv_img_hash470.lib"
    "D:\\Tools\\opencv4.9\\opencv\\build\\x64\\vc16\\lib\\opencv_world490.lib"
    "D:\\Tools\\camera\\Galaxy\\GalaxySDK\\Development\\C++ SDK\\lib\\x64\\GxIAPICPPEx.lib"
)
#---------------------------------------------------------------------
target_link_directories(${PROJECT_NAME} PRIVATE
    ${TENSORRT_LIB_DIR}
    "${OpenCV_DIR}/install/x64/vc16/lib"
)
#-------------------------------------------------------------------------