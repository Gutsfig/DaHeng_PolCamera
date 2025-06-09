#include <iostream>
#include <thread>
#include <opencv2/opencv.hpp> // 引入OpenCV头文件
#include "ThreadSafeQuene.h"
#include "GalaxyIncludes.h" //大恒头文件
#include"polarization_kernels.h" //偏振核函数头文件
#include "CudaImageUtils.h"

//定义全局变量
bool Capture_Flag = true;  //控制线程的标志
int visdeviceID = 0;
int channels = 1; 

//定义参数(调节曝光时间和增益)
#define AE_Time_MAX 10000 //最大曝光时长
#define INIT_GAIN 10.0f //初始增益
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// int Width = 2448; // 可见光宽度
// int Height = 2048;// 可见光高度
//控制器全局声明
CGXFeatureControlPointer ObjFeatureControlPtr;
CGXStreamPointer ObjStreamPtr;
ICaptureEventHandler* pCaptureEventHandler = NULL; 
CGXDevicePointer ObjDevicePtr;


//原始DoFP队列
ThreadSafeMatQueue VisimageQuene_A;
//显示线程
ThreadSafeMatQueue ProcessedDisPlayQueue_AoLP;
ThreadSafeMatQueue ProcessedDisPlayQueue_DoLP;
ThreadSafeMatQueue ProcessedDisPlayQueue_I90;
ThreadSafeMatQueue ProcessedDisPlayQueue_S0;
ThreadSafeMatQueue ProcessedDisPlayQueue_DoFP;
//保存线程
ThreadSafeMatQueue ProcessedSaveQueue_AoLP;
ThreadSafeMatQueue ProcessedSaveQueue_DoLP;
ThreadSafeMatQueue ProcessedSaveQueue_I90;
ThreadSafeMatQueue ProcessedSaveQueue_S0;
ThreadSafeMatQueue ProcessedSaveQueue_DoFP;




//函数声明
void VISprocessThread(); 
void VISdisplayThread();
void VISsaveThread();
void CleanupSession();

void VisInit();
void VisStartcap();
void VisStopcap();

//cuda实现，待完成
/*----------------------- AoLP可视化 ------------------------- */    
// cv::Mat visualizeAoLP(const cv::Mat& aolp, const cv::Mat& dolp) {
//     cv::Mat hsv(aolp.size(), CV_8UC3);
//     cv::Mat bgr(aolp.size(), CV_8UC3);

//     for (int r = 0; r < aolp.rows; ++r) {
//         for (int c = 0; c < aolp.cols; ++c) {
//             float aolp_rad = aolp.at<float>(r, c); // AoLP in radians [-pi/2, pi/2]
//             float dolp_val = dolp.at<float>(r, c); // DoLP [0, 1]
//             // Map AoLP from [-pi/2, pi/2] to Hue [0, 180] for OpenCV HSV
//             unsigned char hue = static_cast<unsigned char>(((aolp_rad / M_PI) + 0.5) * 180.0); // Maps -pi/2->0, 0->90, pi/2->180
//             // Set Saturation based on DoLP (higher DoLP = more saturated color)
//             unsigned char saturation = static_cast<unsigned char>(dolp_val * 255.0);
//             // Set Value to max (or base it on S0 intensity if desired)
//             unsigned char value = 255;
//             hsv.at<cv::Vec3b>(r, c) = cv::Vec3b(hue, saturation, value);
//         }
//     }
//     cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
//     return bgr;
// } 

/*----------------------- 可见光偏振相机回调函数 ------------------------- */       
// 用户定义回调采集函数的具体实现，当回调采集事件发生时，自动调用该函数
// class CSampleCaptureEventHandler : public ICaptureEventHandler {
// public:
//     void DoOnImageCaptured(CImageDataPointer &objImageDataPointer, void *pUserParam) 
// 	{
//         if (pRaw8Buffer != nullptr) pRaw8Buffer = nullptr;
//         if (GX_FRAME_STATUS_SUCCESS == objImageDataPointer->GetStatus()) {
//             int width = objImageDataPointer->GetWidth();
//             int height = objImageDataPointer->GetHeight();
//             // 假设原始数据是Mono8图像->GX_BIT_0_7
//             void* pRaw8Buffer = objImageDataPointer->ConvertToRaw8(GX_BIT_0_7);
//             cv::Mat frame_sdk_wrapper(height, width, CV_8UC1, pRaw8Buffer);
//             // unsigned char* h_input = static_cast<unsigned char*>(pRaw8Buffer);            
//             VisimageQuene_A.push(frame_sdk_wrapper.clone());
//             // VisimageQuene.push(flip_ptr, width, height, channels);
//         }
//     }
// };
//256fps
class CSampleCaptureEventHandler : public ICaptureEventHandler {
    public:
        std::atomic<long long> total_callback_time_us{0}; // 总回调时间 (微秒)
        std::atomic<int> callback_count{0};               // 回调次数
    
        void DoOnImageCaptured(CImageDataPointer &objImageDataPointer, void *pUserParam)
        {
            auto start_time = std::chrono::high_resolution_clock::now(); // 记录开始时间
    
            if (GX_FRAME_STATUS_SUCCESS == objImageDataPointer->GetStatus()) {
                int width = objImageDataPointer->GetWidth();
                int height = objImageDataPointer->GetHeight();
                //产生mono8的单桢
                void* pSDKBuffer = objImageDataPointer->ConvertToRaw8(GX_BIT_0_7);
                if (pSDKBuffer) {
                    cv::Mat frame_sdk_wrapper(height, width, CV_8UC1, pSDKBuffer);
                    VisimageQuene_A.push(frame_sdk_wrapper.clone());
                } else {
                    fprintf(stderr, "错误: 回调函数中 ConvertToRaw8 失败。\n");
                }
            }
    
            auto end_time = std::chrono::high_resolution_clock::now(); // 记录结束时间
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            total_callback_time_us += duration.count();
            callback_count++;
    
            // 每隔一定数量的回调打印一次平均执行时间，避免频繁打印影响性能
            if (callback_count > 0 && callback_count % 100 == 0) {
                printf("回调函数平均执行时间: %lld 微秒\n", total_callback_time_us / callback_count);
            }
        }
    };  
/*----------------------- 主函数 ------------------------- */       
int main() {
    std::cout << "----------------- System Start ------------------\n" << std::endl;
    // 初始化库，才能调用相关的功能
    IGXFactory::GetInstance().Init();
    try {
        VisInit();
        // 启动可见光摄像头线程：处理
        std::thread processThread(VISprocessThread);
        // 启动可见光摄像头线程：显示
        std::thread displayThread(VISdisplayThread);
        //启动可见光摄像头线程：保存
        std::thread saveThread(VISsaveThread);
        // -----------------    Galaxy    ----------------- //
		VisStartcap();
        // 此时开采成功,控制台打印信息,直到输入任意键继续
		while (Capture_Flag){Sleep(1);}
		VisStopcap();
        CleanupSession();
        processThread.join();
        displayThread.join();
        saveThread.join();
    }
    catch (CGalaxyException &e) {
        std::cout << "错误码: " << e.GetErrorCode() << std::endl;
        std::cout << "错误描述信息 : " << e.what() << std::endl;
    }
    catch (std::exception &e) {
        std::cout << "错误描述信息 : " << e.what() << std::endl;
    }
 
    // 反初始化库
    IGXFactory::GetInstance().Uninit();   
    // 销毁事件回调指针
    if (NULL != pCaptureEventHandler) {
        delete pCaptureEventHandler;
        pCaptureEventHandler = NULL;
    }

	std::cout << "\n----------------- System Exit -------------------" << std::endl;
    return 0;
}

void VisInit(){
     // 枚举设备
		GxIAPICPP::gxdeviceinfo_vector vectorDeviceInfo;
        //参数扫描时长，并将获取的信息保存到相应列表之中。
		IGXFactory::GetInstance().UpdateDeviceList(1000, vectorDeviceInfo);
		if (0 == vectorDeviceInfo.size()) {
			std::cout << "无可用设备!" << std::endl;
		}
		// 打开第一台设备以及设备下面第一个流
		ObjDevicePtr = IGXFactory::GetInstance().OpenDeviceBySN(vectorDeviceInfo[0].GetSN(), GX_ACCESS_EXCLUSIVE);
		ObjStreamPtr = ObjDevicePtr->OpenStream(visdeviceID);
		// 获取远端设备属性控制器
		ObjFeatureControlPtr = ObjDevicePtr->GetRemoteFeatureControl();
		// 获取流层属性控制器
		CGXFeatureControlPointer objStreamFeatureControlPtr = ObjStreamPtr->GetFeatureControl();
		// 设置 Buffer 处理模式
		objStreamFeatureControlPtr->GetEnumFeature("StreamBufferHandlingMode")->SetValue("OldestFirst");
		// 设置曝光时间
        ObjFeatureControlPtr->GetFloatFeature("ExposureTime")->SetValue(AE_Time_MAX);
        // 设置增益
        CFloatFeaturePointer objGain = ObjFeatureControlPtr->GetFloatFeature("Gain");
        objGain->SetValue(INIT_GAIN);
        std::cout << "VIS camera: " << vectorDeviceInfo[0].GetModelName() << ": 初始化成功！" << std::endl;
}
void VisStartcap(){
    // 注册回调采集
    pCaptureEventHandler = new CSampleCaptureEventHandler();
    ObjStreamPtr->RegisterCaptureCallback(pCaptureEventHandler, NULL);
    // 开启流层通道
    ObjStreamPtr->StartGrab();
    //给设备发送开采命令
    ObjFeatureControlPtr->GetCommandFeature("AcquisitionStart")->Execute();
}

void VisStopcap(){
    // 发送停采命令
    ObjFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();
    // ObjFeatureControlPtr->GetEnumFeature("ExposureAuto")->SetValue("Off");
    ObjStreamPtr->StopGrab();
    // 注销采集回调
    ObjStreamPtr->UnregisterCaptureCallback();
    // 释放资源
    ObjStreamPtr->Close();
    ObjDevicePtr->Close();
    printf("VIS camera MER2-502-79U3C: 关闭！\n");
}
/*---------------------- Process Frame -----------------------------*/
void VISprocessThread() {
    
	cv::Mat i0, i45, i90, i135,s0,s1,s2, dolp, aolp; // Add dolp, aolp
	cv::Mat i0_display, i45_display, i90_display, i135_display, s0_display,s1_display,s2_display,dolp_display,aolp_display;
    cv::Mat VisdisplayFrame_DoFP;
	//核函数解算初始化
    PolarizationProcessor polar_processor(2448, 2048, true); // Use true for pinned memory
    auto lastTime = std::chrono::steady_clock::now();
    int frameCount = 0;
    double fps = 0.0;

    printf("ProcessThread 启动\n");
    while (Capture_Flag) {
        cv::Mat received_frame = VisimageQuene_A.pop();
            if (received_frame.empty()) {
                if (!Capture_Flag) {
                    printf("VISprocessThread: 收到退出信号，线程结束。\n");
                    break;
                }
                printf("VISprocessThread: 从 VisimageQuene_A 收到空帧！\n");
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }                
    
            //深拷贝DoFP
            VisdisplayFrame_DoFP = received_frame.clone();
            // ProcessedDisPlayQueue_DoFP.push(VisdisplayFrame_DoFP);
            
           

            //双线性插值解马赛克cuda运算
            // bool success_polarization = Bilinear_Interpolation_And_Polarization_CUDA(received_frame, i0, i45, i90, i135,s0,s1,s2, dolp, aolp);
            //直接提取解马赛克cuda运算
            bool success = polar_processor.process_frame(received_frame,i0, i45, i90, i135,s0, s1, s2,dolp, aolp);

            //归一化cuda运算
            bool norm_ok = true;
            // norm_ok = normalize_minmax_cuda_32f_to_8u(dolp, dolp_display);
            // norm_ok =  normalize_minmax_cuda_32f_to_8u(i90, i90_display);
            norm_ok =  normalize_minmax_cuda_32f_to_8u(s0, s0_display);
            // norm_ok = normalize_minmax_cuda_32f_to_8u(aolp, aolp_display);



            //显示
            // ProcessedDisPlayQueue_I90.push(i90_display.clone()); // 存入处理后的图像
            // ProcessedDisPlayQueue_DoLP.push(dolp_display.clone()); // 存入处理后的图像
            ProcessedDisPlayQueue_S0.push(s0_display.clone()); // 存入处理后的图像
            // ProcessedDisPlayQueue_AoLP.push(dolp_display.clone()); // 存入处理后的图像
            // ProcessedDisPlayQueue_AoLP.push(aolp_display_hsv); // 存入处理后的图像
            

            //保存
            // ProcessedSaveQueue_I90.push(i90_display.clone()); // 存入处理后的图像
			// ProcessedSaveQueue_DoLP.push(dolp_display.clone()); // 存入处理后的图像
            // ProcessedSaveQueue_S0.push(s0_display.clone()); // 存入处理后的图像
            // ProcessedSaveQueue_AoLP.push(aolp_display.clone());
            ProcessedSaveQueue_DoFP.push(VisdisplayFrame_DoFP);
            

            frameCount++;
            auto currentTime = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = currentTime - lastTime;
            if (elapsed.count() >= 1.0) { // 每秒更新帧率
                fps = frameCount / elapsed.count();
                printf("处理 FPS: %.2f\n", fps);
                frameCount = 0;
                lastTime = currentTime;
            }
    }
}


void VISdisplayThread() {
    // 窗口名称
    // std::string VISwindowName_DoLP = "DoLP";
    // std::string VISwindowName_I90 = "I90";
    std::string VISwindowName_S0 = "S0";
    // std::string VISwindowName_AoLP = "AoLP";
    // std::string VISwindowName_DoFP = "DoFP";


    // 创建窗口
    // cv::namedWindow(VISwindowName_I90, cv::WINDOW_NORMAL);
    // cv::namedWindow(VISwindowName_DoLP, cv::WINDOW_NORMAL);
    cv::namedWindow(VISwindowName_S0, cv::WINDOW_NORMAL);
    // cv::namedWindow(VISwindowName_AoLP, cv::WINDOW_NORMAL);
    // cv::namedWindow(VISwindowName_DoFP, cv::WINDOW_NORMAL);


    // 调整窗口大小
    // cv::resizeWindow(VISwindowName_I90, 640, 512);
    // cv::resizeWindow(VISwindowName_DoLP, 640, 512);
    cv::resizeWindow(VISwindowName_S0, 640, 512);
    // cv::resizeWindow(VISwindowName_AoLP, 640, 512);
    // cv::resizeWindow(VISwindowName_DoFP, 640, 512);


    // 用于存储当前要显示的图像的Mat对象
    cv::Mat dolp_to_show, i90_to_show, s0_to_show, aolp_to_show, dofp_to_show;

    // 用于显示FPS的计时器 (可选, 也可以显示来自处理线程的FPS)
    auto lastDisplayTime = std::chrono::steady_clock::now();
    int displayFrameCount = 0;
    double display_fps = 0.0;

    printf("VISdisplayThread 已启动。\n");
    while (Capture_Flag) {

        // cv::Mat temp_dofp = ProcessedDisPlayQueue_DoFP.pop(); // 这是 Q2_DoFP
        // if (!temp_dofp.empty()) {
        //     dofp_to_show = temp_dofp;
        // } else if (!Capture_Flag) { // 如果收到退出信号且队列为空
        //     printf("VISdisplayOnlyThread: 通过空的DoFP收到退出信号，线程结束。\n");
        //     break;
        // }
        
        // cv::Mat temp_dolp = ProcessedDisPlayQueue_DoLP.pop();
        // if (!temp_dolp.empty()) dolp_to_show = temp_dolp;
            
        // cv::Mat temp_i90 = ProcessedDisPlayQueue_I90.pop();
        // if (!temp_i90.empty()) i90_to_show = temp_i90;

        cv::Mat temp_s0 = ProcessedDisPlayQueue_S0.pop();
        if (!temp_s0.empty()) s0_to_show = temp_s0;

        // cv::Mat temp_aolp = ProcessedDisPlayQueue_AoLP.pop();
        // if (!temp_aolp.empty()) aolp_to_show = temp_aolp;

       


        // 显示本地 to_show Mat 对象中的内容
        if (!s0_to_show.empty()) {
            // 计算显示FPS
            displayFrameCount++;
            auto currentTime = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = currentTime - lastDisplayTime;
            if (elapsed.count() >= 1.0) {
                display_fps = displayFrameCount / elapsed.count();
                displayFrameCount = 0;
                lastDisplayTime = currentTime;
            }
            std::string fpsText_display = "FPS: " + std::to_string(static_cast<int>(display_fps));
            // 可以绘制处理FPS (从其他线程获取) 或显示FPS
            // cv::putText(dofp_to_show, processing_fps_text, ...); // 如果传递处理FPS
            cv::putText(s0_to_show, fpsText_display, cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0,255,0), 3);
            cv::imshow(VISwindowName_S0, s0_to_show);
        }

        // if (!dolp_to_show.empty()) cv::imshow(VISwindowName_DoLP, dolp_to_show);
        // if (!i90_to_show.empty()) cv::imshow(VISwindowName_I90, i90_to_show);
        // if (!s0_to_show.empty()) cv::imshow(VISwindowName_S0, s0_to_show);
        // if (!aolp_to_show.empty()) cv::imshow(VISwindowName_AoLP, aolp_to_show); // 如果使用AoLP，取消注释
        // if (!dofp_to_show.empty()) cv::imshow(VISwindowName_DoFP, dofp_to_show); 
        char key = cv::waitKey(1);                                     
    }
    cv::destroyAllWindows(); // 在这里销毁窗口，因为它们是在这个线程中创建的
}
/*---------------------- Saving Frame -----------------------------*/
void VISsaveThread() {
    int saveInterval = 2;  // 每隔 2 帧保存 1 帧
    int VISframeCount;
    printf("SaveThread 已启动。\n");
    while (Capture_Flag) {
         
        // 取出处理后的图像
        cv::Mat processedImag_DoFP = ProcessedSaveQueue_DoFP.pop();
        // cv::Mat processedImag_I90 = ProcessedSaveQueue_I90.pop();
        // cv::Mat processedImag_DoLP = ProcessedSaveQueue_DoLP.pop();
        // cv::Mat processedImag_AoLP = ProcessedSaveQueue_AoLP.pop();
        // cv::Mat processedImag_S0 = ProcessedSaveQueue_S0.pop();
        VISframeCount++;
        // 如果达到保存间隔，才保存
        if (VISframeCount % saveInterval == 0) {
            // std::string filename_90 = "D:/image/caiji/I90/" + std::to_string(VISframeCount) + ".bmp";
            // cv::imwrite(filename_90, processedImag_I90);
            // std::string filename_dolp = "D:/image/temp/DoLP/" + std::to_string(VISframeCount) + ".bmp";
            // cv::imwrite(filename_dolp, processedImag_DoLP);
            // std::string filename_aolp = "D:/image/caiji/AoLP/" + std::to_string(VISframeCount) + ".bmp";
            // cv::imwrite(filename_aolp, processedImag_AoLP);
            // std::string filename_s0 = "D:/image/temp/S0/" + std::to_string(VISframeCount) + ".bmp";
            // cv::imwrite(filename_s0, processedImag_S0);
            std::string filename_dofp = "D:/image/temp/DoFP/" + std::to_string(VISframeCount) + ".bmp";
            cv::imwrite(filename_dofp, processedImag_DoFP);
        }
    }
}
/*---------------------- 清除缓存 -----------------------------*/
void CleanupSession() {
    //原始队列
    VisimageQuene_A.push(cv::Mat());
    //显示队列
    ProcessedDisPlayQueue_DoLP.push(cv::Mat());
    ProcessedDisPlayQueue_I90.push(cv::Mat());
    ProcessedDisPlayQueue_S0.push(cv::Mat());
    // ProcessedDisPlayQueue_DoFP.push(cv::Mat());
    // ProcessedImageQueue_AoLP.push(cv::Mat());

    // ProcessedSaveQueue_DoLP.push(cv::Mat());
    // ProcessedSaveQueue_I90.push(cv::Mat());
    // ProcessedSaveQueue_S0.push(cv::Mat());
    ProcessedSaveQueue_DoFP.push(cv::Mat());
    cv::destroyAllWindows();
}
