#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <iostream>
#include <thread>
#include <opencv2/opencv.hpp> // 引入OpenCV头文件
#include "ThreadSafeQuene.h"
#include "GalaxyIncludes.h" //大恒头文件
#include"bilinear.h" //双线性插值头文件
//定义全局变量
bool Capture_Flag = true;  //控制线程的标志
// int IRdeviceID = 0;
int visdeviceID = 0;
int channels = 1; 

#define AE_Time_INIT 300 //初始曝光时长
#define AE_Time_MAX 60000 //最大曝光时长
#define INIT_GAIN 10.0f //初始增益
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void* pRaw8Buffer = nullptr;     
unsigned char* pVISBuffer = nullptr;    

// 创建全局队列实例
ThreadSafeQueue VisimageQuene; // 可见光摄像头采集图像队列
ThreadSafeQueue SaveVisimageQuene;
ThreadSafeMatQueue ProcessedImageQueue_AoLP;
ThreadSafeMatQueue ProcessedImageQueue_DoLP;
ThreadSafeMatQueue ProcessedImageQueue_I90;
ThreadSafeMatQueue ProcessedImageQueue_S0;
cv::Mat VisdisplayFrame;
// cv::Mat inputname;
// cv::Mat SaveVisdisplayFrame;
// 全局计数器，用于给帧命名
std::atomic<int> frameCounter(0);   //多线程计数
int VISframeCount = 0;
//函数声明
/*----------------------- Galaxy ------------------------- */  
void VISdisplayThread(); // 单采集25帧
/*----------------------- Galaxy ------------------------- */  

/*----------------------- 其它操作 ------------------------- */  
void keyListener() ;
void CleanupSession();
/*----------------------- Save ------------------------- */       
// void SaveCameraFrame(const unsigned char* imageData, int width, int height, const char* folderPath) ;
void FrameStore();
/*----------------------- AoLP可视化 ------------------------- */    
cv::Mat visualizeAoLP(const cv::Mat& aolp, const cv::Mat& dolp) {
    cv::Mat hsv(aolp.size(), CV_8UC3);
    cv::Mat bgr(aolp.size(), CV_8UC3);

    for (int r = 0; r < aolp.rows; ++r) {
        for (int c = 0; c < aolp.cols; ++c) {
            float aolp_rad = aolp.at<float>(r, c); // AoLP in radians [-pi/2, pi/2]
            float dolp_val = dolp.at<float>(r, c); // DoLP [0, 1]

            // Map AoLP from [-pi/2, pi/2] to Hue [0, 180] for OpenCV HSV
            unsigned char hue = static_cast<unsigned char>(((aolp_rad / M_PI) + 0.5) * 180.0); // Maps -pi/2->0, 0->90, pi/2->180

            // Set Saturation based on DoLP (higher DoLP = more saturated color)
            unsigned char saturation = static_cast<unsigned char>(dolp_val * 255.0);

            // Set Value to max (or base it on S0 intensity if desired)
            unsigned char value = 255;

            hsv.at<cv::Vec3b>(r, c) = cv::Vec3b(hue, saturation, value);
        }
    }
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    return bgr;
}
/*----------------------- AoLP可视化 ------------------------- */    
/*----------------------- 可见光偏振相机回调函数 ------------------------- */       
// 用户定义回调采集函数的具体实现，当回调采集事件发生时，自动调用该函数
class CSampleCaptureEventHandler : public ICaptureEventHandler {
public:
    void DoOnImageCaptured(CImageDataPointer &objImageDataPointer, void *pUserParam) 
	{
        if (pRaw8Buffer != nullptr) pRaw8Buffer = nullptr;
        if (objImageDataPointer->GetStatus() == GX_FRAME_STATUS_SUCCESS) {
            int width = objImageDataPointer->GetWidth();
            int height = objImageDataPointer->GetHeight();
            // void* pRaw8Buffer = objImageDataPointer->ConvertToRGB24(GX_BIT_0_7, GX_RAW2RGB_NEIGHBOUR, false);
            void* pRaw8Buffer = objImageDataPointer->ConvertToRaw8(GX_BIT_0_7);
            unsigned char* h_input = static_cast<unsigned char*>(pRaw8Buffer);            
            VisimageQuene.push(h_input, width, height, channels);
            
        }
        
    }
};
/*----------------------- 可见光偏振相机回调函数 ------------------------- */       
/*----------------------- 主函数 ------------------------- */       
int main() {
    std::cout << "----------------- System Start ------------------\n" << std::endl;
    ICaptureEventHandler* pCaptureEventHandler = NULL; //<采集回调对象
    // 初始化库，才能调用相关的功能
    IGXFactory::GetInstance().Init();
    try {
        // 枚举设备
		GxIAPICPP::gxdeviceinfo_vector vectorDeviceInfo;
        //参数扫描时长，并将获取的信息保存到相应列表之中。
		IGXFactory::GetInstance().UpdateDeviceList(1000, vectorDeviceInfo);
		if (0 == vectorDeviceInfo.size()) {
			std::cout << "无可用设备!" << std::endl;
		}
		// 打开第一台设备以及设备下面第一个流
		CGXDevicePointer ObjDevicePtr = IGXFactory::GetInstance().OpenDeviceBySN(vectorDeviceInfo[0].GetSN(), GX_ACCESS_EXCLUSIVE);
		CGXStreamPointer ObjStreamPtr = ObjDevicePtr->OpenStream(visdeviceID);
		// 获取远端设备属性控制器
		CGXFeatureControlPointer ObjFeatureControlPtr = ObjDevicePtr->GetRemoteFeatureControl();
		// 获取流层属性控制器
		CGXFeatureControlPointer objStreamFeatureControlPtr = ObjStreamPtr->GetFeatureControl();
		// 设置 Buffer 处理模式
		objStreamFeatureControlPtr->GetEnumFeature("StreamBufferHandlingMode")->SetValue("OldestFirst");
		// 设置曝光时间
        // ObjFeatureControlPtr->GetFloatFeature("ExposureTime")->SetValue(AE_Time_INIT); 
        // ObjFeatureControlPtr->GetFloatFeature("AutoExposureTimeMax")->SetValue(AE_Time_MAX); 
        // ObjFeatureControlPtr->GetEnumFeature("ExposureTimeMode")->SetValue("STANDARD");
        // ObjFeatureControlPtr->GetEnumFeature("ExposureAuto")->SetValue("Continuous"); //Continuous、Off、Once !!!需要在退出时设置为off
        // 设置增益
        CFloatFeaturePointer objGain = ObjFeatureControlPtr->GetFloatFeature("Gain");
        //CFloatFeaturePointer objET = ObjFeatureControlPtr->GetFloatFeature("ExposureTime");
        objGain->SetValue(INIT_GAIN);
        std::cout << "VIS camera: " << vectorDeviceInfo[0].GetModelName() << ": 初始化成功！" << std::endl;
        // 启动可见光摄像头线程：显示
        std::thread visdisplay(VISdisplayThread);
        // 创建两个线程分别处理两个相机的图像保存
        std::thread SaveThread(FrameStore);
        // -----------------    Galaxy    ----------------- //
		// 注册回调采集
		pCaptureEventHandler = new CSampleCaptureEventHandler();
		ObjStreamPtr->RegisterCaptureCallback(pCaptureEventHandler, NULL);
		// 开启流层通道
		ObjStreamPtr->StartGrab();
        //给设备发送开采命令
        ObjFeatureControlPtr->GetCommandFeature("AcquisitionStart")->Execute();

		// 此时开采成功,控制台打印信息,直到输入任意键继续
		while (Capture_Flag){Sleep(1);}

		// 发送停采命令
		ObjFeatureControlPtr->GetCommandFeature("AcquisitionStop")->Execute();
        //ObjFeatureControlPtr->GetEnumFeature("ExposureAuto")->SetValue("Off");
		ObjStreamPtr->StopGrab();
		// 注销采集回调
		ObjStreamPtr->UnregisterCaptureCallback();
		// 释放资源
		ObjStreamPtr->Close();
		ObjDevicePtr->Close();
        printf("VIS camera MER2-502-79U3C: 关闭！\n");
        // -----------------    Galaxy    ----------------- //

        CleanupSession();

        visdisplay.join();
        SaveThread.join();
        //  按任意键退出程序
        // listenerThread.join();  // 等待键盘监听线程结束
        // 主线程等待两个摄像头线程完成（实际上这里是永远等待，除非中断）
        // SaveThread.join();

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
/*---------------------- Displaying Frame -----------------------------*/
void VISdisplayThread() {
    std::string VISwindowName_DoLP = "DoLP";
    std::string VISwindowName_I90 = "I90";
    std::string VISwindowName_S0 = "S0";
    std::string VISwindowName_AoLP = "AoLP";
    cv::namedWindow("I90", cv::WINDOW_NORMAL);
    cv::namedWindow("DoLP", cv::WINDOW_NORMAL);
    cv::namedWindow("AoLP", cv::WINDOW_NORMAL);
    cv::namedWindow("S0", cv::WINDOW_NORMAL);
    cv::resizeWindow("I90", 640, 480);
    cv::resizeWindow("DoLP", 640, 480);
    cv::resizeWindow("AoLP", 640, 480);
    cv::resizeWindow("S0", 640, 480);
	cv::Mat i0, i45, i90, i135,s0,s1,s2, dolp, aolp; // Add dolp, aolp
	cv::Mat i0_display, i45_display, i90_display, i135_display, s0_display,s1_display,s2_display,dolp_display;
	cv::Mat aolp_display_hsv;
    auto lastTime = std::chrono::steady_clock::now();
    int frameCount = 0;
    double fps = 0.0;
    while (Capture_Flag) {
        auto [pVISBuffer, width, height, channels] = VisimageQuene.pop(); // 获取处理后的数据
        if (pVISBuffer == nullptr || width == 0){
            printf("pVISBuffer is Null!\n");
        } // 如果收到空指针，退出线程   
        else{                        
            // 转换为 OpenCV Mat
			VisdisplayFrame = cv::Mat(height, width, CV_8U, pVISBuffer); // 处理后的图像是单通道的
			bool success = Bilinear_Interpolation_And_Polarization_CUDA(VisdisplayFrame, i0, i45, i90, i135,s0,s1,s2, dolp, aolp);
            cv::normalize(dolp, dolp_display, 0, 255, cv::NORM_MINMAX, CV_8U);
			cv::normalize(i90, i90_display, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::normalize(s0, s0_display, 0, 255, cv::NORM_MINMAX, CV_8U);
			aolp_display_hsv = visualizeAoLP(aolp, dolp);
            ProcessedImageQueue_I90.push(i90_display); // 存入处理后的图像
			ProcessedImageQueue_DoLP.push(dolp_display); // 存入处理后的图像
            ProcessedImageQueue_S0.push(s0_display); // 存入处理后的图像
            ProcessedImageQueue_AoLP.push(aolp_display_hsv); // 存入处理后的图像
            // SaveCameraFrame(pVISBuffer, width, height, "Result/VIS");

            frameCount++;
            auto currentTime = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = currentTime - lastTime;
            if (elapsed.count() >= 1.0) { // 每秒更新帧率
                fps = frameCount / elapsed.count();
                frameCount = 0;
                lastTime = currentTime;
            }
            // 在图像上显示帧率
            std::string fpsText = "FPS: " + std::to_string(static_cast<int>(fps));
            //显示dolp
            cv::putText(dolp_display, fpsText, cv::Point(10, 130), cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 0, 255), 10); 
            cv::imshow(VISwindowName_DoLP, dolp_display);
            //显示aolp
			cv::putText(aolp_display_hsv, fpsText, cv::Point(10, 130), cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 0, 255), 10); 
            cv::imshow(VISwindowName_AoLP, aolp_display_hsv);
            //显示I90
			cv::putText(i90_display, fpsText, cv::Point(10, 130), cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 0, 255), 10); 
            cv::imshow(VISwindowName_I90, i90_display);
            //显示S0
            cv::putText(s0_display, fpsText, cv::Point(10, 130), cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(0, 0, 255), 10); 
            cv::imshow(VISwindowName_S0, s0_display);
            keyListener();
            // 显示完毕清除缓存
        }
    }
    delete[] pVISBuffer;

}
/*---------------------- Displaying Frame -----------------------------*/


/*---------------------- 清除缓存 -----------------------------*/
void CleanupSession() {
    // 通知显示线程退出并等待
    VisimageQuene.push(nullptr, 0, 0, 0); 
    
    if (pVISBuffer != nullptr) free(pVISBuffer);
    cv::destroyAllWindows();
}
/*---------------------- 清除缓存 -----------------------------*/


/*---------------------- 按键监听 -----------------------------*/
void keyListener() {
    // 等待按键输入
    char key = cv::waitKey(1);
    if (key == 'q') {
        // 按下 'q' 键后，退出程序
        Capture_Flag = false;
    }
}
/*---------------------- 按键监听 -----------------------------*/

/*---------------------- Saving Frame -----------------------------*/
void FrameStore() {
    int saveInterval = 5;  // 每隔 5 帧保存 1 帧
    int frameCounter = 0;  // 当前帧计数

    while (Capture_Flag) {
        VISframeCount++;
        frameCounter++;

        // 取出处理后的图像
        cv::Mat processedImag_I90 = ProcessedImageQueue_I90.pop();
        cv::Mat processedImag_DoLP = ProcessedImageQueue_DoLP.pop();
        cv::Mat processedImag_AoLP = ProcessedImageQueue_AoLP.pop();
        cv::Mat processedImag_S0 = ProcessedImageQueue_S0.pop();

        // 如果达到保存间隔，才保存
        if (frameCounter % saveInterval == 0) {
            std::string filename_90 = "D:/image/caiji/I90/" + std::to_string(VISframeCount) + ".png";
            cv::imwrite(filename_90, processedImag_I90);
            std::string filename_dolp = "D:/image/caiji/DoLP/" + std::to_string(VISframeCount) + ".png";
            cv::imwrite(filename_dolp, processedImag_DoLP);
            std::string filename_aolp = "D:/image/caiji/AoLP/" + std::to_string(VISframeCount) + ".png";
            cv::imwrite(filename_aolp, processedImag_AoLP);
            std::string filename_s0 = "D:/image/caiji/S0/" + std::to_string(VISframeCount) + ".png";
            cv::imwrite(filename_s0, processedImag_S0);
        }
    }
}

/*---------------------- Saving Frame -----------------------------*/




