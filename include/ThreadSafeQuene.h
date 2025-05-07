#ifndef THREADSAFEQUEUE_H   //头文件只包含一次
#define THREADSAFEQUEUE_H

#include <mutex>//标准库中的互斥量（std::mutex），用于线程间的同步。
#include <condition_variable>//用于线程间的通信和同步。
#include <queue> //队列
#include <tuple>//元组
#include <opencv2/opencv.hpp> // OpenCV库，用于图像处理和计算机视觉任务
//定义一个线程安全队列
class ThreadSafeQueue {
public:   //公共成员。可以被类外部访问
//声明 push 方法，用于将图像数据（指针和尺寸信息）推入队列。
    void push(unsigned char* pRaw8Buffer, int width, int height, int channels);
//声明 pop 方法，用于从队列中获取图像数据，并返回一个包含图像数据及其相关信息的元组。
    std::tuple<unsigned char*, int, int, int> pop();

private:   //只能在类内部访问
    std::mutex mtx; //定义一个互斥量 mtx，用于保护对队列的访问，确保线程安全。
    std::condition_variable cvar;//定义一个条件变量 cvar，用于线程间的通知和等待机制。
    std::queue<std::tuple<unsigned char*,int, int, int>> buffer; // 存储(pRaw8Buffer, width, height)
};


// 新增：专门存储 cv::Mat 的线程安全队列
class ThreadSafeMatQueue {
    public:
        void push(const cv::Mat& mat);  // 存储 cv::Mat
        cv::Mat pop();                  // 取出 cv::Mat
    
    private:
        std::mutex mtx;
        std::condition_variable cvar;
        std::queue<cv::Mat> buffer;     // 存储 cv::Mat
    };
#endif // THREADSAFEQUEUE_H
