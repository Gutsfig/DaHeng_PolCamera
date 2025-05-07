#include "ThreadSafeQuene.h"
/*----------------------- 定义线程安全队列 ------------------------- */      
//允许多个线程安全地推送和弹出图像数据。使用互斥量和条件变量确保了数据的完整性和线程间的同步。
//定义push的方法，将图像推入队列
void ThreadSafeQueue::push(unsigned char* pRaw8Buffer, int width, int height, int channels) {
    std::lock_guard<std::mutex> lock(mtx);   //创建一个 std::lock_guard 对象，使用 mtx 互斥量。这会自动在作用域内加锁
    //使用 emplace 方法将新的图像数据（包含指针和尺寸信息）添加到 buffer 队列中。emplace 会直接在容器中构造元素，避免不必要的拷贝。
    buffer.emplace(pRaw8Buffer, width, height, channels);
    cvar.notify_one(); // 通知处理线程
}
//定义 pop 方法，用于从队列中获取图像数据。返回值是一个包含图像数据及其相关信息的元组。
std::tuple<unsigned char*, int, int, int> ThreadSafeQueue::pop() {
    std::unique_lock<std::mutex> lock(mtx);
    cvar.wait(lock, [this] { return !buffer.empty(); }); // 等待队列非空
    auto item = buffer.front();
    buffer.pop();//从队列中移除前面获取的元素。
    return item;//返回获取的图像数据及其相关信息，作为一个元组。
}


// 新增：ThreadSafeMatQueue 的实现
void ThreadSafeMatQueue::push(const cv::Mat& mat) {
    std::lock_guard<std::mutex> lock(mtx);
    buffer.push(mat.clone());  // 存储副本，避免数据竞争
    cvar.notify_one();         // 通知等待的线程
}

cv::Mat ThreadSafeMatQueue::pop() {
    std::unique_lock<std::mutex> lock(mtx);
    cvar.wait(lock, [this] { return !buffer.empty(); });  // 等待队列非空
    cv::Mat mat = buffer.front();
    buffer.pop();
    return mat;
}