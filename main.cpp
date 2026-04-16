#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <chrono>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <cmath>

using namespace nvinfer1;

struct LetterBoxInfo {
    float scale;
    int pad_w;
    int pad_h;
};

struct Task {
    cv::Mat frame;          // 原图
    std::vector<float> input; // 预处理结果
    LetterBoxInfo lb;       // 还原信息
};

struct Detection {
    int class_id;
    float conf;
    float x, y, w, h;
};


std::vector<char> loadEngine(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

float IoU(const Detection& a, const Detection& b);
LetterBoxInfo letterbox(const cv::Mat& src, cv::Mat& dst, int target = 640);

// 共享队列
std::queue<cv::Mat> frame_queue;
std::queue<cv::Mat> process_queue;
std::mutex mtx1;
std::mutex mtx2;
std::condition_variable cond1;
std::condition_variable cond2;
/*用于跨帧异步, 真TM牛逼兄弟*/
std::queue<Task> task_queue;
std::mutex mtx_input;
std::condition_variable cond_input;
// 控制退出
std::atomic<bool> running(true);
double g_fps = 30.0;


float IoU(const Detection& a, const Detection& b);
// ==================== 生产者线程 ====================
void capture_thread() {
    cv::VideoCapture cap("../OpenCV_test.mp4"); // 先用视频文件，避免摄像头问题
    //cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Error: Cannot open video" << std::endl;
        running = false;
        cond1.notify_all();
        cond2.notify_all();
        return;
    }
	
    cv::Mat frame;
	g_fps = cap.get(cv::CAP_PROP_FPS);
	if (g_fps <= 0) g_fps = 30.0;

	int delay = 1000/g_fps;

    while (running) 
    {
        cap >> frame;
        if (frame.empty()) {
            running = false;
            cond1.notify_all();
            cond2.notify_all();
            break;
        }

        // 入队  
        {     
            std::unique_lock<std::mutex> lock(mtx1);
            frame_queue.push(frame.clone());
            //std::cout << "Queue size: " << frame_queue.size() << std::endl;
        }
        cond1.notify_one(); // 通知处理者
	    //std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    }
}

// ==================== 预处理线程 ====================
void preprocess_thread() {
    while (running) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(mtx1);
        
            cond1.wait(lock, [&] { return !frame_queue.empty() || !running; });
                
            if (!running && frame_queue.empty())    break;

            while (frame_queue.size() > 1) 
            {
                frame_queue.pop();  //只保留最新的队尾数据
            }
        
            frame = frame_queue.front();
            frame_queue.pop();

            lock.unlock();
        }

        //🔥 预处理
        Task task;
        task.frame = frame.clone();
        cv::Mat img = frame.clone();
        task.lb = letterbox(frame, img);
        //BGR2(RGB+float)
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        img.convertTo(img, CV_32F, 1.0 / 255);
        //HWC2CHW
        std::vector<float> input(3 * 640 * 640);
        std::vector<cv::Mat> channels(3);
        cv::split(img, channels);
        for (int c = 0; c < 3; c++) {
            memcpy(input.data() + c * 640 * 640,
            channels[c].data,
            640 * 640 * sizeof(float));
        }
        task.input = std::move(input);
        {
            std::unique_lock<std::mutex> lock(mtx_input);
            task_queue.push(std::move(task));
        }
        cond_input.notify_one();

    }
}

// ==================== 推理线程 ====================
void inference_thread() {
    Logger logger;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::vector<void*> outputBuffers;
    std::vector<std::string> outputNames;
    std::vector<int> outputSizes;   // ⭐新增（非常关键）
    float *inputDevice[2], *outputDevice;
    int channelSize = 640 * 640;
    int inputSize = 3 * 640 * 640;
    float* inputHost[2];
    for (int i = 0; i < 2; i++) {
        cudaMalloc((void **)&inputDevice[i], inputSize * sizeof(float));
        inputHost[i] = new float[inputSize];
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 1️⃣ 读取 engine
    auto engineData = loadEngine("../yolov5n_fp16.engine");

    // 2️⃣ 创建 runtime
    IRuntime* runtime = createInferRuntime(logger);

    // 3️⃣ 反序列化 engine
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());

    // 4️⃣ 创建 context
    IExecutionContext* context = engine->createExecutionContext();

    std::cout << "Engine loaded successfully!" << std::endl;
    //绑定输入输出
    for (int i = 0; i < engine->getNbIOTensors(); i++) {
    const char* name = engine->getIOTensorName(i);
    auto mode = engine->getTensorIOMode(name);

        if (mode == TensorIOMode::kINPUT) {
            context->setTensorAddress(name, inputDevice[0]);
            context->setTensorAddress(name, inputDevice[1]);
        } 
        else if (mode == TensorIOMode::kOUTPUT) {

            // ⭐ 1. 获取 shape
            nvinfer1::Dims dims = context->getTensorShape(name);

            // ⭐ 2. 计算 size
            int size = 1;
            for (int j = 0; j < dims.nbDims; j++) {
                size *= dims.d[j];
            }
            //std::cout << "Output: " << name << " size = " << size << std::endl;
            
            // ⭐ 3. 分配 GPU 内存
            float* outputDev;
            cudaMalloc((void**)&outputDev, size * sizeof(float));

            // ⭐ 4. 绑定
            context->setTensorAddress(name, outputDev);

            // ⭐ 5. 保存
            outputBuffers.push_back(outputDev);
            outputNames.push_back(name);
            outputSizes.push_back(size);
        }
    }
    int size = 2142000;
    float* outputHost[2];
    outputHost[0] = new float[size];
    outputHost[1] = new float[size];

    //上一帧
    bool has_prev = false;
    Task prev_task;
    while (running) {
        static int frame_count = 0;
        //当前帧索引
        int idx = frame_count % 2; 
        //上一帧帧索引
        int prev = 1 - idx;

        static auto t_start = std::chrono::steady_clock::now();
        frame_count++;
        auto t_now = std::chrono::steady_clock::now();
        float sec = std::chrono::duration<float>(t_now - t_start).count();

        if (sec >= 1.0f) {
            std::cout << "FPS: " << frame_count << std::endl;
            frame_count = 0;
            t_start = t_now;
        }
        /*************处理上一帧**************/
        std::vector<Detection> detections;
        if (has_prev){
            cudaStreamSynchronize(stream);  //CPU在这里等待推理完成
            int stride = 85;
            float* data = outputHost[prev];
            
            for (int k = 0; k < 25200; k++) {
                float *Current_Box = data + k * stride;
                float obj = data[k * stride + 4];
                if (obj < 0.9) continue;
                if (obj > 0.5) {
                    float max_class_score = 0;
                    int class_id = -1;
                    
                    for (int j = 0; j < 80; j++) 
                    {
                        if (Current_Box[5 + j] > max_class_score) 
                        {
                            max_class_score = Current_Box[5 + j];
                            class_id = j;
                        }
                    }
                    float final_conf = obj * max_class_score;
                    if (final_conf < 0.5) continue;

                    float x = data[k * stride + 0];
                    float y = data[k * stride + 1];
                    float w = data[k * stride + 2];
                    float h = data[k * stride + 3];
                    detections.push_back({class_id, final_conf, x, y, w, h});
                    /*已经通过置信度筛选成功了，现在是需要开始进行NMS，然后画框*/
                }
            }
            /*开始NMS(去重框)*/
            //先排序
            std::sort(detections.begin(), detections.end(),
            [](const Detection& a, const Detection& b) {
                return a.conf > b.conf;
            });
            if (detections.size() > 100)   detections.resize(100);
            //建立容器类, 用于存放NMS结果
            std::vector<Detection> nms_result;
            float iou_threshold = 0.5;

            for (size_t i = 0; i < detections.size(); i++) {
                bool keep = true;

                for (size_t j = 0; j < nms_result.size(); j++) {
                    if (IoU(detections[i], nms_result[j]) > iou_threshold) {
                        keep = false;
                        break;
                    }
                }

                if (keep) {
                    nms_result.push_back(detections[i]);
                }
            }

            //🔥 draw
            for (auto& det : nms_result)    
            {
                float x1 = (det.x - det.w / 2 - prev_task.lb.pad_w) / prev_task.lb.scale;
                float y1 = (det.y - det.h / 2 - prev_task.lb.pad_h) / prev_task.lb.scale;
                float x2 = (det.x + det.w / 2 - prev_task.lb.pad_w) / prev_task.lb.scale;
                float y2 = (det.y + det.h / 2 - prev_task.lb.pad_h) / prev_task.lb.scale;

                x1 = std::max(0, (int)x1);
                x2 = std::min(prev_task.frame.cols - 1, (int)x2);
                y1 = std::max(5, (int)y1);
                y2 = std::min(prev_task.frame.rows - 1, (int)y2);

                cv::rectangle(prev_task.frame,
                    cv::Point(x1, y1),
                    cv::Point(x2, y2),
                    cv::Scalar(0, 255, 0),
                    2);
                
                std::string label = std::to_string(det.class_id) +
                            " " + std::to_string(det.conf);

                cv::putText(prev_task.frame, label,
                            cv::Point(x1, y1 - 5),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.5,
                            cv::Scalar(0, 255, 0),
                            1);
            }
        }

        /*取当前帧*/
        std::unique_lock<std::mutex> lock(mtx_input);
        cond_input.wait(lock, [&] { return !task_queue.empty() || !running; });
        /*丢帧*/
        while (task_queue.size() > 1) {
            task_queue.pop();
        }
        Task task = task_queue.front();
        task_queue.pop();
        lock.unlock();
        
        // 🔥 提交当前帧
        memcpy(inputHost[idx], task.input.data(), task.input.size() * sizeof(float));
        
        cudaMemcpyAsync(inputDevice[idx], inputHost[idx],
                inputSize * sizeof(float),
                cudaMemcpyHostToDevice,
                stream);

        context->enqueueV3(stream);

        cudaMemcpyAsync(outputHost[idx], outputBuffers[0],
                size * sizeof(float),
                cudaMemcpyDeviceToHost,
                stream);

        
        // 放入第二个队列
        if(has_prev)
        {
            std::unique_lock<std::mutex> lock2(mtx2);
            while (process_queue.size() > 1) {
                process_queue.pop();
            }
            process_queue.push(prev_task.frame.clone());     
            cond2.notify_one();
        }
        prev_task = task;
        has_prev = true;
    }
    delete[] outputHost[0];
    delete[] outputHost[1];

    cudaFree(inputDevice[0]);
    cudaFree(inputDevice[1]);

    delete[] inputHost[0];
    delete[] inputHost[1];

    delete context;
    delete engine;
    delete runtime;
}
// ==================== 主函数 ====================
int main() {
    std::thread t1(capture_thread);
    std::thread t2(preprocess_thread);
    std::thread t3(inference_thread);

    int delay = 1;
	//if (g_fps > 0) delay = 1000 / g_fps;
    while (running) {
        std::unique_lock<std::mutex> lock(mtx2);

        // 等待队列有数据
        cond2.wait(lock, [&] { return !process_queue.empty() || !running; });

        if (!running && process_queue.empty())
            break;
        cv::Mat frame = process_queue.front();
        process_queue.pop();

        lock.unlock(); // 早点释放锁（很重要）

        cv::imshow("Video", frame);

        if (cv::waitKey(delay) == 27) {
            running = false;
            cond1.notify_all();
            cond2.notify_all();
            break;
        }
    }
    t1.join();
    t2.join();
    t3.join();
    return 0;
}

float IoU(const Detection& a, const Detection& b) 
{
    //因为对象都是长方形框，这样可以省事的找出两个长方形围出的小长方形，从而计算IoU
    float x1 = std::max(a.x - a.w / 2, b.x - b.w / 2);
    float y1 = std::max(a.y - a.h / 2, b.y - b.h / 2);
    float x2 = std::min(a.x + a.w / 2, b.x + b.w / 2);
    float y2 = std::min(a.y + a.h / 2, b.y + b.h / 2);

    float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);

    float area_a = a.w * a.h;
    float area_b = b.w * b.h;

    return inter_area / (area_a + area_b - inter_area);
}

LetterBoxInfo letterbox(const cv::Mat& src, cv::Mat& dst, int target) {
    int w = src.cols, h = src.rows;
    //eg: min(640 / 1920, 640 / 1080) = 640 / 1920
    //new_w = 1920 * 640 / 1920 = 640
    //new_h = 1080 * 640 / 1920 = 360
    //这里其实是在等比例缩放
    float scale = std::min(target / (float)w, target / (float)h);

    int new_w = int(w * scale);
    int new_h = int(h * scale);
    /*完成等比例缩放*/
    cv::resize(src, dst, cv::Size(new_w, new_h));

    int pad_w = target - new_w;
    int pad_h = target - new_h;

    int left = pad_w / 2;
    int top = pad_h / 2;

    cv::copyMakeBorder(dst, dst,
        top, pad_h - top,
        left, pad_w - left,
        cv::BORDER_CONSTANT,
        cv::Scalar(114,114,114));

    return {scale, left, top};
}