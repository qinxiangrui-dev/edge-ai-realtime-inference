#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <chrono>

struct Detection {
    int class_id;
    float conf;
    float x, y, w, h;
};

// 共享队列
std::queue<cv::Mat> frame_queue;
std::queue<cv::Mat> process_queue;
std::mutex mtx1;
std::mutex mtx2;
std::condition_variable cond1;
std::condition_variable cond2;

// 控制退出
std::atomic<bool> running(true);
double g_fps = 30.0;


float IoU(const Detection& a, const Detection& b);
// ==================== 生产者线程 ====================
void capture_thread() {
    cv::VideoCapture cap("../OpenCV_test.mp4"); // 先用视频文件，避免摄像头问题

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
        std::cout << "capture running" << std::endl;
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
	    std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    }
}

// ==================== 消费者线程 ====================
void display_thread() {
	int delay = 33;
	if (g_fps > 0) delay = 1000 / g_fps;
    while (running) {
        std::cout << "display running" << std::endl;
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
}
void process_thread() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLO");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    Ort::Session session(env, "../yolov5n.onnx", session_options);

    std::cout << "Model loaded!" << std::endl;
    static auto last_time = std::chrono::steady_clock::now();

    while (running) {
        std::cout << "process running" << std::endl;
        std::unique_lock<std::mutex> lock(mtx1);
    
        cond1.wait(lock, [&] { return !frame_queue.empty() || !running; });
        if (!running && frame_queue.empty())
            break;

        while (frame_queue.size() > 1) 
        {
            frame_queue.pop();  //只保留最新的队尾数据
        }
        
        cv::Mat frame = frame_queue.front();
        frame_queue.pop();

        lock.unlock();

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count() < 200) {
            continue;
        }
        last_time = now;

        // 🔥AI处理
        /*🔥 获得图像拷贝*/
        cv::Mat input = frame.clone();
        /*🔥 修改图像大小，便于yolo处理*/
        cv::resize(input, input, cv::Size(320, 320));
        /*🔥 修改色系坐标，OpenCV默认用BGR的图，Yolo默认使用RBG的图*/
        cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
        /*🔥转为浮点表示 色系：整型0 - 255, 浮点：0 - 1, 所以精度是 1.0 / 255.0*/
        input.convertTo(input, CV_32F, 1.0 / 255.0);

        /*将图片拷贝为输入张量形式*/
        std::vector<float> input_tensor_values(1 * 3 * 320 * 320);
        std::vector<cv::Mat> channels(3);
        cv::split(input, channels);

        for (int i = 0; i < 3; i++) {
            memcpy(input_tensor_values.data() + i * 320 * 320,
                   channels[i].data,
                   320 * 320 * sizeof(float));
        }
        //创建张量
        std::vector<int64_t> input_shape = {1, 3, 320, 320};

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            input_shape.data(),
            input_shape.size()
        );
        // 🔥 7. 获取输入输出名字
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
        const char* input_name = input_name_ptr.get();
        auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
        const char* output_name = output_name_ptr.get();

        // 🔥 8. 推理
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            &input_name,
            &input_tensor,
            1,
            &output_name,
            1
        );
        // 🔥 9. 输出结果shape
        auto& output = output_tensors[0];

        auto shape = output.GetTensorTypeAndShapeInfo().GetShape();

        //🔥 10. 找出我们需要的候选框
        float* output_data = output.GetTensorMutableData<float>();

        int num_boxes = shape[1];   // 🔥 动态获取
        int num_classes = shape[2] - 5; //eg: 85 -5

        std::vector<Detection> detections;
        for (int i = 0; i < num_boxes; i++) {
            //每个框都是一个浮点数组，大小为85，这一句通过浮点指针来遍历框
            float* row = output_data + i * shape[2];
            /*通过置信度来过滤垃圾框*/
            float obj_conf = row[4];
            if (obj_conf < 0.5) continue;
            /*通过类别得分来寻找类别*/
            float max_class_score = 0;
            int class_id = -1;

            for (int j = 0; j < num_classes; j++) 
            {
                if (row[5 + j] > max_class_score) 
                {
                    max_class_score = row[5 + j];
                    class_id = j;
                }
            }
            /*最终置信度等于基本置信度 * 类别得分*/
            float final_conf = obj_conf * max_class_score;
            if (final_conf < 0.5) continue;
            
            /*找出合格框的坐标信息*/
            float x = row[0];
            float y = row[1];
            float w = row[2];
            float h = row[3];
            detections.push_back({class_id, final_conf, x, y, w, h});
        }
        // std::cout << "Total detections: " << detections.size() << std::endl;


        /*🔥 11. NMS*/
        //排序
        std::sort(detections.begin(), detections.end(),
            [](const Detection& a, const Detection& b) {
                return a.conf > b.conf;
            });
        //NMS主逻辑
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
        // std::cout << "After NMS: " << nms_result.size() << std::endl;

        float scale_x = frame.cols / 320.0;
        float scale_y = frame.rows / 320.0;
        
        //🔥 12. 画框
        for (auto& det : nms_result) {
        float x1 = det.x - det.w / 2;
        float y1 = det.y - det.h / 2;
        float x2 = det.x + det.w / 2;
        float y2 = det.y + det.h / 2;
        
        x1 = std::max(0, (int)(x1 * scale_x));
        x2 = std::min(frame.cols - 1, (int)(x2 * scale_x));
        y1 = std::max(5, (int)(y1 * scale_y));
        y2 = std::min(frame.rows - 1, (int)(y2 * scale_y));

        cv::rectangle(frame,
                    cv::Point(x1, y1),
                    cv::Point(x2, y2),
                    cv::Scalar(0, 255, 0),
                    2);

        // 类别 + 置信度
        std::string label = std::to_string(det.class_id) +
                            " " + std::to_string(det.conf);

        cv::putText(frame, label,
                    cv::Point(x1, y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 255, 0),
                    1);
        }


        // 放入第二个队列
        {
            std::unique_lock<std::mutex> lock2(mtx2);
            
            while (process_queue.size() > 1) {
                process_queue.pop();
            }
            process_queue.push(frame.clone());
            std::cout << "process_queue size: " << process_queue.size() << std::endl;
        }
        cond2.notify_one();
    }
}
// ==================== 主函数 ====================
int main() {
    std::thread t1(capture_thread);
    std::thread t2(display_thread);
    std::thread t3(process_thread);

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