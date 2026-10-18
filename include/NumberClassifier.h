#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// =====================================================================
// NumberClassifier —— 数字识别（作业要求：传统方案结合数字识别网络）
//
// 作业提供的数字识别网络为 onnx 交换格式，这里用 OpenCV DNN 模块
// 直接加载推理：传统视觉负责框出装甲板（灯条配对），把候选四边形
// 透视矫正后送入数字网络，得到装甲板上的数字类别。
//
// 若模型路径无效或加载失败，classify() 返回 '?'，不影响主流程。
// =====================================================================
class NumberClassifier {
public:
    struct Result {
        char label = '?';       // 识别出的数字字符（'1'~'5' 等），失败为 '?'
        float confidence = 0.f; // softmax 置信度
    };

    NumberClassifier() = default;

    /**
     * @param onnx_path 数字识别网络 onnx 路径
     * @param input_w/input_h 网络输入尺寸。默认 28x20 对应
     *        rm_auto_aim 的 mlp.onnx（onnx 输入形状 [1,20,28,1]：
     *        高 20、宽 28），换其它模型时需按实际形状调整
     */
    explicit NumberClassifier(const std::string& onnx_path,
                              int input_w = 28, int input_h = 20);

    bool loaded() const { return loaded_; }

    /**
     * 对装甲板区域做数字识别
     * @param image  原始 BGR 图
     * @param quad   装甲板四角点（左上/左下/右下/右上）
     */
    Result classify(const cv::Mat& image, const std::vector<cv::Point2f>& quad);

    /**
     * 透视矫正装甲板区域（也用于结果展示/调试输出）
     */
    cv::Mat cropArmor(const cv::Mat& image, const std::vector<cv::Point2f>& quad) const;

private:
    int w_ = 28, h_ = 20;
    bool loaded_ = false;
    cv::dnn::Net net_;              // OpenCV DNN 加载的 onnx 网络
    std::vector<std::string> classes_ = {"1", "2", "3", "4", "5"};
};
