#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <string>
#include "DetectorCommon.h"
#include "NNDetector.hpp"

// =====================================================================
// NeuralDetector —— 神经网络装甲板检测器（深大开源库封装）
// 输入一帧图像，输出被选中装甲板的 4 个灯条角点（像素坐标）
// =====================================================================
class NeuralDetector {
public:
    NeuralDetector(const std::string& config_path, const std::string& model_dir);

    /**
     * @param image        输入图像（BGR）
     * @param out_points   输出 4 个角点，顺序：左上/左下/右下/右上
     * @param select       多装甲板时的目标选择策略
     * @return             检测成功返回 true
     */
    bool detect(const cv::Mat& image, std::vector<cv::Point2f>& out_points,
                armor::SelectMode select = armor::SelectMode::LEFT);

    /**
     * 检出图中全部候选装甲板（供上层用重投影误差等准则自行挑选）
     * @return 至少找到一个候选返回 true
     */
    bool detectAll(const cv::Mat& image, std::vector<std::vector<cv::Point2f>>& all_points);

private:
    std::unique_ptr<ArmorModel> armor_model_;
    // 缓存上一帧的全部候选，detect() 内部复用，避免重复推理
    std::vector<std::vector<cv::Point2f>> last_candidates_;
};
