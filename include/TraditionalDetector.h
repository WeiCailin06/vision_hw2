#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "DetectorCommon.h"

// =====================================================================
// TraditionalDetector —— 传统视觉装甲板检测器（OpenCV 手写，不依赖神经网络）
//
// 流程：
//   1. 用"蓝色显著度" B - max(G,R) 得到灯条响应图（代替灰度阈值：
//      灰度公式里蓝色权重只有 0.114，蓝灯条会被压暗而与背景混在一起）
//   2. 多级阈值从高到低尝试，形态学闭运算后 findContours
//   3. 从轮廓中筛选"灯条"（细长区块，PCA 求主轴方向，投影极值取上下端点）
//   4. 按 平行性 / 长度接近 / 间距比例 / 连线水平度 等几何约束配对灯条
//   5. 由左右灯条的上下端点构成装甲板四角点
//
// 角点顺序与神经网络方案完全一致：左上 / 左下 / 右下 / 右上
// 因此可直接复用同一套 PnPSolver 与物理尺寸模型（armor_width/height）
// =====================================================================
class TraditionalDetector {
public:
    struct Params {
        float min_area = 3.f;        // 灯条最小轮廓面积 (px^2)
        float min_ratio = 1.8f;      // 灯条最小长宽比
        float min_length = 8.f;      // 灯条最小长度 (px)
        float max_tilt = 0.70f;      // 单灯条允许的最大倾斜角 (rad, 约 34°)
        float max_tilt_diff = 0.25f; // 两灯条允许的最大倾斜差 (rad, 约 14°)
        float min_len_ratio = 0.55f;  // 两灯条长度比下限
        // 同一装甲板上 灯条中心距/灯条长 的理论值 = 13.5/5.5 ≈ 2.45，
        // 且该比值在同一平面内不随距离/角度变化。上限收紧到 3.0：
        // 跨装甲板的假配对（把不相邻的两根灯条接到一起）比值通常 >3
        float min_dist_ratio = 1.15f;
        float max_dist_ratio = 3.0f;
        float max_link_angle = 0.35f;// 灯条中心连线与水平方向最大夹角 (rad, 约 20°)
        // 语义约束：装甲板两灯条之间必贴数字贴纸（灰白色像素），
        // 候选四边形内部灰白像素占比下限。可过滤跨板误配/杂散反光假配对
        float min_inner_white = 0.05f;
        // 亚像素端点精修开关：分位点端点会把灯条略微缩短，而 PnP 物体模型
        // 假设灯条占满装甲板全高，缩短会带来系统性 Z 偏小，故默认关闭
        bool refine_endpoints = false;
    };

    TraditionalDetector() = default;
    explicit TraditionalDetector(const Params& p) : params_(p) {}

    /**
     * @param image      输入图像（BGR）
     * @param out_points 输出 4 个角点，顺序：左上/左下/右下/右上
     * @param select     多装甲板时的目标选择策略
     * @return           检测成功返回 true；同时输出调试图 debug_points_trad.jpg
     */
    bool detect(const cv::Mat& image, std::vector<cv::Point2f>& out_points,
                armor::SelectMode select = armor::SelectMode::LEFT);

    /**
     * 检出图中全部候选装甲板（供上层用重投影误差等准则自行挑选）
     * @return 至少找到一个候选返回 true
     */
    bool detectAll(const cv::Mat& image, std::vector<std::vector<cv::Point2f>>& all_points);

private:
    Params params_;
};