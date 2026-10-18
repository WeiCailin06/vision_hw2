#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "DetectorCommon.h"

// =====================================================================
// PnPSolver —— 装甲板位置解算（2D 四角点 → 3D 位姿）
//
// 物体点（车体坐标系，单位 cm）按装甲板编号区分：
//   1/3 号板：装甲板平面在车体 X-Z 平面，法线沿车体 Y 轴
//   2/4 号板：装甲板平面在车体 Y-Z 平面，法线沿车体 X 轴
// 解算输出：
//   tvec —— 装甲板中心在相机坐标系下的位置 (cm)
//   rvec —— 车体坐标系相对相机系的姿态（ Rodrigues 向量）
// =====================================================================
class PnPSolver {
public:
    PnPSolver(const cv::Mat& camera_matrix, const cv::Mat& distort_coeffs,
              float armor_w, float armor_h);

    /**
     * @param image_points  4 个图像角点，顺序：左上/左下/右下/右上
     * @param tvec, rvec    输出位姿
     * @param armor_id      装甲板编号（决定物体点所在平面）
     * @param reproj_error  可选输出：最优解的平均重投影误差 (px)
     */
    bool solve(const std::vector<cv::Point2f>& image_points,
               cv::Vec3d& tvec, cv::Vec3d& rvec,
               int armor_id = 1, double* reproj_error = nullptr);

private:
    cv::Mat camera_matrix_;
    cv::Mat distort_coeffs_;
    float width_;
    float height_;
};