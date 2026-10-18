#include "PnPSolver.h"
#include <iostream>

PnPSolver::PnPSolver(const cv::Mat& camera_matrix, const cv::Mat& distort_coeffs,
                     float armor_w, float armor_h) {
    camera_matrix_ = camera_matrix.clone();
    distort_coeffs_ = distort_coeffs.clone();
    width_ = armor_w;
    height_ = armor_h;
}

bool PnPSolver::solve(const std::vector<cv::Point2f>& image_points,
                      cv::Vec3d& tvec, cv::Vec3d& rvec,
                      int armor_id, double* reproj_error) {
    if (image_points.size() != 4) {
        std::cerr << "❌ [PnPSolver] 需要 4 个图像点，实际收到 " << image_points.size() << " 个" << std::endl;
        return false;
    }
    // ---- 防御性检查：物理尺寸非法会让物体点全部退化为原点，
    //      这正是之前 "DLT algorithm needs at least 6 points" 崩溃的根源 ----
    if (width_ <= 0.f || height_ <= 0.f) {
        std::cerr << "❌ [PnPSolver] 装甲板物理尺寸非法 (w=" << width_
                  << ", h=" << height_ << ")，请检查 camera_params.yml" << std::endl;
        return false;
    }

    float w = width_;
    float h = height_;
    std::vector<cv::Point3f> object_points;

    // 严格遵循作业 .md 文件里的定义（车体坐标系，单位 cm）
    if (armor_id == 1 || armor_id == 3) {
        // 1/3 号板：宽沿 X，高沿 Z，法线沿 Y (Y=0)
        object_points = {
            {-w/2, 0, -h/2}, // 左上
            {-w/2, 0,  h/2}, // 左下
            { w/2, 0,  h/2}, // 右下
            { w/2, 0, -h/2}  // 右上
        };
    } else if (armor_id == 2 || armor_id == 4) {
        // 2/4 号板：宽沿 Y，高沿 Z，法线沿 X (X=0)
        object_points = {
            {0, -w/2, -h/2}, // 左上
            {0, -w/2,  h/2}, // 左下
            {0,  w/2,  h/2}, // 右下
            {0,  w/2, -h/2}  // 右上
        };
    } else {
        // 默认：装甲板平面取 Z=0（面向相机）
        object_points = {
            {-w/2, -h/2, 0.0f}, {-w/2,  h/2, 0.0f},
            { w/2,  h/2, 0.0f}, { w/2, -h/2, 0.0f}
        };
    }

    // ---- 使用 IPPE（Infinitesimal Plane-based Pose Estimation）----
    // 专为"共面 4 点"设计：不会退化到需要 6 点的 DLT，且天然给出两个
    // 平面二义性解，这里用平均重投影误差挑选最优解。
    std::vector<cv::Mat> rvecs, tvecs;
    int n = 0;
    try {
        n = cv::solvePnPGeneric(object_points, image_points,
                                camera_matrix_, distort_coeffs_,
                                rvecs, tvecs, false, cv::SOLVEPNP_IPPE);
    } catch (const cv::Exception& e) {
        std::cerr << "❌ [PnPSolver] solvePnP 异常: " << e.what() << std::endl;
        return false;
    }
    if (n <= 0) {
        std::cerr << "❌ [PnPSolver] solvePnP 未得到有效解" << std::endl;
        return false;
    }

    // 取平均重投影误差最小的解
    int best = 0;
    double best_err = 1e18;
    for (int i = 0; i < n; ++i) {
        std::vector<cv::Point2f> proj;
        cv::projectPoints(object_points, rvecs[i], tvecs[i],
                          camera_matrix_, distort_coeffs_, proj);
        double e = 0;
        for (size_t k = 0; k < proj.size(); ++k)
            e += cv::norm(proj[k] - image_points[k]);
        e /= static_cast<double>(proj.size());
        if (e < best_err) { best_err = e; best = i; }
    }

    rvec = cv::Vec3d(rvecs[best]);
    tvec = cv::Vec3d(tvecs[best]);
    if (reproj_error) *reproj_error = best_err;

    return tvec[2] > 0;
}