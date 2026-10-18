#include "NeuralDetector.h"
#include <cmath>
#include <iostream>

namespace {
// 鞋带公式求多边形面积，用于 LARGEST 策略
float polygonArea(const std::vector<cv::Point2f>& pts) {
    float a = 0;
    for (size_t i = 0; i < pts.size(); ++i) {
        const auto& p = pts[i];
        const auto& q = pts[(i + 1) % pts.size()];
        a += p.x * q.y - q.x * p.y;
    }
    return std::abs(a) * 0.5f;
}
} // namespace

NeuralDetector::NeuralDetector(const std::string& config_path, const std::string& model_dir) {
    JsonConfig armor_config{config_path, "armor_v8", model_dir};
    armor_model_ = std::make_unique<ArmorModel>(armor_config);
}

bool NeuralDetector::detectAll(const cv::Mat& image,
                               std::vector<std::vector<cv::Point2f>>& all_points) {
    all_points.clear();
    last_candidates_.clear();
    if (image.empty()) return false;

    auto results = armor_model_->netProcess(image, 2);
    if (results.empty()) return false;

    for (const auto& armor : results) {
        if (armor.points.size() < 4) continue;   // 角点不全的框直接丢弃
        std::vector<cv::Point2f> pts;
        pts.reserve(armor.points.size());
        for (const auto& pt : armor.points) {
            pts.push_back(cv::Point2f(static_cast<float>(pt.x), static_cast<float>(pt.y)));
        }
        all_points.push_back(std::move(pts));
    }

    last_candidates_ = all_points;
    return !all_points.empty();
}

bool NeuralDetector::detect(const cv::Mat& image, std::vector<cv::Point2f>& out_points,
                            armor::SelectMode select) {
    out_points.clear();
    if (last_candidates_.empty() && !detectAll(image, last_candidates_)) return false;

    // ---- 按策略从所有检测结果中选择目标装甲板 ----
    const std::vector<cv::Point2f>* target = nullptr;
    float best_score = 0;
    for (const auto& cand : last_candidates_) {
        float center_x = 0;
        for (const auto& pt : cand) center_x += pt.x;
        center_x /= static_cast<float>(cand.size());

        float area = polygonArea(cand);

        bool better = false;
        switch (select) {
            case armor::SelectMode::LEFT:    better = !target || center_x < best_score; break;
            case armor::SelectMode::RIGHT:   better = !target || center_x > best_score; break;
            case armor::SelectMode::LARGEST: better = !target || area > best_score; break;
        }
        if (better) {
            target = &cand;
            best_score = (select == armor::SelectMode::LARGEST) ? area : center_x;
        }
    }
    if (!target) return false;

    out_points = *target;

    // ---- 调试图：标注角点与顺序，便于人工核对四点顺序是否正确 ----
    cv::Mat debug_img = image.clone();
    for (const auto& cand : last_candidates_) {   // 未选中的候选画灰框
        if (&cand == target) continue;
        for (size_t i = 0; i < cand.size(); ++i)
            cv::line(debug_img, cand[i], cand[(i + 1) % cand.size()], cv::Scalar(140, 140, 140), 1);
    }
    for (size_t i = 0; i < out_points.size(); ++i) {
        cv::circle(debug_img, out_points[i], 5, cv::Scalar(0, 0, 255), -1);
        cv::putText(debug_img, std::to_string(i), out_points[i],
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    }
    for (size_t i = 0; i < out_points.size(); ++i) {   // 依次连线，直观展示点序
        cv::line(debug_img, out_points[i], out_points[(i + 1) % out_points.size()],
                 cv::Scalar(0, 255, 255), 2);
    }
    cv::imwrite("debug_points_nn.jpg", debug_img);

    return out_points.size() == 4;
}