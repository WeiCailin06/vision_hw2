#include "TraditionalDetector.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace {

// 单根灯条的几何描述
struct LightBar {
    cv::Point2f top;      // 灯条靠上的一端
    cv::Point2f bottom;   // 灯条靠下的一端
    cv::Point2f center;
    float length = 0;     // 灯条长度
    float width = 0;      // 灯条宽度
    float angle = 0;      // 主轴与竖直方向夹角 (rad)
    float area = 0;
};

// 用 PCA 求灯条主轴，比 minAreaRect 更稳（对细长条噪声不敏感）
// diff_img：蓝色显著度图，用于亚像素端点 refinement
bool buildLightBar(const std::vector<cv::Point>& contour,
                   const TraditionalDetector::Params& p, LightBar& out,
                   const cv::Mat* diff_img = nullptr) {
    if (contour.size() < 4) return false;

    double area = cv::contourArea(contour);
    if (area < p.min_area) return false;

    cv::RotatedRect rr = cv::minAreaRect(contour);
    float long_side = std::max(rr.size.width, rr.size.height);
    float short_side = std::min(rr.size.width, rr.size.height);
    if (short_side < 1.f) return false;
    if (long_side < p.min_length) return false;
    if (long_side / short_side < p.min_ratio) return false;

    // 点集 PCA 求主轴方向（避免不同 OpenCV 版本 minAreaRect 角度约定差异）
    cv::Mat pts(static_cast<int>(contour.size()), 2, CV_32F);
    for (size_t i = 0; i < contour.size(); ++i) {
        pts.at<float>(static_cast<int>(i), 0) = static_cast<float>(contour[i].x);
        pts.at<float>(static_cast<int>(i), 1) = static_cast<float>(contour[i].y);
    }
    cv::PCA pca(pts, cv::Mat(), cv::PCA::DATA_AS_ROW);
    float vx = pca.eigenvectors.at<float>(0, 0);
    float vy = pca.eigenvectors.at<float>(0, 1);
    if (std::abs(vy) < 1e-6f) return false;   // 纯水平，明显不是灯条

    float angle = std::atan2(vx, vy);         // 与竖直方向夹角
    if (angle > CV_PI / 2) angle -= static_cast<float>(CV_PI);
    if (angle < -CV_PI / 2) angle += static_cast<float>(CV_PI);
    if (std::abs(angle) > p.max_tilt) return false;

    // 沿主轴方向取投影极值点作为灯条上下端点
    // 注意：cv::PCA::mean 是 1x2 的 cv::Mat，需要显式取出坐标
    cv::Point2f c(static_cast<float>(pca.mean.at<float>(0, 0)),
                  static_cast<float>(pca.mean.at<float>(0, 1)));
    float minT = 1e18f, maxT = -1e18f;
    for (const auto& q : contour) {
        float dx = static_cast<float>(q.x) - c.x;
        float dy = static_cast<float>(q.y) - c.y;
        float t = dx * vx + dy * vy;
        minT = std::min(minT, t);
        maxT = std::max(maxT, t);
    }

    // ---- 亚像素端点精修 ----
    // 二值轮廓的极值点受发光晕/阈值影响很大（端点误差 1~2px 在远距离
    // 会被 PnP 放大数厘米）。改用显著度加权投影：在轮廓外扩邻域内
    // 把显著度沿主轴累积，取能量 5%~95% 分位点作为灯条端点。
    cv::Point2f top = c + cv::Point2f(vx, vy) * minT;
    cv::Point2f bottom = c + cv::Point2f(vx, vy) * maxT;
    float refined_len = maxT - minT;
    if (p.refine_endpoints && diff_img && !diff_img->empty()) {
        int R = static_cast<int>(short_side) + 4;
        int x0 = std::max(0, static_cast<int>(c.x) - R - static_cast<int>(refined_len));
        int y0 = std::max(0, static_cast<int>(c.y) - R - static_cast<int>(refined_len));
        int x1 = std::min(diff_img->cols - 1, static_cast<int>(c.x) + R + static_cast<int>(refined_len));
        int y1 = std::min(diff_img->rows - 1, static_cast<int>(c.y) + R + static_cast<int>(refined_len));
        if (x1 > x0 && y1 > y0) {
            // 沿主轴把显著度累积到直方图（bin = 1px）
            int nbins = static_cast<int>(maxT - minT) + 3;
            std::vector<double> hist(nbins, 0.0);
            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    float w = static_cast<float>(diff_img->at<uchar>(y, x));
                    if (w < 20) continue;
                    float dx = static_cast<float>(x) - c.x;
                    float dy = static_cast<float>(y) - c.y;
                    // 只取主轴附近（垂直主轴距离 < 半宽 + 2px）的像素
                    float perp = std::abs(dx * vy - dy * vx);
                    if (perp > short_side * 0.5f + 2.f) continue;
                    float t = dx * vx + dy * vy;
                    int bin = static_cast<int>(t - minT) + 1;
                    if (bin >= 0 && bin < nbins) hist[bin] += w;
                }
            }
            double total = 0;
            for (double v : hist) total += v;
            if (total > 1e-6) {
                const double lo_q = 0.02, hi_q = 0.98;
                double acc = 0;
                float t_lo = minT, t_hi = maxT;
                bool got_lo = false;
                for (int i = 0; i < nbins; ++i) {
                    acc += hist[i];
                    float t = minT + (i - 1);
                    if (!got_lo && acc >= total * lo_q) { t_lo = t; got_lo = true; }
                    if (acc >= total * hi_q) { t_hi = t; break; }
                }
                if (t_hi > t_lo) {
                    top = c + cv::Point2f(vx, vy) * t_lo;
                    bottom = c + cv::Point2f(vx, vy) * t_hi;
                    refined_len = t_hi - t_lo;
                }
            }
        }
    }

    out.center = c;
    out.top = (top.y <= bottom.y) ? top : bottom;
    out.bottom = (top.y <= bottom.y) ? bottom : top;
    out.length = refined_len;
    out.width = (out.length > 1e-3f) ? static_cast<float>(area) / out.length : short_side;
    out.angle = angle;
    out.area = static_cast<float>(area);
    return out.length > 1e-3f;
}

// 判断两灯条能否配对成装甲板
bool canPair(const LightBar& a, const LightBar& b, const TraditionalDetector::Params& p) {
    if (a.center.x > b.center.x) return false;   // 约定 a 在左

    // 1) 倾斜角接近（平行）
    if (std::abs(a.angle - b.angle) > p.max_tilt_diff) return false;

    // 2) 长度接近
    float len_ratio = std::min(a.length, b.length) / std::max(a.length, b.length);
    if (len_ratio < p.min_len_ratio) return false;

    // 3) 间距/灯条长约束：
    //    同一块装甲板上，两灯条中心距(w) 与 灯条长(h) 之比在同一平面内
    //    透视不变，理论值 = armor_width/armor_height ≈ 13.5/5.5 ≈ 2.45，
    //    灯条被遮挡变短时比值会偏大，因此给 [1.3, 3.4] 的容差窗。
    float dist = cv::norm(a.center - b.center);
    float avg_len = (a.length + b.length) * 0.5f;
    float ratio = dist / std::max(avg_len, 1.f);
    if (ratio < p.min_dist_ratio || ratio > p.max_dist_ratio) return false;

    // 4) 两灯条中心连线应接近水平（允许一定透视倾斜）
    float dx = b.center.x - a.center.x;
    float dy = b.center.y - a.center.y;
    float link_angle = std::atan2(std::abs(dy), std::abs(dx));
    if (link_angle > p.max_link_angle) return false;

    // 5) 垂直方向错位不能超过灯条长度
    if (std::abs(dy) > avg_len) return false;

    return true;
}

// 由左右灯条生成装甲板四角点：左上 / 左下 / 右下 / 右上
std::vector<cv::Point2f> makeCorners(const LightBar& l, const LightBar& r) {
    return { l.top, l.bottom, r.bottom, r.top };
}

// 装甲板语义校验：两灯条之间必贴数字贴纸（灰白色，R/G/B 同时高亮）。
// 跨装甲板误配 / 灯条与杂散反光的假配对，四边形内部几乎不含白色像素。
float innerWhiteRatio(const cv::Mat& image, const std::vector<cv::Point2f>& quad) {
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    std::vector<cv::Point> poly;
    for (const auto& p : quad) poly.push_back(cv::Point(cvRound(p.x), cvRound(p.y)));
    cv::fillPoly(mask, poly, 255);

    std::vector<cv::Mat> ch;
    cv::split(image, ch);
    cv::Mat mn;
    cv::min(cv::min(ch[0], ch[1]), ch[2], mn);
    cv::Mat white;
    cv::threshold(mn, white, 100, 255, cv::THRESH_BINARY);

    int inside = cv::countNonZero(mask);
    if (inside == 0) return 0;
    cv::Mat both;
    cv::bitwise_and(mask, white, both);
    return static_cast<float>(cv::countNonZero(both)) / inside;
}

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

bool TraditionalDetector::detect(const cv::Mat& image, std::vector<cv::Point2f>& out_points,
                                 armor::SelectMode select) {
    std::vector<std::vector<cv::Point2f>> all;
    if (!detectAll(image, all) || all.empty()) { out_points.clear(); return false; }

    // ---- 按策略选择目标 ----
    const std::vector<cv::Point2f>* best = nullptr;
    float score = 0;
    for (const auto& c : all) {
        float cx = (c[0].x + c[1].x + c[2].x + c[3].x) / 4.f;
        float a = polygonArea(c);
        bool better = false;
        switch (select) {
            case armor::SelectMode::LEFT:    better = !best || cx < score; break;
            case armor::SelectMode::RIGHT:   better = !best || cx > score; break;
            case armor::SelectMode::LARGEST: better = !best || a > score; break;
        }
        if (better) { best = &c; score = (select == armor::SelectMode::LARGEST) ? a : cx; }
    }
    if (!best) return false;
    out_points = *best;
    return true;
}

bool TraditionalDetector::detectAll(const cv::Mat& image,
                                    std::vector<std::vector<cv::Point2f>>& all_points) {
    all_points.clear();
    if (image.empty() || image.channels() != 3) return false;

    // =================================================================
    // 灯条分割：使用"颜色显著度"而不是灰度。
    //   蓝路：B - max(G, R) —— 蓝色灯条（灰度里蓝色权重只有 0.114，
    //          会被压暗到与背景难分；显著度图上灯条 >50、背景 ≈0）
    //   暖路：R - max(G, B) —— 橙色/红色灯条（实拍视频中灯条常呈橙色）
    // 两路取最大值，蓝、红、橙灯条统一处理；白色高光两路都 ≈0。
    // =================================================================
    std::vector<cv::Mat> ch;
    cv::split(image, ch);
    cv::Mat gr, rb, blue_Resp, warm_Resp;
    cv::max(ch[1], ch[2], gr);           // max(G, R)
    cv::subtract(ch[0], gr, blue_Resp);  // B - max(G, R)
    cv::max(ch[1], ch[0], rb);           // max(G, B)
    cv::subtract(ch[2], rb, warm_Resp);  // R - max(G, B)
    cv::Mat diff = cv::max(blue_Resp, warm_Resp);
    diff.setTo(0, diff < 0);

    // 多级阈值：从高到低，高阈值下灯条干净，低阈值用于远距离暗淡灯条
    const std::vector<int> thresholds = {60, 50, 40, 30, 25};

    for (int th : thresholds) {
        cv::Mat bin;
        cv::threshold(diff, bin, th, 255, cv::THRESH_BINARY);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<LightBar> bars;
        for (const auto& c : contours) {
            LightBar lb;
            if (buildLightBar(c, params_, lb, &diff)) bars.push_back(lb);
        }
        if (bars.size() < 2) continue;

        // ---- 枚举所有可配对的灯条组合 ----
        struct Candidate { std::vector<cv::Point2f> pts; float area; float center_x; };
        std::vector<Candidate> cands;
        for (size_t i = 0; i < bars.size(); ++i) {
            for (size_t j = 0; j < bars.size(); ++j) {
                if (i == j) continue;
                if (!canPair(bars[i], bars[j], params_)) continue;
                auto pts = makeCorners(bars[i], bars[j]);
                float a = polygonArea(pts);
                if (a < 4.f) continue;   // 面积过小视为误检
                // 语义校验：灯条之间必须有数字贴纸（灰白像素）
                if (innerWhiteRatio(image, pts) < params_.min_inner_white) continue;
                float cx = (pts[0].x + pts[1].x + pts[2].x + pts[3].x) / 4.f;
                cands.push_back({ pts, a, cx });
            }
        }
        if (cands.empty()) continue;

        // ---- 本阈值下把全部候选装进结果集 ----
        for (const auto& c : cands) all_points.push_back(c.pts);

        // ---- 调试图 ----
        cv::Mat debug_img = image.clone();
        for (const auto& c : all_points) {   // 候选（灰）
            for (size_t k = 0; k < c.size(); ++k)
                cv::line(debug_img, c[k], c[(k + 1) % c.size()], cv::Scalar(120, 120, 120), 1);
        }
        cv::imwrite("debug_points_trad.jpg", debug_img);
        cv::imwrite("debug_binary.jpg", bin);

        std::cout << "✓ [TraditionalDetector] 阈值 " << th << "：灯条 " << bars.size()
                  << " 根，候选装甲板 " << cands.size() << " 个（共积累 "
                  << all_points.size() << " 个候选）" << std::endl;
        // 继续尝试更低的阈值：远距离/倾斜的装甲板灯条更暗淡，
        // 高阈值可能只看到一部分；多余候选由上层按重投影误差淘汰
    }

    std::cerr << "⚠ [TraditionalDetector] 所有阈值下均未配对出装甲板" << std::endl;
    return false;
}