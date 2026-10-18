#include "NumberClassifier.h"
#include <fstream>
#include <iostream>

NumberClassifier::NumberClassifier(const std::string& onnx_path, int input_w, int input_h)
    : w_(input_w), h_(input_h) {
    try {
        net_ = cv::dnn::readNetFromONNX(onnx_path);
        loaded_ = true;
        std::cout << "✓ [NumberClassifier] 数字识别网络已加载: " << onnx_path
                  << " (输入 " << h_ << "x" << w_ << ")" << std::endl;
        // 若同目录存在 label.txt，按其读取类别表
        std::string dir = onnx_path.substr(0, onnx_path.find_last_of("/\\") + 1);
        std::ifstream fin(dir + "label.txt");
        if (fin.is_open()) {
            std::vector<std::string> labels;
            std::string line;
            while (std::getline(fin, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (!line.empty()) labels.push_back(line);
            }
            if (!labels.empty()) classes_ = labels;
        }
        std::cout << "  类别表: ";
        for (const auto& c : classes_) std::cout << c << " ";
        std::cout << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "⚠ [NumberClassifier] 数字识别网络加载失败（" << e.what()
                  << "），数字显示为 '?'" << std::endl;
    }
}

cv::Mat NumberClassifier::cropArmor(const cv::Mat& image,
                                    const std::vector<cv::Point2f>& quad) const {
    if (quad.size() != 4 || image.empty()) return {};
    // 透视矫正：把任意姿态的装甲板拉成正视图（左上/右上/右下/左下）
    std::vector<cv::Point2f> dst = {
        {0.f, 0.f}, {static_cast<float>(w_), 0.f},
        {static_cast<float>(w_), static_cast<float>(h_)}, {0.f, static_cast<float>(h_)}
    };
    // 输入角点顺序是 左上/左下/右下/右上，转成 左上/右上/右下/左下
    std::vector<cv::Point2f> src = { quad[0], quad[3], quad[2], quad[1] };
    cv::Mat M = cv::getPerspectiveTransform(src, dst);
    cv::Mat out;
    cv::warpPerspective(image, out, M, cv::Size(w_, h_));
    return out;
}

NumberClassifier::Result NumberClassifier::classify(const cv::Mat& image,
                                                    const std::vector<cv::Point2f>& quad) {
    Result r;
    if (!loaded_) return r;
    cv::Mat crop = cropArmor(image, quad);
    if (crop.empty()) return r;

    // 预处理：灰度 → OTSU 二值化（突出白色数字）→ 归一化
    cv::Mat gray;
    cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat blob = cv::dnn::blobFromImage(gray, 1.0 / 255.0);
    net_.setInput(blob);
    cv::Mat prob = net_.forward();
    if (prob.empty()) return r;

    // 兼容 (1,N)/(N,) 输出形状，softmax 后取最大
    cv::Mat p = prob.reshape(1, 1);
    if (p.depth() != CV_32F) p.convertTo(p, CV_32F);
    double maxv = 0;
    cv::minMaxLoc(p, nullptr, &maxv);
    cv::exp(p - maxv, p);
    p /= cv::sum(p)[0];

    int best = 0;
    double bestp = 0;
    for (int i = 0; i < p.cols && i < static_cast<int>(classes_.size()); ++i) {
        double v = p.at<float>(0, i);
        if (v > bestp) { bestp = v; best = i; }
    }
    r.label = classes_[best][0];
    r.confidence = static_cast<float>(bestp);
    return r;
}
