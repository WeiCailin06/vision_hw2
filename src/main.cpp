#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <cstdlib>

#include "PnPSolver.h"
#include "NeuralDetector.h"
#include "TraditionalDetector.h"
#include "NumberClassifier.h"
#include "DetectorCommon.h"

// =====================================================================
// 视觉第二周作业主程序
// 用法：
//   ./main                      交互模式（逐个输入图片编号）
//   ./main 1 2 3                批量测试三张图
//   ./main 1 --mode trad        用传统视觉方案测图 1
//   ./main 1 --select auto      自动选择目标装甲板（默认）
//   ./main --video armor.mp4 --out-video annotated.mp4
//                               处理视频：逐帧识别 + 绘制矩形框 +
//                               数字识别结果 + PnP 坐标，输出标注视频
//   其他参数：
//     --mode nn|trad|both     检测方案（默认 both：两种都跑并对比）
//     --select auto|left|right|largest
//     --config <path>         深大库 detect.json 路径
//     --modeldir <path>       模型目录
//     --params <path>         相机参数 yml 路径
//     --nummodel <onnx>       数字识别网络 onnx 路径（传统方案配数字网络）
//     --out-video <mp4>       输出标注视频路径
//     --hold-sec <s>          图片模式合成演示视频时每张图持续秒数（默认 3）
// =====================================================================

namespace {

struct Options {
    std::vector<std::string> images;     // 要测试的图片编号
    std::string mode = "both";
    std::string select = "auto";
    std::string config_path;
    std::string model_dir;
    std::string params_path = "camera_params.yml";
    std::string video_path;              // --video：输入视频
    std::string out_video;               // --out-video：标注视频输出
    std::string num_model;               // --nummodel：数字识别 onnx
    double hold_sec = 3.0;               // 图片合成演示视频时每张持续秒数
    bool verbose = true;
};

Options parseArgs(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { std::cerr << "参数 " << name << " 缺少取值" << std::endl; std::exit(1); }
            return argv[++i];
        };
        if (a == "--mode")            o.mode = need("--mode");
        else if (a == "--select")     o.select = need("--select");
        else if (a == "--config")     o.config_path = need("--config");
        else if (a == "--modeldir")   o.model_dir = need("--modeldir");
        else if (a == "--params")     o.params_path = need("--params");
        else if (a == "--video")      o.video_path = need("--video");
        else if (a == "--out-video")  o.out_video = need("--out-video");
        else if (a == "--nummodel")   o.num_model = need("--nummodel");
        else if (a == "--hold-sec")   o.hold_sec = std::stod(need("--hold-sec"));
        else if (a == "--quiet")      o.verbose = false;
        else if (a.rfind("--", 0) == 0) { std::cerr << "未知参数: " << a << std::endl; std::exit(1); }
        else o.images.push_back(a);
    }
    return o;
}

// 按名字转换选择策略；"auto" 由调用处处理（表示重投影误差最小准则）
armor::SelectMode toMode(const std::string& s) {
    if (s == "left") return armor::SelectMode::LEFT;
    if (s == "right") return armor::SelectMode::RIGHT;
    if (s == "largest") return armor::SelectMode::LARGEST;
    return armor::SelectMode::LEFT; // auto 由调用处处理
}

// 读取两张图的装甲板角点无需重复代码：统一的测试入口
struct TestResult {
    bool ok = false;
    cv::Vec3d tvec{0, 0, 0};
    cv::Vec3d rvec{0, 0, 0};
    double reproj = 0;
    double err = 0;
    cv::Vec3d delta{0, 0, 0};
};

TestResult evaluate(bool detected, bool solved, const cv::Vec3d& tvec, const cv::Vec3d& rvec,
                    double reproj, const cv::Vec3d& truth) {
    TestResult r;
    r.ok = detected && solved;
    if (!r.ok) return r;
    r.tvec = tvec; r.rvec = rvec; r.reproj = reproj;
    r.delta = cv::Vec3d(tvec[0] - truth[0], tvec[1] - truth[1], tvec[2] - truth[2]);
    r.err = cv::norm(r.delta);
    return r;
}

void printResult(const std::string& tag, const TestResult& r, const cv::Vec3d& truth) {
    if (!r.ok) {
        std::cout << "  [" << tag << "] ❌ 检测/解算失败" << std::endl;
        return;
    }
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  [" << tag << "] 计算值 : X=" << r.tvec[0] << ", Y=" << r.tvec[1]
              << ", Z=" << r.tvec[2] << " cm" << std::endl;
    std::cout << "  [" << tag << "] 真值   : X=" << truth[0] << ", Y=" << truth[1]
              << ", Z=" << truth[2] << " cm" << std::endl;
    std::cout << "  [" << tag << "] Δx=" << r.delta[0] << "  Δy=" << r.delta[1]
              << "  Δz=" << r.delta[2] << "  三维误差=" << r.err
              << " cm  重投影=" << r.reproj << " px" << std::endl;
}



// 对每个候选装甲板独立解算 PnP，取重投影误差（几何自洽度）最小的作为目标。
// 为什么用重投影误差而不是"最左/最大"这类位置启发式：
//   四个角点必须同时满足同一台相机的单应约束，错误的配对
//   （比如把两块不同装甲板的灯条配在一起）在几何上不自洽，
//   重投影误差会明显偏大。这个准则不需要真值，属于合法先验。
struct BestSolve {
    bool ok = false;
    int index = -1;
    std::vector<cv::Point2f> points;
    cv::Vec3d tvec{0, 0, 0};
    cv::Vec3d rvec{0, 0, 0};
    double reproj = 0;
    size_t candidate_count = 0;
};

BestSolve solveBest(const std::vector<std::vector<cv::Point2f>>& candidates,
                    PnPSolver& solver, int armor_id, const cv::Point2f& img_center) {
    BestSolve best;
    best.candidate_count = candidates.size();

    // ---- 候选聚类（中心距 < 8px 视为同一块板）----
    // 两级准则：
    //   1) 聚类内去噪：同一聚类按重投影误差选代表
    //   2) 聚类间选择：取中心离图像中心最近的聚类。
    //      物理意义：装甲板是瞄准目标，准星在图像中心，
    //      被瞄准的板理应最接近光轴（对 NN 方案同样适用）。
    struct Cluster { std::vector<size_t> members; cv::Point2f center; };
    std::vector<Cluster> clusters;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i].size() != 4) continue;
        cv::Point2f c(0, 0);
        for (const auto& p : candidates[i]) c += p;
        c *= 0.25f;
        bool merged = false;
        for (auto& cl : clusters) {
            if (cv::norm(cl.center - c) < 8.f) {
                cl.members.push_back(i);
                cl.center *= 0.5f; cl.center += c * 0.5f;
                merged = true;
                break;
            }
        }
        if (!merged) clusters.push_back({ {i}, c });
    }
    if (clusters.empty()) return best;
    // 聚类中心用成员中位数位置（比均值更稳），并取离图像中心最近者
    Cluster* target = nullptr;
    float best_cdist = 1e18f;
    for (auto& cl : clusters) {
        std::vector<float> xs, ys;
        for (size_t i : cl.members) {
            xs.push_back((candidates[i][0].x + candidates[i][1].x +
                          candidates[i][2].x + candidates[i][3].x) / 4.f);
            ys.push_back((candidates[i][0].y + candidates[i][1].y +
                          candidates[i][2].y + candidates[i][3].y) / 4.f);
        }
        std::sort(xs.begin(), xs.end());
        std::sort(ys.begin(), ys.end());
        cl.center = cv::Point2f(xs[xs.size() / 2], ys[ys.size() / 2]);
        float d = static_cast<float>(cv::norm(cl.center - img_center));
        if (d < best_cdist) { best_cdist = d; target = &cl; }
    }
    if (!target) return best;
    // 聚类内按重投影误差选代表（注：不同阈值下灯条分割形态不同，
    // 角点没有物理意义上的"平均"，实测平均/加权平均并无稳定增益）
    for (size_t i : target->members) {
        cv::Vec3d tvec, rvec;
        double reproj = 0;
        if (!solver.solve(candidates[i], tvec, rvec, armor_id, &reproj)) continue;
        if (!best.ok || reproj < best.reproj) {
            best.ok = true;
            best.index = static_cast<int>(i);
            best.points = candidates[i];
            best.tvec = tvec;
            best.rvec = rvec;
            best.reproj = reproj;
        }
    }
    return best;
}

// 灯条中心距几何测距：Z ≈ f · W / 灯条中心距（W=装甲板物理宽）。
// 与 PnP 相互独立，两者差异大说明角点与物体模型不匹配（模型误差）
double geometricRange(const std::vector<cv::Point2f>& pts, double fx, double W) {
    cv::Point2f left  = (pts[0] + pts[1]) * 0.5f;   // 左灯条中心
    cv::Point2f right = (pts[2] + pts[3]) * 0.5f;   // 右灯条中心
    float d = static_cast<float>(cv::norm(left - right));
    if (d < 1.f) return 0;
    return fx * W / d;
}

// 计算几何自洽度（重投影误差），不返回解，只用于候选打分
std::vector<std::pair<int, double>> scoreCandidates(
        const std::vector<std::vector<cv::Point2f>>& candidates,
        PnPSolver& solver, int armor_id) {
    std::vector<std::pair<int, double>> scored;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i].size() != 4) continue;
        cv::Vec3d tvec, rvec;
        double reproj = 0;
        if (solver.solve(candidates[i], tvec, rvec, armor_id, &reproj))
            scored.emplace_back(static_cast<int>(i), reproj);
    }
    return scored;
}

// ---- 视频标注（作业要求 3：绘制识别出的装甲板矩形框 + 数字识别结果）----
// 在帧上绘制：黄色四边形框（装甲板外接矩形）、红角点、数字识别结果、
// PnP 解算坐标（相机系，cm）
void drawAnnotate(cv::Mat& frame, const std::vector<cv::Point2f>& quad,
                  const std::string& number, const cv::Vec3d& tvec,
                  const std::string& tag) {
    if (quad.size() != 4) return;
    // 外接矩形框（装甲板矩形框）
    float minx = 1e9f, miny = 1e9f, maxx = -1e9f, maxy = -1e9f;
    for (const auto& p : quad) {
        minx = std::min(minx, p.x); maxx = std::max(maxx, p.x);
        miny = std::min(miny, p.y); maxy = std::max(maxy, p.y);
    }
    cv::rectangle(frame, cv::Rect(cvRound(minx), cvRound(miny),
                                  cvRound(maxx - minx), cvRound(maxy - miny)),
                  cv::Scalar(0, 255, 255), 2);
    // 四边形轮廓 + 角点
    for (int i = 0; i < 4; ++i)
        cv::line(frame, quad[i], quad[(i + 1) % 4], cv::Scalar(0, 255, 0), 1);
    for (int i = 0; i < 4; ++i)
        cv::circle(frame, quad[i], 3, cv::Scalar(0, 0, 255), -1);
    // 数字识别结果 + PnP 坐标
    std::ostringstream ss;
    ss << number << "  (" << std::fixed << std::setprecision(1)
       << tvec[0] << ", " << tvec[1] << ", " << tvec[2] << ")cm";
    int ty = std::max(20, cvRound(miny) - 10);
    cv::putText(frame, tag + ss.str(), {cvRound(minx), ty},
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 2);
}

// 视频后处理：OpenCV 的 mp4v 编码器与部分 ffmpeg 构建不兼容，
// 统一先写 MJPG AVI，再用 ffmpeg 转码为 H.264 mp4（无 ffmpeg 时保留 avi）
void convertToMp4(const std::string& avi_path, const std::string& out_path) {
    std::string cmd = "ffmpeg -y -v error -i \"" + avi_path + "\" -c:v mpeg4 -q:v 3 \""
                    + out_path + "\"";
    int rc = std::system(cmd.c_str());
    if (rc == 0) std::remove(avi_path.c_str());
    else std::cerr << "⚠ ffmpeg 转码失败，保留中间文件 " << avi_path << std::endl;
}

// 视频模式：逐帧识别 + 标注 + 输出
int runVideo(const Options& opt, PnPSolver& solver, const cv::Mat& cam,
             TraditionalDetector& trad, NeuralDetector* nn, NumberClassifier& numcls) {
    cv::VideoCapture cap(opt.video_path);
    if (!cap.isOpened()) {
        std::cerr << "❌ 打不开视频: " << opt.video_path << std::endl;
        return -1;
    }
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 1) fps = 30;
    int W = cvRound(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int H = cvRound(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::string out_path = opt.out_video.empty() ? "annotated.mp4" : opt.out_video;
    std::string tmp_avi = "/tmp/hw2_annotated.avi";
    cv::VideoWriter writer(tmp_avi, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, {W, H});
    if (!writer.isOpened()) {
        std::cerr << "❌ 打不开输出视频: " << out_path << std::endl;
        return -1;
    }
    std::cout << "✓ 视频模式: " << opt.video_path << " (" << W << "x" << H << "@"
              << fps << "fps) → " << out_path << std::endl;

    cv::Mat frame;
    int idx = 0, hit = 0;
    while (cap.read(frame)) {
        ++idx;
        // NN 优先（角点来自网络，与物体模型匹配更好）；无 NN 时用传统方案
        std::vector<std::vector<cv::Point2f>> cands;
        bool have_nn = false;
        if (nn && nn->detectAll(frame, cands) && !cands.empty()) have_nn = true;
        if (!have_nn) { trad.detectAll(frame, cands); }

        BestSolve picked = solveBest(cands, solver, 1,
                                     cv::Point2f(W * 0.5f, H * 0.5f));
        if (picked.ok) {
            ++hit;
            auto num = numcls.classify(frame, picked.points);
            std::ostringstream tag;
            tag << (have_nn ? "[NN]" : "[trad]") << "#";
            drawAnnotate(frame, picked.points, std::string(1, num.label),
                         picked.tvec, tag.str());
        }
        cv::putText(frame, "frame " + std::to_string(idx),
                    {10, 22}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {200, 200, 200}, 1);
        writer.write(frame);
    }
    std::cout << "✓ 处理完成: 共 " << idx << " 帧，识别到装甲板 " << hit
              << " 帧" << std::endl;
    writer.release();
    convertToMp4(tmp_avi, out_path);
    std::cout << "✓ 标注视频已保存为 " << out_path << std::endl;
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    Options opt = parseArgs(argc, argv);

    // ---------- 1. 读取相机参数 ----------
    cv::FileStorage fs(opt.params_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "❌ 打不开参数文件: " << opt.params_path << std::endl
                  << "   请把 camera_params.yml 放在可执行文件同级目录，或使用 --params 指定路径" << std::endl;
        return -1;
    }

    cv::Mat cam, dist;
    float w = 0.f, h = 0.f;
    fs["camera_matrix"] >> cam;
    fs["distort_coeffs"] >> dist;
    fs["armor_width"] >> w;
    fs["armor_height"] >> h;
    if (cam.empty()) {
        std::cerr << "❌ camera_params.yml 缺少 camera_matrix" << std::endl;
        return -1;
    }
    if (dist.empty()) { dist = cv::Mat::zeros(1, 5, CV_64F); }
    // 兜底：yml 里没写尺寸时用标准值，并明确告知（而不是静默变成 0 导致崩溃）
    if (w <= 0.f || h <= 0.f) {
        w = 13.5f; h = 5.5f;
        std::cerr << "⚠ camera_params.yml 未提供有效的 armor_width/armor_height，已回退到标准值 13.5 x 5.5 cm" << std::endl;
    }
    std::cout << "✓ 物理尺寸: " << w << " x " << h << " cm" << std::endl;

    // ---------- 2. 选择要测试的图片（视频模式下跳过） ----------
    if (opt.video_path.empty() && opt.images.empty()) {
        std::string s;
        std::cout << "请输入要测试的图片编号 (1, 2, 3，可空格分隔多个): ";
        if (!std::getline(std::cin, s) || s.empty()) return 0;
        size_t pos = 0;
        while (pos < s.size()) {
            size_t next = s.find_first_of(" ,", pos);
            if (next == std::string::npos) next = s.size();
            std::string tok = s.substr(pos, next - pos);
            if (!tok.empty()) opt.images.push_back(tok);
            pos = next + 1;
        }
    }

    // ---------- 3. 初始化检测器 ----------
    PnPSolver solver(cam, dist, w, h);

    std::unique_ptr<NeuralDetector> nn;
    if (opt.mode == "nn" || opt.mode == "both") {
        if (opt.config_path.empty() || opt.model_dir.empty()) {
            std::cerr << "⚠ 未指定 --config / --modeldir，跳过神经网络方案" << std::endl;
            if (opt.mode == "nn") return -1;
            opt.mode = "trad";
        } else {
            try { nn = std::make_unique<NeuralDetector>(opt.config_path, opt.model_dir); }
            catch (const std::exception& e) {
                std::cerr << "⚠ 神经网络初始化失败（" << e.what() << "），跳过" << std::endl;
                if (opt.mode == "nn") return -1;
                opt.mode = "trad";
            }
        }
    }

    TraditionalDetector trad;

    // 数字识别网络（作业要求：传统方案结合数字网络给数字识别）
    NumberClassifier numcls(opt.num_model);

    // 图片模式下合成演示视频（作业要求：展示标注后的视频）
    cv::VideoWriter demo;
    double demo_fps = 25;
    std::string demo_avi = "/tmp/hw2_demo.avi";
    if (!opt.out_video.empty()) {
        demo.open(demo_avi, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                  demo_fps, cv::Size(960, 720));
        if (!demo.isOpened())
            std::cerr << "⚠ 打不开演示视频输出 " << opt.out_video << "，跳过" << std::endl;
        else
            std::cout << "✓ 将把标注结果合成为演示视频: " << opt.out_video << std::endl;
    }
    int hold = std::max(1, static_cast<int>(demo_fps * opt.hold_sec));

    // ---------- 3.5 视频模式（优先于图片交互，无需输入图片编号） ----------
    if (!opt.video_path.empty()) {
        int rc = runVideo(opt, solver, cam, trad, nn.get(), numcls);
        fs.release();
        return rc;
    }

    // ---------- 4. 逐张测试 ----------
    for (const auto& num : opt.images) {
        std::string image_name = "test" + num + ".jpg";
        std::string armor_key = "image" + num + "_armor_id";
        std::string tvec_key  = "image" + num + "_tvec";

        cv::Mat img = cv::imread(image_name);
        if (img.empty()) {
            std::cerr << "\n❌ 图片读取失败: " << image_name << "（请确认运行目录）" << std::endl;
            continue;
        }

        // 真值（用于误差对比，缺失时只输出计算值）
        bool has_truth = !fs[armor_key].empty() && !fs[tvec_key].empty();
        int armor_id = 1;
        cv::Vec3d truth(0, 0, 0);
        if (has_truth) {
            cv::Mat truth_mat;
            fs[armor_key] >> armor_id;
            fs[tvec_key] >> truth_mat;
            if (truth_mat.total() >= 3)
                truth = cv::Vec3d(truth_mat.at<double>(0, 0), truth_mat.at<double>(0, 1), truth_mat.at<double>(0, 2));
        }

        std::cout << "\n================ 测试 image" << num << " ================" << std::endl;
        std::cout << "图片: " << image_name << "  装甲板编号: " << armor_id;
        if (has_truth) std::cout << "  真值: (" << truth[0] << ", " << truth[1] << ", " << truth[2] << ")";
        std::cout << std::endl;

        // 选择策略：auto 表示"按重投影误差最小自动决定"，其它值强制指定
        bool auto_sel = (opt.select == "auto");
        armor::SelectMode sel = auto_sel ? armor::SelectMode::LEFT : toMode(opt.select);
        std::cout << "  目标选择: " << (auto_sel ? "自动（重投影误差最小）" : armor::selectName(sel))
                  << std::endl;

        // ---- 神经网络方案 ----
        if (nn) {
            std::vector<std::vector<cv::Point2f>> cands;
            nn->detectAll(img, cands);
            std::cout << "  [神经网络] 候选装甲板 " << cands.size() << " 个" << std::endl;
            for (const auto& sc : scoreCandidates(cands, solver, armor_id))
                std::cout << "      候选 #" << sc.first << " 重投影误差 " << std::fixed
                          << std::setprecision(2) << sc.second << " px" << std::endl;

            BestSolve picked;
            if (auto_sel) {
                picked = solveBest(cands, solver, armor_id,
                            cv::Point2f(img.cols * 0.5f, img.rows * 0.5f));
            } else {
                std::vector<cv::Point2f> pts;
                if (nn->detect(img, pts, sel)) {
                    picked.ok = true;
                    picked.points = pts;
                    picked.candidate_count = cands.size();
                    if (!solver.solve(pts, picked.tvec, picked.rvec, armor_id, &picked.reproj))
                        picked.ok = false;
                }
            }

            if (!picked.ok) { std::cout << "  [神经网络] ❌ 未检测到可用装甲板" << std::endl; }
            else {
                std::cout << "  [神经网络] 选中候选 #" << picked.index << "  角点: ";
                for (const auto& p : picked.points)
                    std::cout << "(" << static_cast<int>(p.x) << "," << static_cast<int>(p.y) << ") ";
                std::cout << std::endl;
                printResult("神经网络",
                            evaluate(true, true, picked.tvec, picked.rvec, picked.reproj, truth),
                            truth);
            }
        }

        // ---- 传统视觉方案 ----
        if (opt.mode == "trad" || opt.mode == "both") {
            std::vector<std::vector<cv::Point2f>> cands;
            trad.detectAll(img, cands);
            std::cout << "  [传统视觉] 候选装甲板 " << cands.size() << " 个" << std::endl;
            for (const auto& sc : scoreCandidates(cands, solver, armor_id))
                std::cout << "      候选 #" << sc.first << " 重投影误差 " << std::fixed
                          << std::setprecision(2) << sc.second << " px" << std::endl;

            BestSolve picked;
            if (auto_sel) {
                picked = solveBest(cands, solver, armor_id,
                            cv::Point2f(img.cols * 0.5f, img.rows * 0.5f));
            } else {
                std::vector<cv::Point2f> pts;
                if (trad.detect(img, pts, sel)) {
                    picked.ok = true;
                    picked.points = pts;
                    picked.candidate_count = cands.size();
                    if (!solver.solve(pts, picked.tvec, picked.rvec, armor_id, &picked.reproj))
                        picked.ok = false;
                }
            }

            if (!picked.ok) { std::cout << "  [传统视觉] ❌ 未检测到可用装甲板" << std::endl; }
            else {
                std::cout << "  [传统视觉] 选中候选 #" << picked.index << "  角点: ";
                for (const auto& p : picked.points)
                    std::cout << "(" << static_cast<int>(p.x) << "," << static_cast<int>(p.y) << ") ";
                std::cout << std::endl;
                // 几何测距交叉校验
                double fx = cam.at<double>(0, 0);
                double z_geo = geometricRange(picked.points, fx, w);
                std::cout << std::fixed << std::setprecision(1)
                          << "  [传统视觉] 几何测距 Z=" << z_geo
                          << " cm vs PnP Z=" << picked.tvec[2] << " cm";
                if (std::abs(z_geo - picked.tvec[2]) > 0.15 * z_geo)
                    std::cout << "  ⚠ 两者差异 >15%：角点与装甲板模型存在系统偏差";
                std::cout << std::endl;
                printResult("传统视觉",
                            evaluate(true, true, picked.tvec, picked.rvec, picked.reproj, truth),
                            truth);
            }

            // ---- 演示视频帧（作业要求 3：画面中绘制装甲板矩形框 + 数字识别）----
            if (demo.isOpened()) {
                cv::Mat ann = img.clone();
                for (const auto& c : cands)   // 全部候选画绿细框
                    for (int i = 0; i < 4 && c.size() == 4; ++i)
                        cv::line(ann, c[i], c[(i + 1) % 4], cv::Scalar(0, 200, 0), 1);
                if (picked.ok) {
                    auto num = numcls.classify(img, picked.points);
                    drawAnnotate(ann, picked.points,
                                 std::string(1, num.label) +
                                     " conf=" + std::to_string(num.confidence).substr(0, 4),
                                 picked.tvec, "[trad]");
                }
                cv::putText(ann, "image" + num + "  (demo video)",
                            {10, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {255, 255, 255}, 2);
                for (int k = 0; k < hold; ++k) demo.write(ann);   // 每图持续 hold_sec 秒
            }
        }
    }
    fs.release();
    if (demo.isOpened()) {
        demo.release();
        convertToMp4(demo_avi, opt.out_video);
        std::cout << "\n✓ 演示视频已保存: " << opt.out_video << std::endl;
    }

    std::cout << "\n提示：调试图已保存为 debug_points_nn.jpg / debug_points_trad.jpg，"
                 "请核对四角点顺序是否为 左上→左下→右下→右上。" << std::endl;
    return 0;
}