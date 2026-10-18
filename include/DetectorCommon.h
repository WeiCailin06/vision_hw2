#pragma once
// =====================================================================
// DetectorCommon.h
// 两个检测器（神经网络 / 传统视觉）共用的公共类型定义
// =====================================================================

namespace armor {

// 多目标场景下的"目标装甲板"选择策略
enum class SelectMode {
    LEFT,     // 选图像中最靠左的装甲板（对应真值 tvec.x < 0 的场景）
    RIGHT,    // 选图像中最靠右的装甲板（对应真值 tvec.x > 0 的场景）
    LARGEST   // 选投影面积最大的装甲板（一般是最正对相机的那个）
};

inline const char* selectName(SelectMode m) {
    switch (m) {
        case SelectMode::LEFT:    return "最左侧装甲板";
        case SelectMode::RIGHT:   return "最右侧装甲板";
        case SelectMode::LARGEST: return "最大面积装甲板";
    }
    return "?";
}

// 统一的四点顺序约定（与 PnPSolver 中的物体点顺序一一对应）：
//   points[0] = 左上  points[1] = 左下  points[2] = 右下  points[3] = 右上
// （"左/右"以图像视角为准，"上/下"以灯条端点为准）

} // namespace armor