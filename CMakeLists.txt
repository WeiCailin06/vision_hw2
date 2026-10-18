cmake_minimum_required(VERSION 3.22)
project(VisionHomework LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

# ---------------------------------------------------------------------
# 深大开源库路径：优先用命令行传入，否则使用默认值
#   cmake .. -DSZU_ROOT=/你的/路径/26_NNDeployment_Lib_and_Detection_Models-main
# ---------------------------------------------------------------------
if(NOT DEFINED SZU_ROOT)
    set(SZU_ROOT "/home/weicailin/26_NNDeployment_Lib_and_Detection_Models-main")
endif()

set(SZU_INCLUDE_DIRS
    "${SZU_ROOT}/src/core/algorithm/NNdeployment/interface"
    "${SZU_ROOT}/src/app_plugin/detector/include"
)
set(SZU_LIB "${SZU_ROOT}/build/lib/libNNdeployment_lib.so")

if(NOT EXISTS "${SZU_LIB}")
    message(FATAL_ERROR
        "\n找不到深大库: ${SZU_LIB}\n"
        "请检查路径是否正确，或用 -DSZU_ROOT=... 指定。\n")
endif()

file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(main ${SOURCES})

target_include_directories(main PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${SZU_INCLUDE_DIRS}
)

target_link_libraries(main PRIVATE
    ${OpenCV_LIBS}
    openvino::runtime
    "${SZU_LIB}"
)

# 运行时可执行文件要能找到 .so（省去手动设 LD_LIBRARY_PATH）
set_target_properties(main PROPERTIES
    BUILD_RPATH "${SZU_ROOT}/build/lib"
    INSTALL_RPATH "${SZU_ROOT}/build/lib"
)

# 把配置文件复制到可执行文件目录
add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/camera_params.yml"
        $<TARGET_FILE_DIR:main>/
)

# 把测试图片也复制过去，避免"图片读取失败"
file(GLOB TEST_IMAGES "${CMAKE_CURRENT_SOURCE_DIR}/test*.jpg")
if(TEST_IMAGES)
    add_custom_command(TARGET main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TEST_IMAGES} $<TARGET_FILE_DIR:main>/
    )
endif()