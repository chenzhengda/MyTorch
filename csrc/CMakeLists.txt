add_library(MyTorch SHARED tensor.cpp cpu.cpp)
target_include_directories(MyTorch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# 添加编译选项
target_compile_options(MyTorch PRIVATE -Wall -Wextra)

# 设置安装目标，指定 .so 文件安装路径
install(TARGETS MyTorch
    LIBRARY DESTINATION /data/jieni/workspace/code/MyTorch/mytorch    # 这里改为你想要的具体路径
)