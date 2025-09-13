cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(gpt149)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PATH} ${CMAKE_PREFIX_PATH})

# 1. 关键步骤：查找 PyTorch 包
find_package(Torch REQUIRED)

# 2. 添加你的可执行文件或库
add_executable(module module.cpp)
# 或者添加库：add_library(your_library SHARED your_source.cpp)

# 3. 链接 Torch 库
target_link_libraries(module "${TORCH_LIBRARIES}")
# 如果是库，同样链接：target_link_libraries(your_library "${TORCH_LIBRARIES}")

# 4. (可选) 传递必要的编译标志
target_compile_options(module PRIVATE ${TORCH_CXX_FLAGS})
