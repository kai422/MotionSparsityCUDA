cmake_minimum_required(VERSION 3.16)

project(MotionSparsity)

set(CMAKE_PREFIX_PATH "$ENV{HOME}/libtorch/share/cmake/Torch")
set(PYTHON_INCLUDE_DIRS "$ENV{HOME}/anaconda3/envs/PyTorch/include/python3.7m")
set(PYTHON_LIBRARIES "$ENV{HOME}/anaconda3/envs/PyTorch/lib/libpython3.7m.so")
set(PYTORCH_INCLUDE_DIRS 
"$ENV{HOME}/anaconda3/envs/PyTorch/lib/python3.7/site-packages/torch/include"
"$ENV{HOME}/anaconda3/envs/PyTorch/lib/python3.7/site-packages/torch/include/torch/csrc/api/include"
"$ENV{HOME}/anaconda3/envs/PyTorch/lib/python3.7/site-packages/torch/include/TH"
"$ENV{HOME}/anaconda3/envs/PyTorch/lib/python3.7/site-packages/torch/include/THC")
set(PYTORCH_LIBRARIES "$ENV{HOME}/anaconda3/envs/PyTorch/lib/python3.7/site-packages/torch/lib")


message(STATUS "Pytorch status:")
message(STATUS "    include path: ${PYTORCH_INCLUDE_DIRS}")
message(STATUS "    libraries: ${PYTORCH_LIBRARIES}")

message(STATUS "PYTHON library status:")
message(STATUS "    version: ${PYTHONLIBS_VERSION_STRING}")
message(STATUS "    libraries: ${PYTHON_LIBRARIES}")
message(STATUS "    include path: ${PYTHON_INCLUDE_DIRS}")


#add_subdirectory(libraries)
#add_subdirectory(executables)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(MotionSparsity
        pybind.cpp
        data_create.cpp)

target_include_directories(MotionSparsity
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PYTORCH_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS})


target_link_directories(MotionSparsity
        PUBLIC
        ${PYTORCH_LIBRARIES})
set_property(TARGET MotionSparsity PROPERTY CXX_STANDARD 14)



