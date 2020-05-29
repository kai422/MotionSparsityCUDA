cmake_minimum_required(VERSION 3.16)
SET(CMAKE_CXX_COMPILER /usr/local/bin/g++-9)

project(MotionSparsity)

set(CMAKE_PREFIX_PATH "/Users/kai/CLionProjects/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)        # find required libtorch
find_package(OpenCV REQUIRED)       # find required opencv
find_package(PythonLibs REQUIRED)

if (NOT Torch_FOUND)
    message(FATAL_ERROR "Pytorch Not Found!")
endif (NOT Torch_FOUND)

message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

message(STATUS "PYTHON library status:")
message(STATUS "    version: ${PYTHONLIBS_VERSION_STRING}")
message(STATUS "    libraries: ${PYTHON_LIBRARIES}")
message(STATUS "    include path: ${PYTHON_INCLUDE_DIRS}")


#add_subdirectory(libraries)
#add_subdirectory(executables)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(MotionSparsity
        src/add.cpp
        #src/data_create.cpp
        #src/gridpool.cpp
        #src/merge.cpp
        #src/tensor_combine.cpp
        #src/tensor_split.cpp
        pybind/bind.cpp)

#target_source(hello PRIVATE **.cpp)
target_include_directories(MotionSparsity
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS})

target_link_libraries(MotionSparsity
        PUBLIC
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        ${PYTHON_LIBRARIES})
set_property(TARGET MotionSparsity PROPERTY CXX_STANDARD 14)



