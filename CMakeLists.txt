SET(execName realsense_bag_extract)
CMAKE_MINIMUM_REQUIRED(VERSION 3.1)
PROJECT(${execName})

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

find_package( OpenCV REQUIRED )
find_package( realsense2 REQUIRED )

FILE(GLOB_RECURSE SRC_FILES src/*.cpp)
FILE(GLOB_RECURSE HDR_FILES include/*.hpp)

ADD_EXECUTABLE( ${execName} ${HDR_FILES} ${SRC_FILES})
TARGET_LINK_LIBRARIES( ${execName} ${OpenCV_LIBS} ${realsense2_LIBRARY})
include_directories(include ${realsense_INCLUDE_DIR})
