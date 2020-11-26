cmake_minimum_required(VERSION 3.10)
project(imgStitch)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -std=c++11 -Wall -fPIC")
SET(SOURCE_FILES                                  #设置变量，表示所有的源文件
     imgStitch.cpp
     )
INCLUDE_DIRECTORIES(                          #相关头文件的目录
     /usr/local/include
     )
 
    LINK_DIRECTORIES(                                  #相关库文件的目录
    /usr/local/libd
    )
add_library(imgStitch ${SOURCE_FILES})
target_link_libraries(imgStitch ${AVCODEC_LIBRARY} ${AVFORMAT_LIBRARY} ${AVUTIL_LIBRARY} ${SWSCALE_LIBRARY})



