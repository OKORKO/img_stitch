set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -std=c++11 -Wall -fPIC")
SET(SOURCE_FILES                                  #设置变量，表示所有的源文件
     streamer.cpp
     )
INCLUDE_DIRECTORIES(                          #相关头文件的目录
     /usr/local/include
     )
 
    LINK_DIRECTORIES(                                  #相关库文件的目录
    /usr/local/lib
    )


add_library(streamer ${SOURCE_FILES})
# add_library(Detection Detection.cpp)
target_link_libraries(streamer ${AVCODEC_LIBRARY} ${AVFORMAT_LIBRARY} ${SWSCALE_LIBRARY} ${AVUTIL_LIBRARY} )
# target_link_libraries(Detection ${AVCODEC_LIBRARY} ${AVFORMAT_LIBRARY} ${AVUTIL_LIBRARY} ${SWSCALE_LIBRARY})




