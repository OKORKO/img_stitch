#include <stdio.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <cstdlib>
#include <unistd.h>
#include <chrono>
#include <queue>
#include <pthread.h>
#include <unistd.h>
#include <sched.h>
#include "opencv2/highgui.hpp"
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/calib3d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/core/cuda.hpp"
#include "imgStitch/imgStitch.cpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/core/cuda.hpp"
#include "omp.h"
using namespace cv;
using namespace std;
using namespace cv::detail;
using namespace streamer;
using time_point = std::chrono::high_resolution_clock::time_point;
using high_resolution_clock = std::chrono::high_resolution_clock;


void frame_read_gpu_useone_fun(){

}


int main()
{

        size_t nums_images = 12;
        vector<VideoCapture> videoCapture_list(nums_images);

//        vector<queue<Mat>> images_queue;
//        vector<string> cam_ips = { "188.10.34.102","188.10.34.102","188.10.34.102"
//                                   ,"188.10.34.102","188.10.34.102","188.10.34.102"
//                                   ,"188.10.34.102","188.10.34.102","188.10.34.102"
//                                   ,"188.10.34.102","188.10.34.102","188.10.34.102"
//                                   ,"188.10.34.102","188.10.34.102","188.10.34.102","188.10.34.102"};
//        for (size_t i = 0; i < nums_images; i++)
//        {
//                string video_fname = "rtsp://admin:admin123456@" + cam_ips[i] + ":554/cam/realmonitor?channel=1&subtype=0";
//                //cout << video_fname << endl;
//                VideoCapture video_capture = VideoCapture(video_fname, cv::CAP_FFMPEG);

//                if (!video_capture.isOpened()) {
//                        fprintf(stderr, "could not open video %s\n", video_fname.c_str());
//                        cout << "video_fname" << endl;
//                        video_capture.release();
//                }
//                videoCapture_list.push_back(video_capture);
//        }

        string rtmp_url="rtmp://188.10.34.82/live/test";
        imgStitch *stitch = new imgStitch(nums_images,rtmp_url);


        //stitch->frame_remap_vs();
        stitch->frame_remap();

         //#pragma omp parallel num_threads(12)frame_read_to_vec
          //#pragma omp for
//        for (size_t i = 0; i < nums_images; i++)
//        {
//             string video_fname = "rtsp://admin:admin123456@" + cam_ips[i] + ":554/cam/realmonitor?channel=1&subtype=0";
//             //std::thread fm_read(&imgStitch::frame_read_cpu, stitch, videoCapture_list[i], i);
//             std::thread fm_read_all(&imgStitch::frame_read_to_vec, stitch, video_fname,i);//frame_operwrite

//             //std::thread fm_write(&imgStitch::frame_read_gpu, stitch, i);//frame_operwrite

//             //std::thread fm_write(&imgStitch::frame_operwrite, stitch);//frame_operwrite

//             //fm _read.detach(); //需要等子线程执行完毕
//             fm_read_all.detach(); //不需要等子线程执行完毕
//             //fm_write.detach(); //不需要等子线程执行完毕
//        }
        //stitch->frame_read_gpu_all();

        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(0, &mask);
        if (sched_setaffinity(0, sizeof(mask), &mask) < 0)
        {
            perror("sched_setaffinity");
        }

        pthread_t fm_map_id=0;
        pthread_create(&fm_map_id, NULL,imgStitch::frame_read_gpu_useone_fun,stitch);


//        if (pthread_create(&fm_map_id, NULL,imgStitch::frame_read_gpu_useone_fun,stitch) != 0) {
//                perror("pthread_create");
//            }

//        std::thread fm_map(&imgStitch::frame_read_gpu_useone, stitch);//frame_operwrite
//        thread::id th_id = fm_map.get_id();

//        fm_map.detach();


//        std::thread fm_write_stitch(&imgStitch::frame_write_gpu, stitch);//frame_operwrite
//        fm_write_stitch.detach();

//        std::thread fm_gpu_down_cpu(&imgStitch::final_stitch_gpu_to_cpu, stitch);//frame_operwrite
//        fm_gpu_down_cpu.detach();

//        std::thread fm_rtmp_push(&imgStitch::final_rtmp_push, stitch);//frame_operwrite
//        fm_rtmp_push.detach();


        while (1) { if (waitKey(0) == 27)break; }

        return 0;

}


