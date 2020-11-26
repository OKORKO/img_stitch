
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


int main()
{

        size_t nums_images = 4;
        vector<VideoCapture> videoCapture_list(nums_images);

        vector<queue<Mat>> images_queue;
        vector<string> cam_ips = { "188.10.34.102","188.10.34.104","188.10.34.115","188.10.34.103"
                                   ,"188.10.34.102","188.10.34.102"
                                   ,"188.10.34.102","188.10.34.102","188.10.34.102"
                                   ,"188.10.34.102","188.10.34.102","188.10.34.102"
                                   ,"188.10.34.102","188.10.34.102","188.10.34.102","188.10.34.102"};
        for (size_t i = 0; i < nums_images; i++)
        {
//                string video_fname = "rtsp://admin:admin123456@" + cam_ips[i] + ":554/cam/realmonitor?channel=1&subtype=0";
//                //cout << video_fname << endl;
//                VideoCapture video_capture = VideoCapture(video_fname, cv::CAP_FFMPEG);

//                if (!video_capture.isOpened()) {
//                        fprintf(stderr, "could not open video %s\n", video_fname.c_str());
//                        cout << "video_fname" << endl;
//                        video_capture.release();
//                }
//                Mat tst;
//                video_capture >> tst;
//                imwrite("/home/chukunpeng/cproject/stitch/stitchV3/datasets/cam01/pic_raw/S00000"+to_string(i+1)+".png",tst);
//                videoCapture_list.push_back(video_capture);
        }
//        clock_t startTime, endTime;
//        Mat tst;
//        while(true)
//        {
//            startTime=clock();
//            for (size_t i = 0; i < nums_images; i++)
//            {
//                videoCapture_list[i] >> tst;
//            }
//            endTime=clock();
//            waitKey(30);
//            cout << "frame_read_gpu remap run time is: " << (double)(endTime - startTime) * 1000 / CLOCKS_PER_SEC << "ms" << endl;
//        }

//        int num_devices;
//        cudaGetDeviceCount(&num_devices);

//        cout << "cudaGetDeviceCount:" << to_string(num_devices) << endl;
//        cudaSetDevice(0);

        string rtmp_url="rtmp://172.16.20.19/live/test";
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

//             //fm_read.detach(); //需要等子线程执行完毕
//             fm_read_all.detach(); //不需要等子线程执行完毕
//             //fm_write.detach(); //不需要等子线程执行完毕
//        }
        //stitch->frame_read_gpu_all();

        std::thread fm_cpu_gpu(&imgStitch::frame_cpu_to_gpu, stitch);//frame_operwrite
        fm_cpu_gpu.detach();

        std::thread fm_map(&imgStitch::frame_read_gpu_useone, stitch);//frame_operwrite
        fm_map.detach();



        std::thread fm_write_stitch_0(&imgStitch::frame_gpumat_twostitch, stitch,0,1,0,true);//frame_operwrite
        fm_write_stitch_0.detach();
        std::thread fm_write_stitch_1(&imgStitch::frame_gpumat_twostitch, stitch,0,2,1,false);//frame_operwrite
        fm_write_stitch_1.detach();
        std::thread fm_write_stitch_2(&imgStitch::frame_gpumat_twostitch, stitch,1,3,2,false);//frame_operwrite
        fm_write_stitch_2.detach();



//        std::thread fm_write_stitch(&imgStitch::frame_write_gpu, stitch);//frame_operwrite
//        fm_write_stitch.detach();

        std::thread fm_gpu_down_cpu(&imgStitch::final_stitch_gpu_to_cpu, stitch);//frame_operwrite
        fm_gpu_down_cpu.detach();

        std::thread fm_rtmp_push(&imgStitch::final_rtmp_push, stitch);//frame_operwrite
        fm_rtmp_push.detach();



        while (1) { if (waitKey(0) == 27)break; }

        return 0;

}


