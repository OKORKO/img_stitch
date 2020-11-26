#include <iostream>
#include <fstream>
#include "opencv2/core.hpp"
#include <vector>
#include <queue>
#include <chrono>
#include <thread>
#include <fstream>
#include <sys/time.h>
#include <cstdlib>
#include <unistd.h>
#include <X11/Xlib.h>
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
#include <opencv2/cudawarping.hpp>
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include "opencv2/core/cuda.hpp"
//#include "cuda_runtime.h"
//#include "cuda_device_runtime_api.h"
#include "imgStitch.hpp"
#include "../streamer/streamer.hpp"
#include "omp.h"

using namespace cv;
using namespace std;
using namespace streamer;
using namespace cv::detail;
using time_point = std::chrono::high_resolution_clock::time_point;
using high_resolution_clock = std::chrono::high_resolution_clock;
std::mutex mtx;
extern "C"
void optimizeSeamKernelFun(cv::cuda::GpuMat matLeft, cv::cuda::GpuMat matRight, int seamColIndex);

imgStitch::imgStitch(size_t cam_nums,string rtmp_url) {
     _nums_images=cam_nums;
     _rtmp_url=rtmp_url;
     img_queue_cpu_list = vector<queue<Mat>>(cam_nums);
     img_queue_gpu_list = vector<queue<cuda::GpuMat>>(cam_nums);
     img_queue_cpu_warp_list = vector<queue<Mat>>(cam_nums);
     img_queue_gpu_warp_list = vector<queue<cuda::GpuMat>>(cam_nums);
     img_st_two_list = vector<queue<cuda::GpuMat>>(cam_nums-1);
     this->_fps=20;
}
imgStitch::~imgStitch() {

}

void imgStitch::frame_read_gpu_useone()
{
        //cv::gpu::setDeivce(0);
        clock_t startTime, endTime;
        cuda::GpuMat input,warp;
        Mat tmat;

        while (true)
        {

//            #pragma omp parallel num_threads(12)
//            #pragma omp for
            for(int i=0;i<this->_nums_images;i++)
            {

                cuda::GpuMat tmp;
                if(this->img_queue_gpu_list[i].empty())
                {
                    continue;
                }
                mtx.lock();
                input = this->img_queue_gpu_list[i].front();
                this->img_queue_gpu_list[i].pop();
                mtx.unlock();
                startTime = clock();//计时开始
                cv::cuda::remap(input, warp, this->_final_gpumap_vec[i * 2], this->_final_gpumap_vec[i * 2 +1], INTER_LINEAR);
                warp.copyTo(tmp);
                endTime = clock();//计时结束
                cout << "frame_read_gpu remap run time is: " << (double)(endTime - startTime) * 1000 / CLOCKS_PER_SEC << "ms" << endl;
                //warp.download(tmat);
                //imwrite("/home/chukunpeng/cproject/stitch/stitchV1/datasets/cam01/pic_undistored/warp"+to_string(i)+".jpg",tmat);
                mtx.lock();
                //std::lock_guard<std::mutex>lk(this->t_mut);
                this->img_queue_gpu_warp_list[i].push(tmp);
                if(this->img_queue_gpu_warp_list[i].size()>10)
                {
                    this->img_queue_gpu_warp_list[i].pop();
                }
                mtx.unlock();
                cout << "++++存在量：" + to_string(i) + "-------" << this->img_queue_gpu_warp_list[i].size() << endl;


            }
            //this->frame_write_gpu();

        }
}


void imgStitch::frame_read_gpu_all()
{
    cuda::GpuMat input,warp;
    Mat tmat;
    for(int i=0;i<this->_nums_images;i++)
    {
        cuda::GpuMat tmp;
        if(this->img_queue_gpu_list[i].empty()){continue;}
        input = this->img_queue_gpu_list[i].front();
        //this->img_queue_gpu_list[i].pop();
        //input.download(tmat);
        //imwrite("/home/chukunpeng/cproject/stitch/stitchV1/datasets/cam01/pic_undistored/warp_s"+to_string(i)+".jpg",tmat);
        cv::cuda::remap(input, warp, this->_final_gpumap_vec[i * 2], this->_final_gpumap_vec[i * 2 +1], INTER_LINEAR);
        //warp.download(tmat);
        //imwrite("/home/chukunpeng/cproject/stitch/stitchV1/datasets/cam01/pic_undistored/warp_f"+to_string(i)+".jpg",tmat);
        warp.copyTo(tmp);
        this->img_queue_gpu_warp_list[i].push(tmp);

    }

    vector<cuda::GpuMat> qgmatList;
    cuda::GpuMat t_mat;

    for(size_t i=0;i<this->_nums_images;i++)
    {
         cuda::GpuMat mp_map;
        if(!this->img_queue_gpu_warp_list[i].empty())
        {
            t_mat=this->img_queue_gpu_warp_list[i].front();
            t_mat.copyTo(mp_map);
            qgmatList.push_back(mp_map);
            t_mat.download(tmat);
            imwrite("/home/chukunpeng/cproject/stitch/stitchV3/datasets/cam01/pic_undistored/warp_mp"+to_string(i)+".jpg",tmat);
            //this->img_queue_gpu_warp_list[i].push(warp);
        }
    }


    if(qgmatList.size()==this->_nums_images)
    {
        for(size_t i=0;i<this->_nums_images-1;i++)
        {
            optimizeSeamKernelFun(qgmatList[i], qgmatList[i+1], this->_sec_index_vec[i]);
            this->img_queue_gpu_warp_list[i].pop();
            qgmatList[i+1].download(tmat);
            imwrite("/home/chukunpeng/cproject/stitch/stitchV1/datasets/cam01/pic_undistored/warp"+to_string(i)+".jpg",tmat);
        }
        this->img_queue_gpu_warp_list[this->_nums_images-1].pop();

    }
    qgmatList.clear();
}

void imgStitch::frame_read_gpu(size_t index)
{

        clock_t startTime, endTime;

        while (true)
        {
            if(this->img_queue_gpu_list[index].empty()){continue;}

            cuda::GpuMat input,warp;
            input = this->img_queue_gpu_list[index].front();
            this->img_queue_gpu_list[index].pop();
            //input_gpu= this->img_queue_gpu_list[index].front();
            //input_gpu.upload(input);
            startTime = clock();//计时开始
            cv::cuda::remap(input, warp, this->_final_gpumap_vec[index * 2], this->_final_gpumap_vec[index * 2 +1], INTER_LINEAR);

            endTime = clock();//计时结束
            cout << "frame_read_gpu remap run time is: " << (double)(endTime - startTime) * 1000 / CLOCKS_PER_SEC << "ms" << endl;
            cout << "存在量：" + to_string(index) + "-------" << this->img_queue_gpu_list[index].size() << endl;

            //this->img_queue_gpu_warp_list[index].push(warp);

        }
}

void imgStitch::frame_read_to_vec(string cam_url,size_t index)
{
     VideoCapture videoCapture = VideoCapture(cam_url, cv::CAP_FFMPEG);

     while (true)
     {
         Mat input;
         videoCapture >> input;
         if (input.empty())
         {
             continue;
         }
//         cuda::GpuMat input_gpu;
//         input_gpu.upload(input);
         mtx.lock();
         this->img_queue_cpu_list[index].push(input);
         if (this->img_queue_cpu_list[index].size() > 10)
         {
             for (size_t i = 0; i < this->img_queue_cpu_list[index].size() / 2; i++)
             {
                 this->img_queue_cpu_list[index].pop();
             }
         }

//         if (this->img_queue_gpu_list[index].size() > 10)
//         {
//             for (size_t i = 0; i < this->img_queue_gpu_list[index].size() / 2; i++)
//             {
//                 this->img_queue_gpu_list[index].pop();
//             }
//         }

//         //this->img_queue_cpu_list[index].push(input);
//         this->img_queue_gpu_list[index].push(input_gpu);
         mtx.unlock();
         //cout << "存在量：" + to_string(index) + "-------" << this->img_queue_gpu_list[index].size() << endl;

     }

}


void imgStitch::frame_read_cpu(size_t index) {

        Mat input;
        clock_t startTime, endTime;

        while (true)
        {
            if(this->img_queue_cpu_list[index].empty()){
                continue;}

            startTime = clock();//计时开始
            input=this->img_queue_cpu_list[index].front();
            this->img_queue_cpu_list[index].pop();
            Mat warp;
            cv::remap(input, warp, this->_final_map_list[index].xmap, this->_final_map_list[index].ymap, INTER_LINEAR);
            endTime = clock();//计时结束
            cout << "frame_read_cpu remap run time is: " << (double)(endTime - startTime) * 1000 / CLOCKS_PER_SEC << "ms" << endl;
            cout << "存在量：" + to_string(index) + "-------" << this->img_queue_cpu_list[index].size() << endl;

            this->img_queue_cpu_warp_list[index].push(warp);

        }

}

void imgStitch::frame_write(size_t index) {

        Mat output;
        cout << "5555";
        clock_t startTime, endTime;
        //#pragma omp parallel num_threads(6)
        //#pragma omp for
        //for(int i=0; i<1;i--)
        {
            while(1)
            {
                //cout << this->img_queue_cpu_list[index].size() << endl;
                if (!this->img_queue_cpu_list[index].empty())
                {

                     output = this->img_queue_cpu_list[index].front();
                     startTime = clock();//计时开始
                     Mat warp;
                     //#pragma omp parallel num_threads(4)
                     {
                         cv::remap(output, warp, this->_final_map_list[index].xmap, this->_final_map_list[index].ymap, INTER_LINEAR);
                     }
                     endTime = clock();//计时结束
                     cout << "the " << index << "The run time is: " << (double)(endTime - startTime) * 1000 / CLOCKS_PER_SEC << "ms" << endl;

                     //this->img_queue_cpu_list[index].pop();
                     //cout<< "显示："+to_string(index) + "-------" << this->img_queue_cpu_list[index].size() << endl;
                     //imshow("test"+to_string(index), output);
                }
            }
        }
}

void imgStitch::frame_write_gpu()
{
        cout << "stitch starat" << endl;
        vector<cuda::GpuMat> qgmatList;
        cuda::GpuMat t_mat;
        Mat sw_mat;
        size_t nums_images=this->_nums_images;
        clock_t startTime, endTime;
        while (true)
        {
            startTime = clock();//计时开始
            bool flag=false;
            cuda::GpuMat smp;
            for(size_t i=0;i<nums_images;i++)
            {
                cuda::GpuMat tmp;
                if(!this->img_queue_gpu_warp_list[i].empty())
                {
                    mtx.lock();
                    t_mat=this->img_queue_gpu_warp_list[i].front();
                    mtx.unlock();
                    t_mat.copyTo(tmp);
                    qgmatList.push_back(tmp);
                }
                tmp.release();
            }
            if(qgmatList.size()==nums_images)
            {
                for(size_t i=0;i<nums_images-1;i++)
                {
                    optimizeSeamKernelFun(qgmatList[i], qgmatList[i+1], this->_sec_index_vec[i]);
                    mtx.lock();
                    this->img_queue_gpu_warp_list[i].pop();
                    mtx.unlock();
                }
                mtx.lock();
                this->img_queue_gpu_warp_list[nums_images-1].pop();
                mtx.unlock();
                qgmatList[nums_images-1].copyTo(smp);
                mtx.lock();
                this->_final_gpustitchmat_vec.push(smp);
                mtx.unlock();
                //qgmatList[nums_images-1].download(sw_mat);
                //imwrite("/home/chukunpeng/cproject/stitch/stitchV1/datasets/cam01/pic_undistored/warp_mp111"+to_string(this->count++)+".jpg",sw_mat);

            }
            else
            {
                std::chrono::milliseconds dura(20);
                std::this_thread::sleep_for(dura);
            }
            qgmatList.clear();
            endTime = clock();//计时结束
            cout << "frame_write_gpu stitch run time is: " << (double)(endTime - startTime) * 1000 / CLOCKS_PER_SEC << "ms" << endl;

        }
}

void imgStitch::final_stitch_gpu_to_cpu()
{
    cout << "stitch starat" << endl;
    clock_t startTime, endTime;

    while(true)
    {
        Mat stitch_mat,ctmp;
        cuda::GpuMat tmp;
        startTime = clock();//计时开始
        {
            std::lock_guard<std::mutex>lk(this->t_mut);
            //mtx.lock();
            if(!this->img_st_two_list[this->_nums_images-2].empty())
            {
                tmp=this->img_st_two_list[this->_nums_images-2].front();
                this->img_st_two_list[this->_nums_images-2].pop();
            }
        }
        //mtx.unlock();

        tmp.download(stitch_mat);
        if(stitch_mat.empty())
        {
            continue;
        }
        cv::resize(stitch_mat,ctmp,Size(3200,980));
        //imwrite("/home/chukunpeng/cproject/stitch/stitchV3/datasets/cam01/pic_undistored/warp_mp"+to_string(this->count++)+".jpg",ctmp);
        mtx.lock();
        this->_final_cpumat_vec.push(ctmp);
        if(this->_final_cpumat_vec.size()>10)
        {
            this->_final_cpumat_vec.pop();
        }
        mtx.unlock();
        endTime = clock();//计时结束
        //cout << "final_stitch_gpu_to_cpu down run time is: " << (double)(endTime - startTime) * 1000 / CLOCKS_PER_SEC << "ms" << endl;
        //cout << "####存在量：-------" << this->_final_cpumat_vec.size() << endl;
        //imshow("stitch",stitch_mat);


    }
}
void imgStitch::frame_cpu_to_gpu()
{
    while (true)
    {
        for(int i=0;i<this->_nums_images;i++)
        {
            if(!this->img_queue_cpu_list[i].empty())
            {
                cuda::GpuMat g_mat;
                Mat c_mat;
                mtx.lock();
                c_mat = this->img_queue_cpu_list[i].front();
                //this->img_queue_cpu_list[i].pop();
                mtx.unlock();
                g_mat.upload(c_mat);
                mtx.lock();
                this->img_queue_gpu_list[i].push(g_mat);
                if(this->img_queue_gpu_list[i].size()>10)
                {
                    this->img_queue_gpu_list[i].pop();
                }
                mtx.unlock();
                //cout << "first two img_queue_gpu_list size is：" << to_string(i) << "---" << this->img_queue_gpu_list[i].size() << endl;
            }
            else
            {
                sleep_ms(3);
            }

        }
    }
}

void imgStitch::frame_gpumat_twostitch(int left_index,int right_index,int stitch_index,bool ismax_left)
{

    while (true)
    {
        cuda::GpuMat left_mat,right_mat,stitch_mat;
        bool lflag=false,nflag=false;
        if(ismax_left)
        {
            if(!this->img_queue_gpu_warp_list[left_index].empty() && !this->img_queue_gpu_warp_list[right_index].empty())
            {
                mtx.lock();
                left_mat=this->img_queue_gpu_warp_list[left_index].front();
                this->img_queue_gpu_warp_list[left_index].pop();

                right_mat=this->img_queue_gpu_warp_list[right_index].front();
                this->img_queue_gpu_warp_list[right_index].pop();
                mtx.unlock();

                optimizeSeamKernelFun(left_mat, right_mat, this->_sec_index_vec[stitch_index]);
                right_mat.copyTo(stitch_mat);
                //Mat tst;
                //stitch_mat.download(tst);
                //imwrite("/home/chukunpeng/cproject/stitch/stitchV3/datasets/cam01/pic_undistored/warp_mp"+to_string(this->count++)+".jpg",tst);
                std::lock_guard<std::mutex>lk(this->t_mut);
                this->img_st_two_list[stitch_index].push(stitch_mat);
                if(this->img_st_two_list[stitch_index].size()>10)
                {
                    this->img_st_two_list[stitch_index].pop();
                }
            }
            else
            {
                sleep_ms(3);
            }
        }
        else
        {
            if(!this->img_st_two_list[left_index].empty() && !this->img_queue_gpu_warp_list[right_index].empty())
            {
                {
                    std::lock_guard<std::mutex>lk(this->t_mut);
                    if(!this->img_st_two_list[left_index].empty())
                    {
                        left_mat=this->img_st_two_list[left_index].front();
                        this->img_st_two_list[left_index].pop();
                    }
                }
                mtx.lock();
                right_mat=this->img_queue_gpu_warp_list[right_index].front();
                this->img_queue_gpu_warp_list[right_index].pop();
                mtx.unlock();

                optimizeSeamKernelFun(left_mat, right_mat, this->_sec_index_vec[stitch_index]);
                right_mat.copyTo(stitch_mat);
                //Mat tst;
                //stitch_mat.download(tst);
                //imwrite("/home/chukunpeng/cproject/stitch/stitchV3/datasets/cam01/pic_undistored/11warp_mp"+to_string(this->count++)+".jpg",tst);

                std::lock_guard<std::mutex>lk(this->t_mut);
                this->img_st_two_list[stitch_index].push(stitch_mat);
                if(this->img_st_two_list[stitch_index].size()>10)
                {
                    this->img_st_two_list[stitch_index].pop();
                }
            }
            else
            {
                sleep_ms(3);
            }

        }

        //if(!left_mat.empty() && !right_mat.empty())


        //cout << "----img_st_two_list size is ---" << to_string(stitch_index) << " ----" <<"---:"<<this->img_st_two_list[stitch_index].size() << endl;

    }
}

void imgStitch::final_rtmp_push()
{
    bool from_camera = false;
    int stream_fps = this->_fps;
    size_t streamed_frames = 0;
    int bitrate = 1976018;
    //int bitrate = 500000;
    Streamer streamer;
    StreamerConfig streamer_config(3200, 980,
        3200, 980,
        stream_fps, bitrate, "main", this->_rtmp_url);

    cout << this->_rtmp_url<<endl;
    streamer.enable_av_debug_log();

    streamer.init(streamer_config);

    high_resolution_clock clk;
    time_point time_start = clk.now();
    time_point time_prev = time_start;

    MovingAverage moving_average(10);
    double avg_frame_time;

    time_point time_stop = clk.now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(time_stop - time_start);
    auto frame_time = std::chrono::duration_cast<std::chrono::duration<double>>(time_stop - time_prev);


    cout << "final_rtmp_push starat" << endl;

    struct timeval tv1,tv2;
    long long T;

    Mat stitch_mat;

    //stitch_mat=imread("/home/chukunpeng/cproject/stitch/stitchV3/map_mat_list10.jpg");
    while(true)
    {
        if(!this->_final_cpumat_vec.empty())
        //if(0<1)
        {
            gettimeofday(&tv1, NULL);
            mtx.lock();
            stitch_mat=this->_final_cpumat_vec.front();
            this->_final_cpumat_vec.pop();
            mtx.unlock();
            //imshow("stitch",stitch_mat);
            //imwrite("/home/chukunpeng/cproject/stitch/stitchV3/datasets/cam01/pic_undistored/warp_mp"+to_string(this->count++)+".jpg",stitch_mat);
            if (!from_camera) {
                  stream_frame(streamer, stitch_mat);
              }
              else {
                  stream_frame(streamer, stitch_mat, frame_time.count() * streamer.inv_stream_timebase);
              }
              time_stop = clk.now();
              elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(time_stop - time_start);
              frame_time = std::chrono::duration_cast<std::chrono::duration<double>>(time_stop - time_prev);

              if (!from_camera) {
                  streamed_frames++;
                  moving_average.add_value(frame_time.count());
                  avg_frame_time = moving_average.get_average();
                  add_delay(streamed_frames, stream_fps, elapsed_time.count(), avg_frame_time);
              }
              time_prev = time_stop;
              //结束计时
              gettimeofday(&tv2, NULL);
              T = (tv2.tv_sec - tv1.tv_sec) * 1000 + (tv2.tv_usec - tv1.tv_usec) / 1000;
              cout << "final_rtmp_push run time is T: " << T << "ms" << endl;
              //cout << "final_rtmp_push run time is: " << (double)(endTime - startTime) * 1000 / CLOCKS_PER_SEC << "ms" << endl;
        }
//        else
//        {
//            std::chrono::milliseconds dura(10);
//            std::this_thread::sleep_for(dura);
//        }

    }
    stitch_mat.release();
}

void imgStitch::frame_remap_vs() {
        FileStorage fs_read(".//params//camchain-awsome.yaml", FileStorage::READ);
        if (!fs_read.isOpened()) {
                fprintf(stderr, "%s:%d:loadParams falied. 'camera.yml' does not exist\n", __FILE__, __LINE__);
                return ;
        }
        Mat K, UK;
        Mat R = Mat::eye(3, 3, CV_32F);
        Mat EYE = Mat::eye(3, 3, CV_32F);
        int width = (int)fs_read["width"];
        int height = (int)fs_read["height"];
        fs_read["KMat"] >> K;
        //fs_read["RMat"] >> R;
        //fs_read["EYEMat"] >> EYE;
        //K.copyTo(UK);
        vector<double> D(4);
        fs_read["D"] >> D;


        Ptr<WarperCreator> warper_creator = makePtr<cv::CylindricalWarper>();  //柱面投影
        Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(1850.0));  //因为图像焦距都一样
        Size img_size(width, height);

        Mat cylindrical_xmap, cylindrical_ymap;
        Mat map1, map2;
        Mat FK, FR;
        K.convertTo(FK, CV_32F);
        R.convertTo(FR, CV_32F);
        //imgwp->buildMaps(img_size, K, R, xmap, ymap);
        warper->buildMaps(img_size, FK, FR, cylindrical_xmap, cylindrical_ymap);
        cv::initUndistortRectifyMap(FK, D, FR, UK, img_size, CV_32FC1, map1, map2);
        Rectify_map rectify_map = { map1,map2 };
        vector<Rectify_map> rectify_map_List = { rectify_map, rectify_map, rectify_map, rectify_map, rectify_map,rectify_map, rectify_map, rectify_map, rectify_map, rectify_map };

        string imdir = "./datasets/cam01/pic_raw/";
        vector<string> imList = { "00067.png","00078.png","00083.png","00095.png" ,"00100.png","00067.png","00078.png","00083.png","00095.png" ,"00100.png" };//

        size_t num_images = 5;
        vector<Mat> imMatList;
        vector<Mat> img_undistort_list;

        vector<Mat> img_cylindrical_list;
        for (int i = 0; i < num_images; i++)
        {
                Mat rdframe = imread(imdir + imList[i]);

                for (int j = 0; j < num_images; j++)
                {
                    this->img_queue_cpu_list[j].push(rdframe);
                }
                imMatList.push_back(rdframe);
                Mat img_undistort;
                //去畸变
                cv::remap(rdframe, img_undistort, rectify_map_List[i].xmap, rectify_map_List[i].ymap, INTER_LINEAR);

                //重投影
                Mat img_cylindrical;
                cv::remap(img_undistort, img_cylindrical, cylindrical_xmap, cylindrical_ymap, INTER_LINEAR);

                img_undistort_list.push_back(img_undistort);
                img_cylindrical_list.push_back(img_cylindrical);
                //imshow("cylindr" + to_string(i), img_cylindrical);
                //imwrite("cylindr" + to_string(i) + ".jpg", img_cylindrical);
                /*imwrite("distort" + to_string(i) + ".jpg", img_undistort);*/
        }
        vector<Mat> homography_list;
        Mat DEYE;
        EYE.convertTo(DEYE, CV_64F);
        homography_list.push_back(DEYE);
        Mat mask;

        for (size_t i = 0; i < num_images - 1; i++)
        {
                mask = myfindHomography(img_cylindrical_list[i], img_cylindrical_list[i + 1]);

                homography_list.push_back(homography_list[i] * mask);
        }
        mask.release();

        vector<four_corners_t> four_corners_tvec;
        float maxWidth, maxHeight;
        for (size_t i = 0; i < num_images; i++)
        {
                four_corners_t fc = CalcCorners(homography_list[i], img_cylindrical_list[i]);
                maxWidth = MAX(fc.right_top.x, fc.right_bottom.x);
                maxHeight = MAX(fc.left_bottom.y, fc.right_bottom.y);
                four_corners_tvec.push_back(fc);
        }
        //maxWidth = width * 2;
        for (size_t i = 0; i < four_corners_tvec.size() - 1; i++)
        {
                int index = four_corners_tvec[i + 1].left_bottom.x + (four_corners_tvec[i].right_bottom.x - four_corners_tvec[i + 1].left_bottom.x) / 2;
                this->_sec_index_vec.push_back(index);
        }
        this->_four_corners_tvec = four_corners_tvec;


        vector<Rectify_map> undistort_rectify_map_list;
        vector<Mat> mask_List;
        vector<double> TD = { 0,0,0,0 };

        vector<Mat> warped_list;

        for (size_t i = 0; i < num_images; i++)
        {
                Mat	H = homography_list[i];
                Mat img = img_cylindrical_list[i];
                Mat _map1, _map2;
                cv::initUndistortRectifyMap(EYE, TD, H, EYE, Size(maxWidth, maxHeight), CV_32FC1, _map1, _map2);
                Mat warped;
                cv::remap(img, warped, _map1, _map2, INTER_LINEAR);
                //imwrite("warped" + to_string(i) + ".jpg", warped);
                //param_img[param_img == 0] == warped[param_img.];
                Rectify_map xymap = { _map1, _map2 };
                undistort_rectify_map_list.push_back(xymap);
                warped_list.push_back(warped);
                //mask_List.push_back()
        }
        /*_map1.release();
        _map2.release();*/



        vector<Rectify_map> final_map_list;
        for (size_t i = 0; i < num_images; i++)
        {
                Mat map1x, map1y, mapx, mapy;
                cv::remap(rectify_map_List[i].xmap, map1x, cylindrical_xmap, cylindrical_ymap, INTER_LINEAR);
                cv::remap(rectify_map_List[i].ymap, map1y, cylindrical_xmap, cylindrical_ymap, INTER_LINEAR);

                cv::remap(map1x, mapx, undistort_rectify_map_list[i].xmap, undistort_rectify_map_list[i].ymap, INTER_LINEAR);
                cv::remap(map1y, mapy, undistort_rectify_map_list[i].xmap, undistort_rectify_map_list[i].ymap, INTER_LINEAR);

                Rectify_map xymap = { mapx,mapy };
                final_map_list.push_back(xymap);
        }
        this->_final_map_list = final_map_list;

        clock_t startTime, endTime;
        vector<Mat> map_mat_list;
        for (size_t i = 0; i < num_images; i++)
        {
                Mat warp;
                startTime = clock();//计时开始
                cv::remap(imMatList[i], warp, final_map_list[i].xmap, final_map_list[i].ymap, INTER_LINEAR);
                endTime = clock();//计时结束
                cout << "The run time is: " << (double)(endTime - startTime) * 1000 / CLOCKS_PER_SEC << "ms" << endl;
                //imwrite("warped" + to_string(i) + ".jpg", warp);
                map_mat_list.push_back(warp);
        }
        Mat param_img = Mat(height, maxWidth,CV_8UC3);
        param_img.setTo(0);
        startTime = clock();//计时开始
        for (size_t i = 0; i < num_images - 1; i++)
        {
                OptimizeSeam(warped_list[i + 1], warped_list[i], param_img, four_corners_tvec[i + 1]);
                //imwrite("param_img" + to_string(i)+".jpg", param_img);
                //OptimizeSeam_cpu(map_mat_list[i], map_mat_list[i + 1], this->_sec_index_vec[i]);
                /*imshow("param_img", param_img);
                imshow("map_mat_list", map_mat_list[i + 1]);*/
        }
        endTime = clock();//计时结束
        cout << "The run time is: " << (double)(endTime - startTime) * 1000 / CLOCKS_PER_SEC << "ms" << endl;
        imwrite("param_img.jpg", param_img);

        cout << "拼接完成" << endl;
}


void imgStitch::frame_remap() {
    vector<Rectify_map> rectify_map_List;
    vector<Rectify_map> cylindrical_map_List;
    Mat EYE = Mat::eye(3, 3, CV_32F);
    int width=1920,height=1080;
    Size img_size(width, height);
    for(int i=0;i<this->_nums_images;i++)
    {
        FileStorage fs_read(".//params//camchain-awsome"+to_string(i)+".yaml", FileStorage::READ);
        if (!fs_read.isOpened()) {
                fprintf(stderr, "%s:%d:loadParams falied. 'camera.yml' does not exist\n", __FILE__, __LINE__);
                return ;
        }
        Mat K, UK;
        Mat R = Mat::eye(3, 3, CV_32F);
//        width = (int)fs_read["width"];
//        height = (int)fs_read["height"];
        fs_read["KMat"] >> K;
        //fs_read["RMat"] >> R;
        //fs_read["EYEMat"] >> EYE;
        //K.copyTo(UK);
        vector<double> D(4);
        fs_read["D"] >> D;

        Ptr<WarperCreator> warper_creator = makePtr<cv::CylindricalWarper>();  //柱面投影
        Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(2500.0));  //因为图像焦距都一样

        Mat cylindrical_xmap, cylindrical_ymap;
        Mat map1, map2;
        Mat FK, FR;
        K.convertTo(FK, CV_32F);
        R.convertTo(FR, CV_32F);
        //imgwp->buildMaps(img_size, K, R, xmap, ymap);
        warper->buildMaps(img_size, FK, FR, cylindrical_xmap, cylindrical_ymap);
        cv::initUndistortRectifyMap(FK, D, FR, UK, img_size, CV_32FC1, map1, map2);
        Rectify_map rectify_map = { map1,map2 };
        Rectify_map cylindrical_map = { cylindrical_xmap,cylindrical_ymap };
        rectify_map_List.push_back(rectify_map);
        cylindrical_map_List.push_back(cylindrical_map);
    }


    string imdir = "./datasets/cam01/pic_raw/";
    //vector<string> imList = { "00067.png","00078.png","00083.png","00095.png" ,"00100.png","00067.png","00078.png","00083.png","00095.png" ,"00100.png" ,"00067.png","00078.png","00083.png","00095.png" ,"00100.png" };//,"00078.png"
    vector<string> imList = { "S000001.png","S000002.png","S000003.png","S000004.png" ,"000005.png","000006.png"
                              ,"000007.png","000008.png","000009.png" ,"0000010.png" ,"0000011.png","0000012.png"
                              ,"000001.png","000001.png" ,"000001.png" };

    //size_t num_images = imList.size();
    size_t num_images= this->_nums_images;
    vector<Mat> imMatList;
    vector<Mat> img_undistort_list;
    vector<Mat> img_cylindrical_list;
    for (int i = 0; i < num_images; i++)
    {
        Mat rdframe = imread(imdir + imList[i]);
        imMatList.push_back(rdframe);
        //cuda::GpuMat fgpu_mat;
        //fgpu_mat.upload(rdframe);
        this->img_queue_cpu_list[i].push(rdframe);
        //this->img_queue_gpu_list[i].push(fgpu_mat);

        Mat img_undistort;
        //去畸变
        cv::remap(rdframe, img_undistort, rectify_map_List[i].xmap, rectify_map_List[i].ymap, INTER_LINEAR);

        //重投影
        Mat img_cylindrical;
        cv::remap(img_undistort, img_cylindrical, cylindrical_map_List[i].xmap, cylindrical_map_List[i].ymap, INTER_LINEAR);

        img_undistort_list.push_back(img_undistort);
        img_cylindrical_list.push_back(img_cylindrical);
         //waitKey(10);
        //imshow("cylindr" + to_string(i), img_cylindrical);
        //imwrite("cylindr" + to_string(i) + ".jpg", img_cylindrical);
        //imwrite("distort" + to_string(i) + ".jpg", img_undistort);
    }
    vector<Mat> homography_list;
    Mat DEYE;
    EYE.convertTo(DEYE, CV_64F);
    homography_list.push_back(DEYE);
    Mat mask;
    for (size_t i = 0; i < num_images - 1; i++)
    {
         mask = myfindHomography(img_cylindrical_list[i], img_cylindrical_list[i + 1]);
         //findHomography(img_cylindrical_list[i], img_cylindrical_list[i], mask);
         homography_list.push_back(homography_list[i] * mask);
         //waitKey(10);
    }
    mask.release();

    vector<four_corners_t> four_corners_tvec;
    float maxWidth=0.0, maxHeight=0.0;
    for (size_t i = 0; i < num_images; i++)
    {
        four_corners_t fc = CalcCorners(homography_list[i], img_cylindrical_list[i]);
        maxWidth =MAX(maxWidth,MAX(fc.right_top.x, fc.right_bottom.x));
        maxHeight = MAX(maxHeight,MAX(fc.left_bottom.y, fc.right_bottom.y)) ;
        four_corners_tvec.push_back(fc);
    }
    for (size_t i = 0; i < four_corners_tvec.size() - 1; i++)
    {
        int index = four_corners_tvec[i + 1].left_bottom.x + (four_corners_tvec[i].right_bottom.x - four_corners_tvec[i + 1].left_bottom.x) / 2;
        this->_sec_index_vec.push_back(index);
    }
    this->_four_corners_tvec = four_corners_tvec;

    this->_maxWidth=maxWidth;
    this->_maxHight=maxHeight;

    cout << this->_maxWidth << endl;
    cout << this->_maxHight << endl;
    vector<Rectify_map> undistort_rectify_map_list;
    vector<Mat> mask_List;
    vector<double> TD = { 0,0,0,0 };
    vector<Mat> warped_list;
    for (size_t i = 0; i < num_images; i++)
    {
         Mat H = homography_list[i];
         Mat img = img_cylindrical_list[i];
         Mat _map1, _map2;
         cv::initUndistortRectifyMap(EYE, TD, H, EYE, Size(maxWidth, maxHeight), CV_32FC1, _map1, _map2);
         Mat warped;
         cv::remap(img, warped, _map1, _map2, INTER_LINEAR);
         //imwrite("warped" + to_string(i) + ".jpg", warped);
         cuda::GpuMat gwarped;
         gwarped.upload(warped);
         this->_final_gpuwarpmat_vec.push_back(gwarped);
         Rectify_map xymap = { _map1, _map2 };
         undistort_rectify_map_list.push_back(xymap);
         warped_list.push_back(warped);

         waitKey(10);
         //mask_List.push_back()
    }
    cout << "22222" << endl;


    vector<Rectify_map> final_map_list;
    //vector<Rectify_GpuMap> final_gpumap_list;

    for (size_t i = 0; i < num_images; i++)
    {
            Mat map1x, map1y, mapx, mapy;
            //Rectify_GpuMap gxymap;
            cv::cuda::GpuMat gxmap,gymap;

            cv::remap(rectify_map_List[i].xmap, map1x, cylindrical_map_List[i].xmap, cylindrical_map_List[i].ymap, INTER_LINEAR);
            cv::remap(rectify_map_List[i].ymap, map1y, cylindrical_map_List[i].xmap, cylindrical_map_List[i].ymap, INTER_LINEAR);

            cv::remap(map1x, mapx, undistort_rectify_map_list[i].xmap, undistort_rectify_map_list[i].ymap, INTER_LINEAR);
            cv::remap(map1y, mapy, undistort_rectify_map_list[i].xmap, undistort_rectify_map_list[i].ymap, INTER_LINEAR);

            Rectify_map xymap = { mapx,mapy };
//            gxymap.xgmap.upload(mapx);
//            gxymap.ygmap.upload(mapy);

            gxmap.upload(mapx);
            gymap.upload(mapy);
            final_map_list.push_back(xymap);
            //final_gpumap_list2.push_back(gxymap);
            this->_final_gpumap_vec.push_back(gxmap);
            this->_final_gpumap_vec.push_back(gymap);
            //final_gpumap_list.push_back(gmapx,gmapy);
    }
    this->_final_map_list = final_map_list;
    //_final_gpumap_list=final_gpumap_list;



    vector<Mat> map_mat_list;
    vector<cuda::GpuMat> map_gpumat_list;
    for (size_t i = 0; i < num_images; i++)
    {
        Mat warp;
        cuda::GpuMat gpu_warp;
        cv::remap(imMatList[i], warp, final_map_list[i].xmap, final_map_list[i].ymap, INTER_LINEAR);
        //imwrite("warped" + to_string(i) + ".jpg", warp);
        gpu_warp.upload(warp);
        map_mat_list.push_back(warp);
        map_gpumat_list.push_back(gpu_warp);
    }
    cout << "ttttt"<<endl;
    Mat param_img = Mat(height, maxWidth, CV_8UC3);
    param_img.setTo(0);
    for (size_t i = 0; i < num_images - 1; i++)
    {
         //OptimizeSeam(map_mat_list[i + 1], map_mat_list[i], param_img, four_corners_tvec[i + 1]);
         optimizeSeamKernelFun(map_gpumat_list[i], map_gpumat_list[i+1], this->_sec_index_vec[i]);
         //imwrite("param_img" + to_string(i)+".jpg", param_img);

         /*imshow("param_img", param_img);*/
         Mat mpmat;
         map_gpumat_list[i+1].download(mpmat);
         imwrite("map_mat_list"+ to_string(i)+".jpg", mpmat);
    }
    imwrite("param_img.jpg", param_img);

    cout << "拼接完成" << endl;
}

void imgStitch::stream_frame(Streamer &streamer, const cv::Mat &image)
{
    streamer.stream_frame(image.data);
}


void imgStitch::stream_frame(Streamer &streamer, const cv::Mat &image, int64_t frame_duration)
{
    streamer.stream_frame(image.data, frame_duration);
}
void imgStitch::add_delay(size_t streamed_frames, size_t fps, double elapsed, double avg_frame_time)
{
    //compute min number of frames that should have been streamed based on fps and elapsed
    double dfps = fps;
    size_t min_streamed = (size_t) (dfps*elapsed);
    size_t min_plus_margin = min_streamed + 2;

    if(streamed_frames > min_plus_margin) {
        size_t excess = streamed_frames - min_plus_margin;
        double dexcess = excess;

        //add a delay ~ excess*processing_time
//#define SHOW_DELAY
#ifdef SHOW_DELAY
        double delay = dexcess*avg_frame_time*1000000.0;
        printf("frame %07lu adding delay %.4f\n", streamed_frames, delay);
        printf("avg fps = %.2f\n", streamed_frames/elapsed);
#endif
        usleep(dexcess*avg_frame_time*1000000.0);
    }
}

Mat imgStitch::myfindHomography(Mat& img1, Mat& img2)
{
        Mat image1, image2;
        cvtColor(img1, image1, COLOR_RGB2GRAY);
        cvtColor(img2, image2, COLOR_RGB2GRAY);

        Ptr<Feature2D> featurefinder = xfeatures2d::SURF::create();//特征点检测方法
        ImageFeatures feature1, feature2;  //存储图像特征点

        //提取特征点
        //SiftFeatureDetector siftDetector();  // 海塞矩阵阈值
        vector<KeyPoint> keyPoint1, keyPoint2;
        /*siftDetector.detect(img1, keyPoint1);
        siftDetector.detect(img2, keyPoint2);*/

        computeImageFeatures(featurefinder, image1, feature1);    //计算图像特征
        computeImageFeatures(featurefinder, image2, feature2);    //计算图像特征
        keyPoint1 = feature1.keypoints;
        keyPoint2 = feature2.keypoints;


        FlannBasedMatcher matcher;

        vector<vector<DMatch>> m_knnMatches;
        vector<DMatch> matches;

        const float kRatio = 0.85;

        matcher.knnMatch(feature2.descriptors, feature1.descriptors, m_knnMatches, 2);
        vector<DMatch> GoodMatchePoints;

        for (int i = 0; i < m_knnMatches.size(); i++)
        {
                if (m_knnMatches[i][0].distance < kRatio * m_knnMatches[i][1].distance)
                {
                        GoodMatchePoints.push_back(m_knnMatches[i][0]);
                }
        }

        vector<Point2f> imagePoints1, imagePoints2;

        for (int i = 0; i < GoodMatchePoints.size(); i++)
        {
                imagePoints1.push_back(keyPoint1[GoodMatchePoints[i].trainIdx].pt);
                imagePoints2.push_back(keyPoint2[GoodMatchePoints[i].queryIdx].pt);
        }


        //获取图像1到图像2的投影映射矩阵，尺寸为3*3
        Mat homo = findHomography(imagePoints2, imagePoints1, RANSAC,5.0);

        return homo;

        /*Mat adjustMat = (Mat_<double>(3, 3) << 1.0, 0, img1.cols, 0, 1.0, 0, 0, 0, 1.0);
        Mat adjustHomo = adjustMat * homo;
        return adjustHomo;*/
}

//配准图的4个顶点经过透视变换之后的点坐标
four_corners_t imgStitch::GetRegistrationCorners(Size size, Mat H)
{
        int rows = size.height;
        int cols = size.width;

        four_corners_t corners;

        double v2[] = { 0, 0, 1 };//左上角
        double v1[3];//变换后的坐标值
        Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
        Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量

        V1 = H * V2;
        //左上角(0,0,1)
        corners.left_top.x = v1[0] / v1[2];
        corners.left_top.y = v1[1] / v1[2];

        //左下角(0,src.rows,1)
        v2[0] = 0;
        v2[1] = rows;
        v2[2] = 1;
        V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
        V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
        V1 = H * V2;
        corners.left_bottom.x = v1[0] / v1[2];
        corners.left_bottom.y = v1[1] / v1[2];

        //右上角(src.cols,0,1)
        v2[0] = cols;
        v2[1] = 0;
        v2[2] = 1;
        V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
        V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
        V1 = H * V2;
        corners.right_top.x = v1[0] / v1[2];
        corners.right_top.y = v1[1] / v1[2];

        //右下角(src.cols,src.rows,1)
        v2[0] = cols;
        v2[1] = rows;
        v2[2] = 1;
        V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
        V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
        V1 = H * V2;
        corners.right_bottom.x = v1[0] / v1[2];
        corners.right_bottom.y = v1[1] / v1[2];

        return corners;
}

four_corners_t imgStitch::CalcCorners(const Mat& H, const Mat& src)
{
        four_corners_t corners;
        double v2[] = { 0, 0, 1 };//左上角
        double v1[3];//变换后的坐标值
        Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
        Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量

        V1 = H * V2;
        //左上角(0,0,1)
        cout << "V2: " << V2 << endl;
        cout << "V1: " << V1 << endl;
        corners.left_top.x = v1[0] / v1[2];
        corners.left_top.y = v1[1] / v1[2];

        /*corners.left_top.x = corners.left_top.x < 0 ? 0 : corners.left_top.x;
        corners.left_top.y = corners.left_top.y < 0 ? 0 : corners.left_top.y;*/

        //左下角(0,src.rows,1)
        v2[0] = 0;
        v2[1] = src.rows;
        v2[2] = 1;
        V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
        V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
        V1 = H * V2;
        corners.left_bottom.x = v1[0] / v1[2];
        corners.left_bottom.y = v1[1] / v1[2];

        /*corners.left_bottom.x = corners.left_bottom.x < 0 ? 0 : corners.left_bottom.x;
        corners.left_bottom.y = corners.left_bottom.y < 0 ? 0 : corners.left_bottom.y;*/

        //右上角(src.cols,0,1)
        v2[0] = src.cols;
        v2[1] = 0;
        v2[2] = 1;
        V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
        V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
        V1 = H * V2;
        corners.right_top.x = v1[0] / v1[2];
        corners.right_top.y = v1[1] / v1[2];

        /*corners.right_top.x = corners.right_top.x < 0 ? 0 : corners.right_top.x;
        corners.right_top.y = corners.right_top.y < 0 ? 0 : corners.right_top.y;*/

        //右下角(src.cols,src.rows,1)
        v2[0] = src.cols;
        v2[1] = src.rows;
        v2[2] = 1;
        V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
        V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
        V1 = H * V2;
        corners.right_bottom.x = v1[0] / v1[2];
        corners.right_bottom.y = v1[1] / v1[2];

        /*corners.right_bottom.x = corners.right_bottom.x < 0 ? 0 : corners.right_bottom.x;
        corners.right_bottom.y = corners.right_bottom.y < 0 ? 0 : corners.right_bottom.y;*/

        return corners;
}

//优化两图的连接处，使得拼接自然
void imgStitch::OptimizeSeam(Mat& right_img, Mat& left_img, Mat& dst, four_corners_t corners)
{
        int start = MIN(corners.left_top.x, corners.left_bottom.x) + 300;//开始位置，即重叠区域的左边界
        //start = start < 0 ? 0 : start;
        double processWidth = right_img.cols - start;//重叠区域的宽度
        //double processWidth = 300;//重叠区域的宽度
        int rows = dst.rows;
        int cols = right_img.cols; //注意，是列数*通道数
        double alpha = 1;//img1中像素的权重

        //dst = left_img.clone();
        //start = start < 0 ? 0 : start;
        for (int i = 0; i < rows; i++)
        {
                uchar* p = right_img.ptr<uchar>(i);  //获取第i行的首地址
                uchar* t = left_img.ptr<uchar>(i);
                uchar* d = dst.ptr<uchar>(i);
                for (int j = 0; j < cols; j++)
                {
                        if (j < start )
                        {
                                if (d[j * 3] == 0 && d[j * 3 + 1] == 0 && d[j * 3 + 2] == 0)
                                {
                                        d[j * 3] = t[j * 3];
                                        d[j * 3 + 1] = t[j * 3 + 1];
                                        d[j * 3 + 2] = t[j * 3 + 2];
                                }
                        }
                        else
                        {
                                //如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
                                if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
                                        //if (t[j * 3] == t[j * 3 + 1] && t[j * 3 + 1] == t[j * 3 + 2])
                                {
                                        alpha = 1;
                                }
                                else
                                {
                                        //img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好
                                        alpha = (processWidth - (j - start)) / processWidth;
                                }

                                d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
                                d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
                                d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);
                        }
                }
        }
}


void * imgStitch::  frame_read_gpu_useone_fun(void* arg)
{
    imgStitch *stitch=(imgStitch *)arg;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(1, &mask);
    CPU_SET(2, &mask);
    CPU_SET(3, &mask);
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
    {
        perror("pthread_setaffinity_np");
    }
    stitch->final_rtmp_push();
}
void imgStitch::sleep_ms(unsigned int secs)
{
    struct timeval tval;
    tval.tv_sec=secs/1000;
    tval.tv_usec=(secs*1000)%1000000;
    select(0,NULL,NULL,NULL,&tval);
}
