#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include <opencv2/cudaobjdetect.hpp>
#include <vector>
#include <queue>
#include "../streamer/streamer.hpp"
//#include "kernel_gpu.cuh"
using namespace cv;
using namespace std;
using namespace streamer;
typedef struct
{
    Mat xmap;
    Mat ymap;
}Rectify_map;
//struct Rectify_GpuMap
//{
//        cv::cuda::GpuMat xgmap;
//        cv::cuda::GpuMat ygmap;
//};
typedef struct
{
    Point2f left_top;
    Point2f left_bottom;
    Point2f right_top;
    Point2f right_bottom;
}four_corners_t;
class imgStitch {
public:

        mutable std::mutex t_mut;
        int count=0;

        int _maxWidth;
        int _maxHight;
        int _fps;
        string _rtmp_url;

        vector<queue<cuda::GpuMat>> img_st_two_list;

        size_t _nums_images;
        vector<queue<Mat>> img_queue_cpu_list;
        vector<queue<cuda::GpuMat>> img_queue_gpu_list;

        vector<queue<Mat>> img_queue_cpu_warp_list;
        vector<queue<cuda::GpuMat>> img_queue_gpu_warp_list;

        vector<Rectify_map> _final_map_list;

        //vector<Rectify_GpuMap> _final_gpumap_list;
        vector<cuda::GpuMat> _final_gpumap_vec;

        vector<four_corners_t> _four_corners_tvec;

        vector<int> _sec_index_vec;

        vector<int> secIndex_vec;

        vector<cuda::GpuMat> _final_gpuwarpmat_vec;

        queue<cuda::GpuMat> _final_gpustitchmat_vec;

        queue<Mat> _final_cpumat_vec;

        //vector<threadsafe_queue<Mat>*> img_queue_list;
        //vector<VideoCapture*> videoCapture_list;
public:
        imgStitch(size_t cam_nums,string url);
        ~imgStitch();

        void frame_read_gpu(size_t index);

        void frame_read_cpu(size_t index);

        void frame_read_gpu_useone();
        void frame_read_gpu_all();

        void frame_read_to_vec(string cam_url,size_t index);

        void frame_write(size_t index);

        void frame_write_gpu();

         void frame_cpu_to_gpu();

        void final_stitch_gpu_to_cpu();

        void frame_gpumat_twostitch(int left_index,int right_index,int stitch_index,bool ismax_left);

        void final_rtmp_push();

        void frame_remap_vs();
        void frame_remap();

        Mat myfindHomography(Mat& img1, Mat& img2);

        void stream_frame(Streamer &streamer, const cv::Mat &image);

        void stream_frame(Streamer &streamer, const cv::Mat &image, int64_t frame_duration);

        void add_delay(size_t streamed_frames, size_t fps, double elapsed, double avg_frame_time);

        four_corners_t GetRegistrationCorners(Size size, Mat H);

        four_corners_t CalcCorners(const Mat& H, const Mat& src);

        void OptimizeSeam(Mat& right_img, Mat& left_img, Mat& dst, four_corners_t corners);

        static void * frame_read_gpu_useone_fun(void* arg);

        static void sleep_ms(unsigned int secs);

        //extern "C" void optimizeSeamKernelFun(cv::cuda::GpuMat matLeft, cv::cuda::GpuMat matRight, int seamColIndex);
};
class MovingAverage
{
    int size;
    int pos;
    bool crossed;
    std::vector<double> v;

public:
    explicit MovingAverage(int sz)
    {
        size = sz;
        v.resize(size);
        pos = 0;
        crossed = false;
    }

    void add_value(double value)
    {
        v[pos] = value;
        pos++;
        if(pos == size) {
            pos = 0;
            crossed = true;
        }
    }

    double get_average()
    {
        double avg = 0.0;
        int last = crossed ? size : pos;
        int k=0;
        for(k=0;k<last;k++) {
            avg += v[k];
        }
        return avg / (double)last;
    }
};
