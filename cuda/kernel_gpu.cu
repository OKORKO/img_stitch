#include <opencv2/opencv.hpp>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "kernel_gpu.cuh"
using namespace cv;
using namespace std;
//出错处理函数
#define CHECK_ERROR(call){\
    const cudaError_t err = call;\
    if (err != cudaSuccess)\
    {\
        printf("Error:%s,%d,",__FILE__,__LINE__);\
        printf("code:%d,reason:%s\n",err,cudaGetErrorString(err));\
        exit(1);\
    }\
}
__global__ void optimizeSeamKernel(cv::cuda::PtrStepSz<uchar3> left, cv::cuda::PtrStepSz<uchar3> right, int seamColIndex)
{
        int i = threadIdx.x + blockIdx.x*blockDim.x;//列
        int j = threadIdx.y + blockIdx.y*blockDim.y;//行

        int start = seamColIndex - 50;//开始位置，即重叠区域的左边界
        double processWidth = 100;//重叠区域的宽度
        double alpha = 1;//img1中像素的权重

        if (i >= 0 && i < right.cols)
        {
                if (j >= 0 && j < right.rows)
                {
                        if (i >= 0 && i < start + processWidth)
                        {
                                if (i < start)
                                {
                                        right(j, i) = left(j, i);
                                }
                                else
                                {
                                    //如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
                                    if (right(j, i).x == 0 && right(j, i).y == 0 && right(j, i).z == 0)
                                    {
                                            alpha = 1;
                                    }
                                    else
                                    {
                                            //img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好
                                            alpha = (processWidth - (i - start)) / processWidth;
                                    }
                                    right(j, i).x = left(j, i).x * alpha + right(j, i).x * (1 - alpha);
                                    right(j, i).y = left(j, i).y * alpha + right(j, i).y* (1 - alpha);
                                    right(j, i).z = left(j, i).z* alpha + right(j, i).z* (1 - alpha);
                                }

                        }

                }
        }

}


__global__ void optimizeSeamKernel_dst(cv::cuda::PtrStepSz<uchar3> left, cv::cuda::PtrStepSz<uchar3> right,cv::cuda::PtrStepSz<uchar3> dst, int seamColIndex)
{
        int i = threadIdx.x + blockIdx.x*blockDim.x;//列
        int j = threadIdx.y + blockIdx.y*blockDim.y;//行

        int start = seamColIndex - 50;//开始位置，即重叠区域的左边界
        double processWidth = 100;//重叠区域的宽度
        double alpha = 1;//img1中像素的权重

        if (i >= 0 && i < right.cols)
        {
                if (j >= 0 && j < right.rows)
                {
                        if (i >= 0 && i < start + processWidth)
                        {
                                if (i < start)
                                {
                                        right(j, i) = left(j, i);
                                }
                                else
                                {
                                    //如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
                                    if (right(j, i).x == 0 && right(j, i).y == 0 && right(j, i).z == 0)
                                    {
                                            alpha = 1;
                                    }
                                    else
                                    {
                                            //img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好
                                            alpha = (processWidth - (i - start)) / processWidth;
                                    }
                                    right(j, i).x = left(j, i).x * alpha + right(j, i).x * (1 - alpha);
                                    right(j, i).y = left(j, i).y * alpha + right(j, i).y* (1 - alpha);
                                    right(j, i).z = left(j, i).z* alpha + right(j, i).z* (1 - alpha);
                                }

                        }

                }
        }

}


extern "C"
void optimizeSeamKernelFun(cv::cuda::GpuMat matLeft, cv::cuda::GpuMat matRight, int seamColIndex)
{
        dim3 threads(32, 32);
        dim3 blocks(matLeft.cols / threads.x, matLeft.rows / threads.y);

        optimizeSeamKernel<< <blocks, threads >> >(matLeft, matRight, seamColIndex);

        CHECK_ERROR(cudaDeviceSynchronize());
}
