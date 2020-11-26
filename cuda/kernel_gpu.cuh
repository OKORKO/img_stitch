#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
using namespace cv;

extern "C" void optimizeSeamKernelFun(cv::cuda::GpuMat matLeft, cv::cuda::GpuMat matRight, int seamColIndex);

