#include "device_routines.hh"
#include <cuda_runtime.h>

#define STRIDE_H 8

__global__ void calib_kernel(ImageDataField* image_data, CalibDataField* calib_data) {
    int m = blockIdx.x;
    int i = blockIdx.y * STRIDE_H + threadIdx.x / FRAME_W;
    int j = threadIdx.x % FRAME_W;
    image_data->pixel_data[m][i][j] =
        (image_data->pixel_data[m][i][j] - calib_data->pedestal[m][i][j]) / calib_data->gain[m][i][j];
}

void gpu_do_calib(ImageDataField* image_data_device, CalibDataField* calib_data_device, cudaStream_t stream) {
    calib_kernel<<<dim3(MOD_CNT, FRAME_H / STRIDE_H), FRAME_W * STRIDE_H, 0, stream>>>(
        image_data_device, calib_data_device);
}
