#include "device_routines.hh"
#include <cuda_runtime.h>

__global__ void calib_kernel(ImageDataField* image_data, CalibDataField* calib_data) {
    int m = blockIdx.x;
    int i = threadIdx.x;
    int j = threadIdx.y;
    image_data->pixel_data[m][i][j] =
        (image_data->pixel_data[m][i][j] - calib_data->pedestal[m][i][j]) / calib_data->gain[m][i][j];
}

void gpu_do_calib(ImageDataField* image_data_device, CalibDataField* calib_data_device, cudaStream_t stream) {
    calib_kernel<<<MOD_CNT, dim3(FRAME_H, FRAME_W), 0, stream>>>(image_data_device, calib_data_device);
}
