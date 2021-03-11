#include "ImageDataField.hh"
#include "CalibDataField.hh"
#include "ImageFeature.hh"
#include <cuda_runtime.h>

bool gpu_do_calib(ImageDataField* image_data_device, CalibDataField* calib_data_device, cudaStream_t stream = 0);