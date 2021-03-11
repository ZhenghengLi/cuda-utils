#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <memory>
#include <chrono>
#include <cstring>

#include "ImageDataField.hh"
#include "CalibDataField.hh"

#define REPEAT_NUM 5000

using namespace std;
using std::chrono::duration;
using std::chrono::system_clock;
using std::micro;

int main(int argc, char** argv) {

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    cout << "deviceCount = " << deviceCount << endl;

    int deviceIndex = 0;
    cout << "deviceIndex = " << flush;
    cin >> deviceIndex;

    if (deviceIndex >= deviceCount) {
        cout << "deviceIndex is out of range [0, " << deviceCount << ")" << endl;
        return 1;
    }

    // select device and print its name on success
    if (cudaSetDevice(deviceIndex) == cudaSuccess) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, deviceIndex);
        cout << "successfully selected device " << deviceIndex << endl;
        // print uuid
        cout << "GPU " << deviceIndex << ": " << deviceProp.name << " (UUID: GPU-";
        for (int i = 0; i < 16; i++) {
            if (i == 4 || i == 6 || i == 8 || i == 10) cout << "-";
            cout << hex << (int)((uint8_t*)deviceProp.uuid.bytes)[i];
        }
        cout << ")" << endl;
    } else {
        cout << "failed to select device " << deviceIndex << endl;
        return 1;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////

    // prepare calib data
    CalibDataField* calib_data_host = nullptr;
    if (cudaMallocHost(&calib_data_host, sizeof(CalibDataField)) != cudaSuccess) {
        cerr << "cudaMallocHost failed for calib_data_host." << endl;
        return 1;
    }
    for (int m = 0; m < MOD_CNT; m++) {
        for (int i = 0; i < FRAME_H; i++) {
            for (int j = 0; j < FRAME_W; j++) {
                calib_data_host->pedestal[m][i][j] = 1000 + random() % 4000;
                calib_data_host->gain[m][i][j] = 100 + random() % 100;
            }
        }
    }

    // prepare image data
    ImageDataField* image_data_host = nullptr;
    ImageDataField* image_data_host_tmp = nullptr;
    if (cudaMallocHost(&image_data_host, sizeof(ImageDataField)) != cudaSuccess) {
        cerr << "cudaMallocHost failed for image_data_host." << endl;
        return 1;
    }
    if (cudaMallocHost(&image_data_host_tmp, sizeof(ImageDataField)) != cudaSuccess) {
        cerr << "cudaMallocHost failed for image_data_host_tmp." << endl;
        return 1;
    }
    // init
    for (int m = 0; m < MOD_CNT; m++) {
        for (int i = 0; i < FRAME_H; i++) {
            for (int j = 0; j < FRAME_W; j++) {
                image_data_host->pixel_data[m][i][j] = 6000 + random() % 9000;
            }
        }
    }

    // allocate memory on device
    CalibDataField* calib_data_device = nullptr;
    ImageDataField* image_data_device = nullptr;
    if (cudaMalloc(&calib_data_device, sizeof(CalibDataField)) != cudaSuccess) {
        cerr << "cudaMalloc failed for calib_data_device." << endl;
        return 1;
    }
    if (cudaMalloc(&image_data_device, sizeof(ImageDataField)) != cudaSuccess) {
        cerr << "cudaMalloc failed for image_data_device." << endl;
        return 1;
    }
    // copy calibration data into device
    if (cudaMemcpy(calib_data_device, calib_data_host, sizeof(CalibDataField), cudaMemcpyHostToDevice) != cudaSuccess) {
        cerr << "cudaMemcpy failed for calibration data." << endl;
        return 1;
    }

    // CPU test begin ///////////////////////////////////////////////
    duration<double, micro> start_time = system_clock::now().time_since_epoch();

    for (int r = 0; r < REPEAT_NUM; r++) {
        memcpy(image_data_host_tmp, image_data_host, sizeof(ImageDataField));
        for (int m = 0; m < MOD_CNT; m++) {
            for (int i = 0; i < FRAME_H; i++) {
                for (int j = 0; j < FRAME_W; j++) {
                    float& pixel = image_data_host_tmp->pixel_data[m][i][j];
                    float& pedestal = calib_data_host->pedestal[m][i][j];
                    float& gain = calib_data_host->gain[m][i][j];
                    pixel = (pixel - pedestal) / gain;
                }
            }
        }
        memcpy(image_data_host, image_data_host_tmp, sizeof(ImageDataField));
    }

    duration<double, micro> finish_time = system_clock::now().time_since_epoch();
    double time_used = finish_time.count() - start_time.count();

    double cpu_fps = REPEAT_NUM * 1000000.0 / time_used;
    cout << "CPU result: " << cpu_fps << " fps" << endl;

    // CPU test end /////////////////////////////////////////////////

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaStreamDestroy(stream);

    // clean data
    cudaFreeHost(calib_data_host);
    cudaFreeHost(image_data_host);
    cudaFreeHost(image_data_host_tmp);
    cudaFree(calib_data_device);
    cudaFree(image_data_device);

    return 0;
}