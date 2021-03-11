#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <memory>

#include "ImageDataField.hh"

using namespace std;

class CalibData {
public:
    CalibData() {
        pedestal = new float[MOD_CNT][FRAME_H][FRAME_W];
        gain = new float[MOD_CNT][FRAME_H][FRAME_W];
    }
    ~CalibData() {
        delete[] pedestal;
        delete[] gain;
    }

public:
    float (*pedestal)[FRAME_H][FRAME_W];
    float (*gain)[FRAME_H][FRAME_W];
};

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
    CalibData calib;
    for (int m = 0; m < MOD_CNT; m++) {
        for (int i = 0; i < FRAME_H; i++) {
            for (int j = 0; j < FRAME_W; j++) {
                calib.pedestal[m][i][j] = 1000 + random() % 4000;
                calib.gain[m][i][j] = 100 + random() % 100;
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

    ImageDataField* image_data_device = nullptr;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaStreamDestroy(stream);

    // clean data
    cudaFreeHost(image_data_host);
    cudaFreeHost(image_data_host_tmp);

    return 0;
}