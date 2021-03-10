#include <iostream>
#include <cuda_runtime.h>

#include "ImageDataField.hh"

using namespace std;

int main(int argc, char** argv) {

    int deviceIndex = 0;

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    cout << "deviceCount = " << deviceCount << endl;

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

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaStreamDestroy(stream);

    return 0;
}