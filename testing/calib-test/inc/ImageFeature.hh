#ifndef __ImageFeature_H__
#define __ImageFeature_H__

#ifndef __CUDACC__
#include <msgpack.hpp>
#endif

struct ImageFeature {
    float global_rms;
    int peak_counts;
#ifndef __CUDACC__
    MSGPACK_DEFINE_MAP(global_rms, peak_counts);
#endif
};

#endif