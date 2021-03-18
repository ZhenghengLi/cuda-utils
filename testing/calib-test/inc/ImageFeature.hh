#ifndef __ImageFeature_H__
#define __ImageFeature_H__

#include <msgpack.hpp>

struct ImageFeature {
    float global_rms;
    int peak_counts;
    MSGPACK_DEFINE_MAP(global_rms, peak_counts)
};

#endif