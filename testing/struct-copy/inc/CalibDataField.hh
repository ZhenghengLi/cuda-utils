#ifndef __CalibDataField_H__
#define __CalibDataField_H__

#include "ImageDataField.hh"

struct CalibDataField {
    float pedestal[MOD_CNT][FRAME_H][FRAME_W];
    float gain[MOD_CNT][FRAME_H][FRAME_W];
};

#endif