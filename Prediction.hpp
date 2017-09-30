#pragma once

#include <set>
#include <functional>

namespace yolo {

struct Box {
    float x, y, w, h;
};

struct Prediction {
    Box box;
    float prob;
    size_t classIndex;
};

}
