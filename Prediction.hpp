#pragma once

#include <set>
#include <functional>

namespace yolo {

struct Box {
    float x, y, w, h;

    float matchRatio(const Box &b) const {
        return operator&(b) / operator|(b);
    }
private:
    static inline float overlap(float x1, float w1, float x2, float w2) {
        float l1 = x1 - w1/2;
        float l2 = x2 - w2/2;
        float left = l1 > l2 ? l1 : l2;
        float r1 = x1 + w1/2;
        float r2 = x2 + w2/2;
        float right = r1 < r2 ? r1 : r2;
        return right - left;
    }

    float operator&(const Box &b) const {
        float ow = overlap(x, w, b.x, b.w);
        float oh = overlap(y, h, b.y, b.h);
        if (ow < 0 || oh < 0)
            return 0;
        float area = ow * oh;
        return area;
    }

    float operator|(const Box &b) const {
        float i = operator&(b);
        float u = w * h + b.w * b.h - i;
        return u;
    }

};

struct Prediction {
    Box box;
    float prob;
    size_t classIndex;
};

}
