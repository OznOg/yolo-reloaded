#pragma once

#include <functional>
#include <list>
#include <vector>

namespace yolo {

struct Box {
    float x, y, w, h;

    float matchRatio(const Box &b) const {
        return operator&(b) / operator|(b);
    }

    Box(float X, float Y, float W, float H) : x(X), y(Y), w(W), h(H) {}
    Box() : x(0), y(0), w(0), h(0) {}

    bool operator!=(const Box &aBox) const {
	return !operator==(aBox);
    }

    bool operator==(const Box &aBox) const {
	return aBox.x == x && aBox.y == y && aBox.h == h && aBox.w == w;
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

    bool operator==(const Prediction &aPred) const {
	return aPred.box == box && aPred.prob == prob && aPred.classIndex == classIndex;
    }

    bool operator!=(const Prediction &aPred) const {
	return !operator==(aPred);
    }

    static std::vector<Prediction> reduce(const std::vector<Prediction> &predictions, float min_probability = 0.6, std::function<bool(Box, Box)> box_comparator = [] (const Box &b1, const Box &b2) { return b1.matchRatio(b2) > 0.8; }) {
	std::list<Prediction> filtered_on_probability;

	for (const auto &pred : predictions) { // FIXME would be nice to use std::reduce
	    if (pred.prob >= min_probability)
		filtered_on_probability.push_back(pred);
	}

        std::vector<Prediction> reduced;

	while (!filtered_on_probability.empty()) {
	    const auto &first = filtered_on_probability.front();

	    auto it = std::next(std::begin(filtered_on_probability));
	    for ( ;it != std::end(filtered_on_probability);) {
                if (first.classIndex == it->classIndex
                    && box_comparator(first.box, it->box)) { // boxes are same class and pointing at the same zone
		    if (first.prob >= it->prob)
			it = filtered_on_probability.erase(it);
		    else {
			filtered_on_probability.push_back(first);
			filtered_on_probability.pop_front();
			break;
		    }
		} else
		    ++it;
	    }
	    if (it == std::end(filtered_on_probability)) {
		reduced.push_back(first);
		filtered_on_probability.pop_front();
	    }
	}

	return reduced;
    }
};

}
