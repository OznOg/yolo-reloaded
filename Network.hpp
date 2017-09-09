
#include <Layer.hpp>

#include <memory>
#include <cmath>
#include <random>
#include <vector>

namespace yolo {

class Policy {
public:
    virtual float get_current_rate_ratio(size_t current_batch) const = 0;
    virtual ~Policy() {}
};


class ConstantPolicy : public Policy {
public:
    float get_current_rate_ratio(size_t /* current_batch */) const override {
	return 1; //legacy value
    }
};

class StepPolicy : public Policy {
public:
    float get_current_rate_ratio(size_t current_batch) const override {
	 return std::pow(_scale, current_batch / _step);
    }
private:
    int _step = 1;
    float _scale = 1.;
};

class StepsPolicy : public Policy {
public:
    float get_current_rate_ratio(size_t current_batch) const override {
	float ratio = 1;
	for (const auto &step : _steps) {
	    if (step.rank > current_batch)
		return ratio;
	    ratio *= step.scale;
	}
	return ratio;
    }

    struct Step {
	size_t rank;
	float scale;
    };
private:
    std::vector<Step> _steps;
};

class ExpPolicy : public Policy {
public:
    float get_current_rate_ratio(size_t current_batch) const override {
	return std::pow(_gamma, current_batch);
    }
private:
    float _gamma = 1;
};

class PolyPolicy : public Policy {
public:
    float get_current_rate_ratio(size_t /*current_batch*/) const override {
	throw "Don't know how to implement this...";
    }
};

class RandomPolicy : public Policy {
public:
    float get_current_rate_ratio(size_t /* current_batch */) const override {
	static std::random_device rd;
	static std::mt19937 gen(rd());
	throw "Not correclty implemented";
	return std::pow(std::uniform_real_distribution<>(0., 1.)(gen), 1 /* FIXME what to do of this? net.power */);
    }
};

class SigPolicy : public Policy {
public:
    float get_current_rate_ratio(size_t current_batch) const override {
	return 1. / (1. + std::exp(_gamma * (current_batch - _step)));
    }
private:
    int _step = 1;
    float _gamma = 1;
};

class Network {
public:

private:
    std::vector<std::unique_ptr<Layer>> _layers;
    int _batch = 1;
    size_t _subdivisions = 1;
    size_t _width = 0;
    size_t _height = 0;
    size_t _channels = 0;
    float _momentum = .9;
    float _decay = .0001;
    float _angle = 0;
    float _saturation = 1;
    float _exposure = 1;
    float _hue = 0;

    float _learning_rate = .001;
    size_t _burn_in = 0;
    size_t _max_batches = 0;

    std::unique_ptr<Policy> _policy;
};

}