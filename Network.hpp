#pragma once

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
    struct Step {
	size_t rank;
	float scale;
	Step(size_t aRank, float aScale) : rank(aRank), scale(aScale) {}
    };

    StepsPolicy(const std::vector<Step> &steps) : _steps(steps) {}

    float get_current_rate_ratio(size_t current_batch) const override {
	float ratio = 1;
	for (const auto &step : _steps) {
	    if (step.rank > current_batch)
		return ratio;
	    ratio *= step.scale;
	}
	return ratio;
    }

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

    int _batch = 1;
    size_t _subdivisions = 1;
    Size   _input_size; // FIXME name not really clear...
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

    void setPolicy(std::unique_ptr<Policy> policy) {
	_policy = std::move(policy);
    }

    void addRoute(const std::vector<int> &layers_idx) {
        std::vector<Layer *> route_layers; // FIXME use weak_ptr
        for (const auto &idx : layers_idx) {
            Layer *l;
            if (idx >= 0) {
                l =_layers[idx].get();
            } else { // negative indexes reference layers starting from back
                l =_layers[_layers.size() + idx].get();
            }
            route_layers.push_back(l);
        }
        addLayer(std::make_unique<RouteLayer>(route_layers));
    }

    void addLayer(std::unique_ptr<Layer> layer) {
	/* For the first layer we give the network input size; for next ones,
	 * each layer N gets the output size of the layer N - 1 */
	const auto &size = _layers.empty() ? _input_size : _layers.back()->getOutputSize();
	layer->setInputFormat(size, _channels, _batch);
	_layers.push_back(std::move(layer));
    }

private:
    std::vector<std::unique_ptr<Layer>> _layers;
    std::unique_ptr<Policy> _policy;
};

}
