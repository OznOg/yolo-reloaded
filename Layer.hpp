#pragma once

#include <Size.hpp>

#include <string>
#include <vector>

namespace yolo {

class Layer {
public:
    virtual void setInputFormat(const Size &s, size_t channels, size_t batch) = 0;
    virtual ~Layer() {}

    const Size &getOutputSize() {
	return _output_size;
    }

protected:
    void setOutputSize(const Size &s) {
	_output_size = s;
    }

private:
    Size _output_size;
};


enum class Activation {
    Elu,
    Hardtan,
    Leaky,
    Lhtan,
    Linear,
    Loggy,
    Logistic,
    Plse,
    Ramp,
    Relie,
    Relu,
    Stair,
    Tanh,
};

static inline Activation activationFromString(const std::string &str)
{
    if (str == "elu")      return Activation::Elu;
    if (str == "hardtan")  return Activation::Hardtan;
    if (str == "leaky")    return Activation::Leaky;
    if (str == "lhtan")    return Activation::Lhtan;
    if (str == "linear")   return Activation::Linear;
    if (str == "loggy")    return Activation::Loggy;
    if (str == "logistic") return Activation::Logistic;
    if (str == "plse")     return Activation::Plse;
    if (str == "ramp")     return Activation::Ramp;
    if (str == "relie")    return Activation::Relie;
    if (str == "relu")     return Activation::Relu;
    if (str == "stair")    return Activation::Stair;
    if (str == "tanh")     return Activation::Tanh;
    return Activation::Relu; // FIXME is that really useful? (Legacy)
}

class ConvolutionalLayer : public Layer {
public:
    ConvolutionalLayer(bool batch_normalize, size_t filters,
                       size_t size, size_t stride, size_t padding, Activation activation) :
         _batch_normalize(batch_normalize), _filters(filters), _size(size), 
	 _stride(stride), _padding(padding), _activation(activation), _weights(), _biases(filters), _bias_updates(filters) {}

    void setInputFormat(const Size &s, size_t channels, size_t batch) override {
	_input_size = s;
	_weights.resize(channels * _filters * _size * _size);
	_weights_updates.resize(_weights.size());

	setOutputSize((_input_size + 2 * _padding - _size) / _stride + 1);

	_output.resize(getOutputSize().width * getOutputSize().height * _filters * batch);

	_delta.resize(_output.size());

        _workspace_size = getOutputSize().width * getOutputSize().height * channels * _size * _size * sizeof(float);

	if (_batch_normalize) {
	    _scales.resize(_filters, 1.);
	    _scale_updates.resize(_filters);

	    _mean.resize(_filters);
	    _variance.resize(_filters);

	    _mean_delta.resize(_filters);
	    _variance_delta.resize(_filters);

	    _rolling_mean.resize(_filters);
	    _rolling_variance.resize(_filters);

	    _x.resize(_output.size());
	    _x_norm.resize(_output.size());
	}
    }

private:
    Size   _input_size;
    bool   _batch_normalize = false;
    size_t _filters = 1;
    size_t _size = 1;
    size_t _stride = 1;
    size_t _padding;
    Activation _activation = Activation::Leaky;
    size_t _workspace_size; // FIXME what is this used for?

    std::vector<float> _weights;
    std::vector<float> _weights_updates;

    std::vector<float> _biases;
    std::vector<float> _bias_updates;

    std::vector<float> _output;
    std::vector<float> _delta;


    // batch stuff
    // FIXME is that really usefull? would it be better to have a dedicated struct?
    std::vector<float> _scales;
    std::vector<float> _scale_updates;
    std::vector<float> _mean;
    std::vector<float> _variance;
    std::vector<float> _mean_delta;
    std::vector<float> _variance_delta;
    std::vector<float> _rolling_mean;
    std::vector<float> _rolling_variance;
    std::vector<float> _x; // FIXME WTF is x?
    std::vector<float> _x_norm; // FIXME WTF is x?

    // end of batch stuff
};

class MaxpoolLayer : public Layer {
public:
    MaxpoolLayer(size_t size, size_t stride, size_t padding) : _size(size), _stride(stride), _padding(padding) {}

    void setInputFormat(const Size &s, size_t channels, size_t batch) override {
	_input_size = s;
	_channels = channels;
	setOutputSize((_input_size + 2 * _padding) / _stride);

	size_t new_data_size = getOutputSize().width * getOutputSize().height * channels * batch;
	_indexes.resize(new_data_size);
	_output.resize(new_data_size);
	_delta.resize(new_data_size);
    }

private:
    Size   _input_size;
    size_t _channels;
    size_t _size;
    size_t _stride;
    size_t _padding;

    std::vector<size_t> _indexes;
    std::vector<float>  _output;
    std::vector<float>  _delta;

};

}
