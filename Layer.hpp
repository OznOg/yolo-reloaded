#pragma once

#include <Size.hpp>

#include <string>
#include <vector>

namespace yolo {

class Layer {
public:
    virtual ~Layer() {}
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
	 _stride(stride), _padding(padding), _activation(activation), _weights() {}
private:
    Size   _input_size;
    bool   _batch_normalize = false;
    size_t _filters = 1;
    size_t _size = 1;
    size_t _stride = 1;
    size_t _padding;
    Activation _activation = Activation::Leaky;

    std::vector<float> _weights;
    std::vector<float> _weights_updates;

    std::vector<float> _biases;
    std::vector<float> _bias_updates;
};



}
