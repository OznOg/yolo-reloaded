#pragma once

#include <Size.hpp>
#include "gemm.hpp"
#include "Prediction.hpp"

#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <list>
#include <map>

extern "C" {
#include "blas.h"
}

namespace yolo {

struct Format {
    Format(size_t w, size_t h, size_t c, size_t b) : width(w), height(h), channels(c), batch(b) {}
    Format() {}
    size_t width = 0;
    size_t height = 0;
    size_t channels = 0;
    size_t batch = 1;
};

struct LayerData {
    LayerData(const Format &format) : _format(format),
              _data(format.width * format.height * format.channels * format.batch) {}
    LayerData() {}

    Format _format;
    std::vector<float> _data;
};


class Layer {
public:
    virtual void setInputFormat(const Format &format) = 0;

    virtual std::string getName() const = 0;

    virtual ~Layer() {}

    const Size getOutputSize() const { // FIXME deprecated
       return Size(_output._format.width, _output._format.height);
    }

    const auto &getOutputChannels() const { // FIXME deprecated
       return _output._format.channels;
    }

    const auto &getOutput() const {
        return _output._data;
    }

    const auto &getOutputFormat() const {
        return _output._format;
    }

    virtual void loadWeights(std::istream &in) = 0;

    virtual const std::vector<float> &forward(const std::vector<float> &) {
        throw std::invalid_argument("Forward not implemented for layer " + getName());
    }
protected:
    Format _input_fmt;
    LayerData _output;
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

static inline float logistic_activate(float x) { return 1. / (1. + exp(-x)); }
static inline float leaky_activate(float x) { return x > 0 ? x : .1 * x; }
static inline float linear_activate(float x) { return x; }

static inline float (*getActivationMethode(Activation a))(float)
{
    switch (a) {
        case Activation::Logistic:
            return logistic_activate;
        case Activation::Leaky:
            return leaky_activate;
        case Activation::Linear:
            return linear_activate;

        case Activation::Elu:
        case Activation::Hardtan:
        case Activation::Lhtan:
        case Activation::Loggy:
        case Activation::Plse:
        case Activation::Ramp:
        case Activation::Relie:
        case Activation::Relu:
        case Activation::Stair:
        case Activation::Tanh:
            throw std::invalid_argument("Activation Not handled");
    }
    return nullptr;
}

static inline void activate_array(Activation a, float *x, size_t n) {
    const auto &activate = getActivationMethode(a);
    for (size_t i = 0; i < n; ++i) {
        x[i] = activate(x[i]);
    }
}

static inline void add_bias(float *output, const float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

static inline void scale_bias(float *output, const float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

class ConnectedLayer : public Layer {
public:
    ConnectedLayer(bool batch_normalize, size_t outputs, Activation activation) :
         _batch_normalize(batch_normalize), _outputs(outputs), _activation(activation), _weights(), _biases(outputs) { }

    void setInputFormat(const Format &format) override {
	_input_fmt = Format(1, 1, format.width * format.height * format.channels, format.batch);

        _output = LayerData(Format(1, 1, _outputs, format.batch));

	_weights.resize(_outputs * format.width * format.height * format.channels);

        if (_batch_normalize){
            _scales.resize(_outputs, 1.);
            _rolling_mean.resize(_outputs);
            _rolling_variance.resize(_outputs);
        }

    }

    std::string getName() const override {
        return "Connected";
    }

    void loadWeights(std::istream &in) override {
        in.read((char *)&_biases[0], _biases.size() * sizeof(float));
        in.read((char *)&_weights[0], _weights.size() * sizeof(float));

        if (_batch_normalize /* FIXME not used yet && (!l.dontloadscales) */ ) {
            in.read((char *)&_scales[0], _scales.size() * sizeof(float))
              .read((char *)&_rolling_mean[0], _rolling_mean.size() * sizeof(float))
              .read((char *)&_rolling_variance[0], _rolling_variance.size() * sizeof(float));
        }
    }

    const std::vector<float> &forward(const std::vector<float> &input) override {

        int m = _input_fmt.batch;
        int k = _input_fmt.channels * _input_fmt.height * _input_fmt.width;
        int n = _outputs;
        const float *a = &input[0];
        const float *b = &_weights[0];
        float *c = &_output._data[0];

        gemm<false, true>(m, n, k, 1, a, k, b, k, 1, c, n);

        if (_batch_normalize) {
            normalize_cpu(c, &_rolling_mean[0], &_rolling_variance[0], _output._format.batch, _output._format.channels, n);
            scale_bias(c, &_scales[0], _output._format.batch, _output._format.channels, n);
        }
        add_bias(c, &_biases[0], m, n, 1);

        activate_array(_activation, &_output._data[0], _output._data.size());

        return _output._data;
    }
private:
    bool _batch_normalize;
    size_t _outputs;
    Activation _activation = Activation::Leaky;
    std::vector<float> _weights;
    std::vector<float> _biases;
    std::vector<float> _scales;
    std::vector<float> _rolling_mean;
    std::vector<float> _rolling_variance;
};

class ConvolutionalLayer : public Layer {
public:
    ConvolutionalLayer(bool batch_normalize, size_t filters,
                       size_t size, size_t stride, size_t padding, Activation activation) :
         _batch_normalize(batch_normalize), _size(size),
	 _stride(stride), _padding(padding), _filters(filters), _activation(activation), _weights(), _biases(filters), _bias_updates(filters) { }

    void setInputFormat(const Format &format) override {
	_input_fmt = format;

        size_t width  = (_input_fmt.width + 2 * _padding - _size) / _stride + 1;
        size_t height = (_input_fmt.height + 2 * _padding - _size) / _stride + 1;
        _output = LayerData(Format(width, height, _filters, format.batch));

	_weights.resize(format.channels * _filters * _size * _size);
	_weights_updates.resize(_weights.size());

	if (_batch_normalize) {
	    _scales.resize(_filters, 1.);
	    _scale_updates.resize(_filters);

	    _mean.resize(_filters);
	    _variance.resize(_filters);

	    _mean_delta.resize(_filters);
	    _variance_delta.resize(_filters);

	    _rolling_mean.resize(_filters);
	    _rolling_variance.resize(_filters);

            const Format &f = _output._format;
	    _x.resize(f.width * f.height * f.channels * f.batch);
	    _x_norm.resize(_x.size());
	}
    }

    std::string getName() const override {
        return "Convolutional";
    }

    void loadWeights(std::istream &in) override {
        in.read((char *)&_biases[0], _biases.size() * sizeof(float));

        if (_batch_normalize /* FIXME not used yet && (!l.dontloadscales) */ ) {
            in.read((char *)&_scales[0], _scales.size() * sizeof(float))
              .read((char *)&_rolling_mean[0], _rolling_mean.size() * sizeof(float))
              .read((char *)&_rolling_variance[0], _rolling_variance.size() * sizeof(float));
        }
        in.read((char *)&_weights[0], _weights.size() * sizeof(float));
    }

    static inline float im2col_get_pixel(const std::vector<float> &im, int height, int width, int /*channels*/,
            int row, int col, int channel, int pad)
    {
        row -= pad;
        col -= pad;

        if (row < 0 || col < 0 || row >= height || col >= width)
            return 0; // missing pixels are extrapolated by black pixels

        return im[col + width * (row + height * channel)];
    }

    //From Berkeley Vision's Caffe!
    //https://github.com/BVLC/caffe/blob/master/LICENSE
    void im2col_cpu(const std::vector<float> &data_im,
            int channels,  int height,  int width,
            int ksize,  int stride, int pad, float *data_col)
    {
        int height_col = (height + 2*pad - ksize) / stride + 1;
        int width_col =  (width  + 2*pad - ksize) / stride + 1;

        int channels_col = channels * ksize * ksize;

        for (int c = 0; c < channels_col; ++c) {

            int w_offset = c % ksize;

            int h_offset = (c / ksize) % ksize;

            int c_im = c / ksize / ksize;

            for (int h = 0; h < height_col; ++h) {
                for (int w = 0; w < width_col; ++w) {

                    int im_row = h_offset + h * stride;
                    int im_col = w_offset + w * stride;

                    int col_index = (c * height_col + h) * width_col + w;

                    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                            im_row, im_col, c_im, pad);
                }
            }
        }
    }

    const std::vector<float> &forward(const std::vector<float> &input) override {

        int m = _filters;
        int k = _input_fmt.channels * _size * _size;
        int n = _output._format.width * _output._format.height;;

        float *temp = new float[n * k];

        im2col_cpu(input, _input_fmt.channels, _input_fmt.height, _input_fmt.width, _size, _stride, _padding, temp);

        float *a = &_weights[0];
        float *b = &temp[0];
        float *c = &_output._data[0];

        gemm<false, false>(m, n, k, 1, a, k, b, n, 1, c, n);
        delete[] temp;

        if (_batch_normalize) {
            _x = _output._data;
            normalize_cpu(c, &_rolling_mean[0], &_rolling_variance[0], _output._format.batch, _output._format.channels, n);
            scale_bias(c, &_scales[0], _output._format.batch, _output._format.channels, n);
        }
        add_bias(c, &_biases[0], _output._format.batch, _output._format.channels, n);

        activate_array(_activation, c, m * n * _output._format.batch);

        return _output._data;
    }
private:
    bool   _batch_normalize = false;
    size_t _size = 1;
    size_t _stride = 1;
    size_t _padding;
    size_t _filters;
    Activation _activation = Activation::Leaky;

    std::vector<float> _weights;
    std::vector<float> _weights_updates;

    std::vector<float> _biases;
    std::vector<float> _bias_updates;

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

    void setInputFormat(const Format &format) override {
	_input_fmt = format;
        size_t width  = (_input_fmt.width + 2 * _padding) / _stride;
        size_t height = (_input_fmt.height + 2 * _padding) / _stride;

        _output = LayerData(Format(width, height, _input_fmt.channels, _input_fmt.batch));

	_indexes.resize(_output._data.size());
	_delta.resize(_output._data.size());
    }

    std::string getName() const override {
        return "Maxpool";
    }

    void loadWeights(std::istream &in) override {
        (void)in;
    }

    const std::vector<float> &forward(const std::vector<float> &input) override {
        int w_offset = -_padding;
        int h_offset = -_padding;

        ssize_t h = getOutputSize().height;
        ssize_t w = getOutputSize().width;
        const auto &c = getOutputChannels();

        for (size_t b = 0; b < _output._format.batch; ++b) {
            for (size_t k = 0; k < c; ++k) {
                for (ssize_t i = 0; i < h; ++i) {
                    for (ssize_t j = 0; j < w; ++j) {

                        int out_index = j + w*(i + h*(k + c*b));

                        float max = std::numeric_limits<float>::lowest();

                        int max_i = -1;

                        for (size_t n = 0; n < _size; ++n) {
                            for (size_t m = 0; m < _size; ++m) {
                                ssize_t cur_h = h_offset + i*_stride + n;
                                ssize_t cur_w = w_offset + j*_stride + m;

                                int valid = (cur_h >= 0 && cur_h < (ssize_t)_input_fmt.height
                                          && cur_w >= 0 && cur_w < (ssize_t)_input_fmt.width);

                                if (valid) {
                                    int index = cur_w + _input_fmt.width * (cur_h + _input_fmt.height * (k + b * _input_fmt.channels));
                                    float val = input[index];
                                    max_i = (val > max) ? index : max_i;
                                    max   = std::max(val, max);
                                }
                            }
                        }
                        _output._data[out_index] = max;
                        _indexes[out_index] = max_i;
                    }
                }
            }
        }

        return _output._data;
    }
private:
    Size   _input_size;
    size_t _size;
    size_t _stride;
    size_t _padding;
    size_t _batch = 1;

    std::vector<size_t> _indexes;
    std::vector<float>  _delta;

};

class RouteLayer : public Layer {
public:
    RouteLayer(const std::vector<Layer *> input_layers) : _input_layers(input_layers) {
	size_t outputs = 0;
	for (const auto *layer : input_layers){
	    outputs += layer->getOutput().size();
	}
	_delta.resize(outputs);
	_output._data.resize(outputs);;


	auto &first = *input_layers[0];
	Size outputSize = first.getOutputSize();
	size_t outChannels = first.getOutputChannels();

	for (const auto *next : input_layers) {
	    if (&first == next)
		continue; // skip first one as it was used for init

	    if(next->getOutputSize() == first.getOutputSize()) {
		outChannels += next->getOutputChannels();
	    } else {
		outputSize = Size(0, 0);
		outChannels = 0;
	    }
	}
	_output = LayerData(Format(outputSize.width, outputSize.height, outChannels, 1));
    }

    void setInputFormat(const Format &f) override {
        // FIXME having this empty really enforce the fact that route is not a layer...
        // need more investigation in order to rework that properly
        _output = LayerData(Format(_output._format.width, _output._format.height, _output._format.channels, f.batch));
    }

    std::string getName() const override {
        return "Route";
    }

    void loadWeights(std::istream &in) override {
        (void)in;
    }

    const std::vector<float> &forward(const std::vector<float> &/* input */) override {
        size_t offset = 0;
        for (const auto *layer : _input_layers) {
            const auto &input = layer->getOutput();
            for (size_t j = 0; j < _output._format.batch; ++j) {
                std::memcpy(&_output._data[offset + j * _output._data.size()], &input[j * input.size()], input.size() * sizeof(float));
            }
            offset += input.size();
        }
        return _output._data;
    }
private:
    std::vector<Layer *> _input_layers;
    std::vector<float>   _delta;
};

class ReorgLayer : public Layer {
public:
    ReorgLayer(size_t stride, bool reverse, bool flatten, bool extra) :
        _stride(stride), _reverse(reverse), _flatten(flatten), _extra(extra) {}

    void setInputFormat(const Format &input_fmt) override {
        _input_fmt = input_fmt;
        size_t channels;
        size_t width;
        size_t height;

        if (_reverse) {
            width =_input_fmt.width * _stride;
            height =_input_fmt.height * _stride;
            channels = input_fmt.channels / (_stride * _stride);
        } else {
            width =_input_fmt.width / _stride;
            height =_input_fmt.height / _stride;
            channels = input_fmt.channels * (_stride * _stride);
        }

        if (_extra) {
            // this extra stuff looks like a crappy hack
            _output = LayerData(Format(0, 0, 0, _input_fmt.batch));
            _output._data.resize((_input_fmt.width * _input_fmt.height * _input_fmt.channels + _extra) * _input_fmt.batch);
        } else {
            _output = LayerData(Format(width, height, channels, _input_fmt.batch));
        }

        _delta.resize(_output._data.size());
    }

    std::string getName() const override {
        return "Reorg";
    }

    void loadWeights(std::istream &in) override {
        (void)in;
    }

    const std::vector<float> &forward(const std::vector<float> &input) override {
        if (_flatten) {
#if 0
            memcpy(_output, _input, l.outputs*l.batch*sizeof(float));
            flatten(l.output, l.w*l.h, l.c, l.batch, !_reverse);
#endif
            throw std::invalid_argument("Reorg case not implemented (yet).");
        } else if (_extra) {
#if 0
            for (size_t i = 0; i < _batch; ++i){
                copy_cpu(l.inputs, net.input + i*l.inputs, 1, l.output + i*l.outputs, 1);
            }
#endif
            throw std::invalid_argument("Reorg case not implemented (yet).");
        } else {
            reorg_cpu(&input[0], _input_fmt.width, _input_fmt.height, _input_fmt.channels, _input_fmt.batch, _stride, _reverse, &_output._data[0]);
        }
        return _output._data;
    }
private:
    std::vector<float>   _delta;
    size_t _stride;
    bool _reverse;
    bool _flatten;
    bool _extra;
};

class DetectionLayer : public Layer {
public:
    DetectionLayer (size_t num, int classes, int coords, size_t side, bool softmax) : _num(num), _classes(classes), _coords(coords), _side(side), _softmax(softmax) {}

    void setInputFormat(const Format &format) override {
        _input_fmt = Format(_side, _side, _classes + _num * (_coords + 1), format.batch);

        _output = LayerData(_input_fmt);
    }

    std::string getName() const override {
        return "Detection";
    }

    void loadWeights(std::istream &in) override {
        (void)in;
    }

    const std::vector<float> &forward(const std::vector<float> &input) override {
        size_t locations = _side * _side;

        _output._data = input;

        if (_softmax) {
            for (size_t b = 0; b < _input_fmt.batch; ++b) {
                size_t index = b * input.size();
                for (size_t offset = 0; offset < locations * _classes; offset += _classes) {
                    softmax(&input[index + offset], _classes, 1, 1,
                            &_output._data[index + offset]);
                }
            }
        }
        return _output._data;
    }

    auto get_boxes(float thresh) const {
        std::vector<Prediction> predictions;

        const float *data = _output._data.data();

        for (size_t i = 0; i < _side * _side; ++i) {
            size_t row = i / _side;
            size_t col = i % _side;

            for (size_t n = 0; n < _num; ++n) {
                int p_index = _side * _side * _classes + i * _num + n;
                float scale = data[p_index];
                int box_index = _side * _side * (_classes + _num) + (i * _num + n) * 4;

                Prediction prediction;
                prediction.box.x = (data[box_index + 0] + col) / _side;
                prediction.box.y = (data[box_index + 1] + row) / _side;
                prediction.box.w = pow(data[box_index + 2], (_sqrt ? 2 : 1));
                prediction.box.h = pow(data[box_index + 3], (_sqrt ? 2 : 1));

                prediction.prob = 0;
                for (size_t j = 0; j < _classes; ++j) {
                    int class_index = i * _classes;
                    float prob = scale * data[class_index + j];
                    if (prediction.prob < prob) {
                        prediction.prob = prob;
                        prediction.classIndex = j;
                    }
                }
		predictions.push_back(prediction);
            }
        }
        return Prediction::reduce(predictions, thresh);
    }

private:
    size_t _num;
    size_t _classes;
    int    _coords;
    size_t _side;
    bool   _softmax;
    bool _sqrt = true;
};

class RegionLayer : public Layer {
public:
    RegionLayer(size_t num, int classes, int coords, size_t side, bool softmax, bool background, const std::vector<float> &biases) :
        _num(num), _classes(classes), _coords(coords), _side(side), _softmax(softmax), _background(background), _biases(biases.begin(), biases.end()) {}

    void setInputFormat(const Format &format) override {
        _input_fmt = Format(format.width, format.height, _num * (_classes + _coords + 1), format.batch);

        _output = LayerData(_input_fmt);

        _biases.resize(format.channels * 2, .5);
        _bias_updates.resize(format.channels * 2);
        _delta.resize(_output._data.size());
    }

    std::string getName() const override {
        return "Region";
    }

    void loadWeights(std::istream &in) override {
        (void)in;
    }

    inline int entry_index(size_t batch, size_t location, size_t entry) const {
        size_t wh = _input_fmt.width * _input_fmt.height;
        size_t n =   location / wh;
        size_t loc = location % wh;
        size_t one_size = _output._format.width * _output._format.height * _output._format.channels;

        return batch * one_size + n * wh * (_coords + _classes + 1) + entry * wh + loc;
    }

    Box get_region_box(const std::vector<float> &x, const std::vector<float> &biases,
                       int n, int index, int i, int j, int w, int h, int stride) const {
        Box b;
        b.x = (i + x[index + 0 * stride]) / w;
        b.y = (j + x[index + 1 * stride]) / h;
        b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
        b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
        return b;
    }

    auto get_region_boxes(float thresh) const {
        std::vector<Prediction> predictions;

        auto w = _input_fmt.width;
        auto h = _input_fmt.height;
        for (size_t i = 0; i < w * h; ++i) {
            int row = i / w;
            int col = i % w;
            for(size_t n = 0; n < _num; ++n) {
                Prediction prediction;

                int index = n * w * h + i;
                int obj_index  = entry_index(0, index, _coords);
                int box_index  = entry_index(0, index, 0);
                float scale = _background ? 1 : _output._data[obj_index];
                prediction.box = get_region_box(_output._data, _biases, n, box_index, col, row, w, h, w * h);

                prediction.prob = 0;
                for (size_t j = 0; j < _classes; ++j) {
                    auto class_index = entry_index(0, index, _coords + 1 + j);

                    float prob = scale * _output._data[class_index];
                    if (prob > prediction.prob) {
                        prediction.classIndex = j;
                        prediction.prob       = prob;
                    }
                }
		predictions.push_back(prediction);
            }
        }
        return Prediction::reduce(predictions, thresh);
    }

    const std::vector<float> &forward(const std::vector<float> &input) override {
        _output._data = input;

        size_t wh = _output._format.width * _output._format.height;
        for (size_t b = 0; b < _output._format.batch; ++b){
            for (size_t n = 0; n < _num; ++n){
                int index = entry_index(b, n * wh, 0);
                activate_array(Activation::Logistic, &_output._data[index], 2 * wh);
                index = entry_index(b, n * wh, _coords);
                if (!_background) {
                    activate_array(Activation::Logistic, &_output._data[index], wh);
                }
            }
        }
#if 0
// FIXME will need to be eventually implemented :)
        if (l.softmax_tree){

            int i;
            int count = l.coords + 1;
            for (i = 0; i < l.softmax_tree->groups; ++i) {
                int group_size = l.softmax_tree->group_size[i];
                softmax_cpu(net.input + count, group_size, l.batch, l.inputs, l.n*l.w*l.h, 1, l.n*l.w*l.h, l.temperature, l.output + count);
                count += group_size;
            }
        } else
#endif
        if (_softmax)
        {
            int index = entry_index(0, 0, _coords + !_background);
            softmax_cpu(&input[index], _classes + _background, _input_fmt.batch * _num,
                        _output._data.size() / _input_fmt.batch / _num, wh, 1, wh, 1, &_output._data[index]);
        }

        return _output._data;
    }
private:
    size_t _num;
    size_t _classes;
    int _coords;
    size_t _side;
    bool _softmax;
    bool _background;

    std::vector<float> _biases;
    std::vector<float> _bias_updates;
    std::vector<float> _delta;
};

}
