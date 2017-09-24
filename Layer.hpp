#pragma once

#include <Size.hpp>

#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <iterator>

extern "C" {
#include "gemm.h"
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

    float im2col_get_pixel(const std::vector<float> &im, int height, int width, int /*channels*/,
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
            int ksize,  int stride, int pad, std::vector<float> &data_col)
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

    void add_bias(float *output, const float *biases, int batch, int n, int size)
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

    void scale_bias(float *output, const float *scales, int batch, int n, int size)
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

    const std::vector<float> &forward(const std::vector<float> &input) override {
        _output._data.assign(1, _output._data.size()); //FIXME seems useless

        std::vector<float> temp(_output._format.width * _output._format.height * (_input_fmt.channels * _size * _size /* ksize */));

        im2col_cpu(input, _input_fmt.channels, _input_fmt.height, _input_fmt.width, _size, _stride, _padding, temp);

        int m = _filters;
        int k = _input_fmt.channels * _size * _size;
        int n = _output._format.width * _output._format.height;;


        float *a = &_weights[0];
        float *b = &temp[0];
        float *c = &_output._data[0];

        gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);

        if (_batch_normalize) {
            _x = _output._data;
            normalize_cpu(c, &_rolling_mean[0], &_rolling_variance[0], _output._format.batch, _output._format.channels, n);
            scale_bias(c, &_scales[0], _output._format.batch, _output._format.channels, n);
        }
        add_bias(c, &_biases[0], _output._format.batch, _output._format.channels, n);

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

#if 0
class MaxpoolLayer : public Layer {
public:
    MaxpoolLayer(size_t size, size_t stride, size_t padding) : _size(size), _stride(stride), _padding(padding) {}

    void setInputFormat(const Size &s, size_t channels, size_t batch) override {
	_input_size = s;
	setOutputSize((_input_size + 2 * _padding) / _stride);

        setOutputChannels(channels);

	size_t new_data_size = getOutputSize().width * getOutputSize().height * channels * batch;
	_indexes.resize(new_data_size);
	_output.resize(new_data_size);
	_delta.resize(new_data_size);
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

        for (size_t b = 0; b < _batch; ++b) {
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
                                int index = cur_w + w*(cur_h + h*(k + b* c));
                                bool valid = (cur_h >= 0 && cur_h < h && cur_w >= 0 && cur_w < w);
                                float val = valid ? input[index] : std::numeric_limits<float>::lowest();
                                max_i = (val > max) ? index : max_i;
                                max   = std::max(val, max);
                            }
                        }
                        _output[out_index] = max;
                        _indexes[out_index] = max_i;
                    }
                }
            }
        }

        return _output;
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
#endif

class RouteLayer : public Layer {
public:
    RouteLayer(const std::vector<Layer *> input_layers) :
        _input_layers(input_layers) {
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
            size_t input_size = layer->getOutputSize().width * layer->getOutputSize().height;
            for (size_t j = 0; j < _output._format.batch; ++j) {
                std::memcpy(&_output._data[offset + j * _output._data.size()], &input[j * input_size], input_size);
            }
            offset += input_size;
        }
        return _output._data;
    }
private:
    std::vector<Layer *> _input_layers;
    std::vector<float>   _delta;
};

#if 0
class ReorgLayer : public Layer {
public:
    ReorgLayer(size_t stride, bool reverse, bool flatten, bool extra) :
        _stride(stride), _reverse(reverse), _flatten(flatten), _extra(extra) {}

    void setInputFormat(const Size &s, size_t channels, size_t batch) override {
        _input_size = s;
        size_t output_channels;
        if (_reverse) {
            setOutputSize(_input_size * _stride);
            output_channels = channels / (_stride * _stride);
        } else {
            setOutputSize(_input_size / _stride);
            output_channels = channels * (_stride * _stride);
        }

	size_t new_data_size = getOutputSize().width * getOutputSize().height * output_channels;
        if (_extra) {
            setOutputSize(Size(0, 0));
            output_channels = 0;
            new_data_size = s.width * s.height * channels + _extra;
        }

        _input_channels = channels;

        new_data_size *= batch;
        _output.resize(new_data_size);
        _delta.resize(new_data_size);
        setOutputChannels(output_channels);
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
            reorg_cpu(&input[0], _input_size.width, _input_size.height, _input_channels, _batch, _stride, _reverse, &_output[0]);
        }
        return _output;
    }
private:
    std::vector<float>   _delta;
    Size   _input_size;
    size_t _input_channels;
    size_t _stride;
    bool _reverse;
    bool _flatten;
    bool _extra;
    size_t _batch = 1;
};

class RegionLayer : public Layer {
public:
    RegionLayer(int num, int classes, int coords, size_t side, bool softmax, const std::vector<float> &biases) :
        _num(num), _classes(classes), _coords(coords), _side(side), _softmax(softmax), _biases(biases.begin(), biases.end()) {}

    void setInputFormat(const Size &s, size_t channels, size_t batch) override {
        _input_size = s;
        _filters = channels * (_classes + _coords + 1);
        setOutputSize(_input_size);
        setOutputChannels(channels);

        _biases.resize(channels * 2, .5);
        _bias_updates.resize(channels * 2);
        _output.resize(getOutputSize().width * getOutputSize().height * channels * (_classes + _coords + 1) * batch);
        _delta.resize(_output.size());
    }

    std::string getName() const override {
        return "Detection";
    }

    void loadWeights(std::istream &in) override {
        (void)in;
    }

    const std::vector<float> &forward(const std::vector<float> &input) override {
        size_t locations = _side * _side;
        _output = input;
        if (_softmax){
            for(size_t b = 0; b < _batch; ++b){
                size_t index = b * _input_size.width * _input_size.height;
                for (size_t i = 0; i < locations; ++i) {
                    size_t offset = i * _classes;
                    softmax(&_output[index + offset], _classes, 1, 1,
                            &_output[index + offset]);
                }
            }
        }
        return _output;
    }
private:
    Size   _input_size;
    size_t _filters;
    int _num;
    int _classes;
    int _coords;
    size_t _side;
    size_t _softmax;
    size_t _batch = 1;

    std::vector<float> _biases;
    std::vector<float> _bias_updates;
    std::vector<float> _delta;
};
#endif

}
