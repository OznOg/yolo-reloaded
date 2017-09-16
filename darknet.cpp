#include <NetworkFactory.hpp>

#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace yolo;

bool run_detect(const std::vector<std::string> &args) {
    if (args.size() < 3) {
	std::cerr << "usage: detect needs at least 3 parameters." << std::endl << std::endl;
        return false;
    }

    auto net = NetworkFactory().createFromFile(args[0]);

    auto weightsFile = std::ifstream(args[1]);

    if (!weightsFile)
        return false;
    net->loadWeights(weightsFile);

    // FIXME not implemented yet... not sure what it is needed for...
    // net->set_batch_network(1);

    std::string file_name = args[2];
    cv::Mat image = cv::imread(file_name);
    if (image.empty()) {
        std::cerr << "Could not load image '" << file_name << "'\n";
        return false;
    }

    cv::Mat resized;
    double resize_factor_x = ((double)net->_input_size.width) / image.cols;
    double resize_factor_y = ((double)net->_input_size.height) / image.rows;

    auto min = std::min(resize_factor_x, resize_factor_y);

    // resize image (keeping ratio) so that it fits the network input
    cv::resize(image, resized, cv::Size(0, 0), min, min);

    cv::Mat letterbox(cv::Size(net->_input_size.width, net->_input_size.height), image.type());
    size_t distance_to_top  = (letterbox.rows - resized.rows) / 2;
    size_t distance_to_side = (letterbox.cols - resized.cols) / 2;

    // resized image is copied into a network expected size, adding grey borders
    copyMakeBorder(resized, letterbox, distance_to_top, distance_to_top,
                   distance_to_side, distance_to_side,
                   cv::BORDER_CONSTANT, cv::Scalar(0x7F, 0x7F, 0x7F));

    cv::Mat floatInput;
    letterbox.convertTo(floatInput, CV_32F);

    return true;
}


int main(int argc, char **argv) {
    std::vector<std::string> args(argv, argv + argc);

    if (args.size() < 2) {
	std::cerr << "usage: " << args[0] << " <function>" << std::endl << std::endl;
	return 1;
    }

    if (args[1] != "detect") {
	std::cerr << "Error: Only detector is supported as function for now." << std::endl << std::endl;
	return 2;
    }

    try {
	if (!run_detect(std::vector<std::string>(&args[2], &args[args.size()]))) {
	    std::cerr << "Error: Detection failed." << std::endl << std::endl;
	    return 3;
	}
    } catch (std::exception &e) {
	std::cerr << "Failed:" << e.what() << std::endl;
    }

    return 0;
}
