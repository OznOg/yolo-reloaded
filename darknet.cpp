#include <NetworkFactory.hpp>

#include <exception>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace yolo;


static Box correctScale(const Box &box, size_t imw, size_t imh, size_t netw, size_t neth)
{
    size_t new_w;
    size_t new_h;
    if ((float)netw / imw < (float)neth / imh) {
        new_w = netw;
        new_h = (imh * netw) / imw;
    } else {
        new_h = neth;
        new_w = (imw * neth) / imh;
    }

    Box b;
    b.x = (box.x - (netw - new_w) / 2. / netw) / ((float)new_w / netw);
    b.y = (box.y - (neth - new_h) / 2. / neth) / ((float)new_h / neth);
    b.w = box.w / ((float)new_w / netw);
    b.h = box.h / ((float)new_h / neth);

    return b;
}

static auto readClassNames(const std::string &nameFile) {
    std::map<size_t, std::string> class2str;

    auto nameStream = std::ifstream(nameFile);

    size_t classIdx = 0;
    std::string name;
    while (std::getline(nameStream, name)) {
        class2str[classIdx] = name;
        classIdx++;
    }
    return class2str;
}

bool run_detect(const std::vector<std::string> &args) {
    if (args.size() != 4) {
	std::cerr << "usage: detect needs at least 4 parameters.\n"
                  << "  ex:  ./darknet detect coco.names yolo.cfg yolo.weights InputImage.jpg" << std::endl << std::endl;
        return false;
    }

    auto net = NetworkFactory().createFromFile(args[1]);

    auto class2name = readClassNames(args[0]);

    auto weightsFile = std::ifstream(args[2]);

    if (!weightsFile)
        return false;
    net->loadWeights(weightsFile);

    // FIXME not implemented yet... not sure what it is needed for...
    // net->set_batch_network(1);

    std::string file_name = args[3];
    cv::Mat imageInteger = cv::imread(file_name);
    if (imageInteger.empty()) {
        std::cerr << "Could not load image '" << file_name << "'\n";
        return false;
    }

    cv::Mat imageFloat(imageInteger.size(), CV_32F);

    imageInteger.convertTo(imageFloat, CV_32F);

    imageFloat /= 255.;

    cv::Mat resized;
    double resize_factor_x = ((double)net->_input_size.width) / imageFloat.cols;
    double resize_factor_y = ((double)net->_input_size.height) / imageFloat.rows;

    auto min = std::min(resize_factor_x, resize_factor_y);

    // resize image (keeping ratio) so that it fits the network input
    cv::resize(imageFloat, resized, cv::Size(0, 0), min, min);

    cv::Mat letterbox(cv::Size(net->_input_size.width, net->_input_size.height), resized.type());
    size_t distance_to_top  = (letterbox.rows - resized.rows) / 2;
    size_t distance_to_side = (letterbox.cols - resized.cols) / 2;

    // resized image is copied into a network expected size, adding grey borders
    copyMakeBorder(resized, letterbox, distance_to_top, distance_to_top,
                   distance_to_side, distance_to_side,
                   cv::BORDER_CONSTANT, cv::Scalar(0.5, 0.5, 0.5));

    // FIXME the predict function expects channels to be separated (full image
    // R, full image G, full image B) but openCV stores the image with channels
    // interleaved.
    // Code next copies each channels in an array... quite under optimal, but I
    // need to go deeper in prediction algo to see if it would be possible to
    // pass to openCV Mat as is.
    cv::Mat bgr[3];
    cv::split(letterbox,bgr);

    std::vector<float> array;
    for (ssize_t channel = 0; channel < 3; channel++) {
        cv::Mat &mat = bgr[channel];
        if (mat.isContinuous()) {
            array.insert(array.end(), (float*)mat.datastart, (float*)mat.dataend);
        } else {
            for (int i = 0; i < mat.rows; ++i) {
                array.insert(array.end(), mat.ptr<float>(i), mat.ptr<float>(i)+mat.cols);
            }
        }
    }

    auto predictions = net->predict(array, 0.6 /* FIXME threshold hardcoded */);

    for (const auto &p : predictions) {
        std::cout << p.prob << " box @" << p.box.x << " " << p.box.y << " class="
                  << class2name[p.classIndex] << std::endl;
        const Box &b = correctScale(p.box, imageInteger.cols, imageInteger.rows, net->_input_size.width, net->_input_size.height);

        int left  = (b.x - b.w / 2.) * imageInteger.cols;
        int right = (b.x + b.w / 2.) * imageInteger.cols;
        int top   = (b.y - b.h / 2.) * imageInteger.rows;
        int bot   = (b.y + b.h / 2.) * imageInteger.rows;

        cv::Scalar color(b.h * 155, b.y * 155, b.x * 155);
        cv::putText(imageInteger, class2name[p.classIndex], cv::Point(left, top),
                    cv::FONT_HERSHEY_COMPLEX, 1, color, 2);
        cv::rectangle(imageInteger, cv::Point(left, top), cv::Point(right, bot), color, 2);
    }
    cv::imshow("Predictions", imageInteger);
    cv::waitKey();

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
