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

void drawDetection(cv::Mat& im, const std::string &label, const Box &box)
{
    int left  = (box.x - box.w / 2.) * im.cols;
    int right = (box.x + box.w / 2.) * im.cols;
    int top   = (box.y - box.h / 2.) * im.rows;
    int bot   = (box.y + box.h / 2.) * im.rows;

    const cv::Scalar color(box.h * 155, box.y * 155, box.x * 155);

    const int font = cv::FONT_HERSHEY_COMPLEX;
    const double scale = 1;
    const int thickness = 1;
    int baseline;

    cv::Size text = cv::getTextSize(label, font, scale, thickness, &baseline);

    cv::rectangle(im, cv::Point(left, top + baseline), cv::Point(left, top) + cv::Point(text.width, -text.height), color, CV_FILLED);

    cv::rectangle(im, cv::Point(left, top), cv::Point(right, bot), color, 2);

    cv::putText(im, label, cv::Point(left, top), font, 1, cv::Scalar(0, 0, 0), 2);
}

bool run_detect(const std::vector<std::string> &_args) {
    auto args = _args;

    if (args.size() < 4) {
	std::cerr << "usage: detect needs at least 4 parameters.\n"
                  << "  ex:  ./darknet detect [--thresh=percentage] coco.names yolo.cfg yolo.weights InputImage.jpg" << std::endl << std::endl;
        return false;
    }

    float threshold = 0.3;
    if (std::string(args[0].substr(0, 9)) == "--thresh=") {
        try {
            threshold = std::stoul(args[0].substr(9, std::string::npos)) / 100.;
        } catch (...) { }
        args.erase(args.begin()); // remove threshold switch from list of options
    }
    std::cout << "Using threshold=" << threshold * 100 << "%" << std::endl;

    auto net = NetworkFactory().createFromFile(args[1], false);

    auto class2name = readClassNames(args[0]);

    auto weightsFile = std::ifstream(args[2]);

    if (!weightsFile)
        return false;
    net->loadWeights(weightsFile);

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
    size_t distance_to_top    = (letterbox.rows - resized.rows) / 2;
    size_t distance_to_left   = (letterbox.cols - resized.cols) / 2;
    // Careful, their might be cases when distance_to_top != distance_to_bottom as
    // the (letterbox.rows - resized.rows) might be an odd number. Idem for
    // distance to bottom
    size_t distance_to_bottom = net->_input_size.height - resized.rows - distance_to_top;
    size_t distance_to_right  = net->_input_size.width - resized.cols - distance_to_left;

    // resized image is copied into a network expected size, adding grey borders
    copyMakeBorder(resized, letterbox, distance_to_top, distance_to_bottom,
                   distance_to_left, distance_to_right,
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

    auto predictions = net->predict(array, threshold);

    for (const auto &p : predictions) {
        std::cout << "box @ x=" << p.box.x << " y=" << p.box.y << " h=" << p.box.h << " w=" << p.box.w << " probability=" << p.prob << " class="
                  << class2name[p.classIndex] << "(" << p.classIndex << ")" << std::endl;
        const Box &b = correctScale(p.box, imageInteger.cols, imageInteger.rows, net->_input_size.width, net->_input_size.height);
        drawDetection(imageInteger, class2name[p.classIndex], b);
    }
    //cv::imshow("Predictions", imageInteger);
    //cv::waitKey();

    return true;
}


int main(int argc, char **argv) {
    std::vector<std::string> args(argv, argv + argc);

    if (args.size() < 2) {
	std::cerr << "usage: " << args[0] << " <function>" << std::endl << std::endl;
	return 1;
    }

    if (args[1] != "detect") {
	std::cerr << "Error: Only detect is supported as function for now." << std::endl << std::endl;
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
