#include <NetworkFactory.hpp>

#include <exception>
#include <iostream>
#include <string>
#include <vector>

using namespace yolo;

bool run_detect(const std::vector<std::string> &args) {
    if (args.size() < 2) {
	std::cerr << "usage: detect needs at least 2 parameters." << std::endl << std::endl;
        return false;
    }

    auto net = NetworkFactory().createFromFile(args[0]);

    auto weightsFile = std::ifstream(args[1]);

    if (!weightsFile)
        return false;
    net->loadWeights(weightsFile);

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
