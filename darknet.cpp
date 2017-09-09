#include <iostream>
#include <string>
#include <vector>


bool run_detect(const std::vector<std::string> &args) {
    if (args.size() < 2) {
	std::cerr << "usage: detect needs at least 2 parameters." << std::endl << std::endl;
        return false;
    }
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

    if (!run_detect(std::vector<std::string>(&args[2], &args[args.size()]))) {
	std::cerr << "Error: Detection failed." << std::endl << std::endl;
	return 3;
    }

    for (const auto &a : args)
	std::cout << a << std::endl;

    return 0;
}
