#include "Layer.hpp"
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Expect 3 args: expected_in expected_out weights_file" << std::endl;
        return 1;
    }


    std::ifstream in(argv[1]);
    std::ifstream out(argv[2]);

    if (!in || !out) {
        std::cerr << "cannot open " << (!in ? "input " : " ") << (!out ? "output " : " ") << "file." << std::endl;
        return 2;
    }

    std::vector<float> input(608 * 608 * 3);
    in.read((char *)&input[0], input.size() * sizeof(float));

    std::vector<float> expected_output(11829248 / sizeof(float));
    out.read((char *)&expected_output[0], expected_output.size() * sizeof(float));


    yolo::ConvolutionalLayer cl(true, 32, 3, 1, 1, yolo::Activation::Leaky);

    cl.setInputFormat(yolo::Format(608, 608, 3, 1));

    yolo::MaxpoolLayer ml(2, 2, 0);

    ml.setInputFormat(yolo::Format(608, 608, 32, 1));

    auto weightsFile = std::ifstream(argv[3]);

    if (!weightsFile) {
        std::cerr << "Cannot open weights file" << std::endl;
        return 3;
    }

    int major, minor, revision;
    weightsFile.read((char *)&major, sizeof(major))
                .read((char *)&minor, sizeof(minor))
                .read((char *)&revision, sizeof(revision));

    size_t _seen = 0;
    if ((major * 10 + minor) >= 2) {
        weightsFile.read((char *)_seen, sizeof(_seen));
    } else {
        int iseen = 0;
        weightsFile.read((char *)&iseen, sizeof(iseen));
        _seen = iseen;
    }

    cl.loadWeights(weightsFile);
    ml.loadWeights(weightsFile);

    const std::vector<float> &output = ml.forward(cl.forward(input));

    size_t idx = 0;
    for (const auto &f : expected_output) {
        if (f != output[idx]) {
            std::cerr << "vector do not match @" << idx << std::endl;
            return 4;
        }
        idx++;
    }

    std::cout << "Success" << std::endl;
    return 0;
}
