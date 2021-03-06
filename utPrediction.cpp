
#include "Prediction.hpp"

#include "gtest/gtest.h"

using namespace yolo;

TEST(Box, matchRatio) {
    std::vector<Box> boxes = {
	{ Box(0.168703, 0.512449, 0.35275 , 0.303689) },
	{ Box(0.354891, 0.506814, 0.381235, 0.190528) },
	{ Box(0.51416 , 0.50779 , 0.386727, 0.199172) },
	{ Box(0.679756, 0.505654, 0.413894, 0.240368) },
	{ Box(0.687505, 0.504608, 0.401359, 0.246871) },
	{ Box(0.890024, 0.491709, 0.370772, 0.226073) },
	{ Box(0.897085, 0.4912  , 0.36957 , 0.215755) } };

    std::vector<float> matchRatios;

    for (const auto &b1 : boxes)
	for (const auto &b2 : boxes)
	    matchRatios.push_back(b1.matchRatio(b2));

    std::vector<float> expected_ratios = {          1,   0.237061426,  0.0269703474,            0,            0,             0,           0,
                                          0.237061426,    1.00000036,   0.400702536, 0.0875160471, 0.0696450546,             0,           0,
                                         0.0269703474,   0.400702536,             1,  0.360259622,  0.332622856, 0.00353827118,           0,
                                                    0,  0.0875160471,   0.360259622,   1.00000012,  0.938164413,   0.278426051, 0.262406796,
                                                    0,  0.0696450546,   0.332622856,  0.938164413,  0.999999821,   0.289252251, 0.269378036,
                                                    0,             0, 0.00353827118,  0.278426051,  0.289252251,    1.00000036,  0.91935128,
                                                    0,             0,             0,  0.262406796,  0.269378036,    0.91935128,           1 };

    EXPECT_EQ(expected_ratios, matchRatios);
}

TEST(Prediction, reduceBasic) {
    const std::vector<Prediction> predictions = {
	{ Box(0.168703, 0.512449, 0.35275 , 0.303689), 0.895, 2 }, // 1
	{ Box(0.354891, 0.506814, 0.381235, 0.190528), 0.734, 2 }, // 2
	{ Box(0.51416 , 0.50779 , 0.386727, 0.199172), 0.678, 2 }, // 3
	{ Box(0.679756, 0.505654, 0.413894, 0.240368), 0.811, 2 }, // 4
	{ Box(0.687505, 0.504608, 0.401359, 0.246871), 0.612, 2 }, // 5 => replaced by 4
	{ Box(0.890024, 0.491709, 0.370772, 0.226073), 0.784, 2 }, // 6
	{ Box(0.897085, 0.4912  , 0.36957 , 0.215755), 0.601, 2 }  // 7 => replaced by 6
    };

    auto red = Prediction::reduce(predictions);

    const std::vector<Prediction> expected_output = {
	{ Box(0.168703, 0.512449, 0.35275 , 0.303689), 0.895, 2 }, // 1
	{ Box(0.354891, 0.506814, 0.381235, 0.190528), 0.734, 2 }, // 2
	{ Box(0.51416 , 0.50779 , 0.386727, 0.199172), 0.678, 2 }, // 3
	{ Box(0.679756, 0.505654, 0.413894, 0.240368), 0.811, 2 }, // 4
	{ Box(0.890024, 0.491709, 0.370772, 0.226073), 0.784, 2 }, // 6
    };

    EXPECT_EQ(expected_output.size(), red.size());
    EXPECT_EQ(expected_output, red);
}

TEST(Prediction, reduceSetThreshold) {
    const std::vector<Prediction> predictions = {
	{ Box(0.168703, 0.512449, 0.35275 , 0.303689), 0.895, 2 }, // 1
	{ Box(0.354891, 0.506814, 0.381235, 0.190528), 0.734, 2 }, // 2
	{ Box(0.51416 , 0.50779 , 0.386727, 0.199172), 0.678, 2 }, // 3
	{ Box(0.679756, 0.505654, 0.413894, 0.240368), 0.811, 2 }, // 4
	{ Box(0.687505, 0.504608, 0.401359, 0.246871), 0.612, 2 }, // 5 => replaced by 4
	{ Box(0.890024, 0.491709, 0.370772, 0.226073), 0.784, 2 }, // 6
	{ Box(0.897085, 0.4912  , 0.36957 , 0.215755), 0.601, 2 }  // 7 => replaced by 6
    };

    auto red = Prediction::reduce(predictions, 0.8);

    const std::vector<Prediction> expected_output = {
	{ Box(0.168703, 0.512449, 0.35275 , 0.303689), 0.895, 2 }, // 1
	{ Box(0.679756, 0.505654, 0.413894, 0.240368), 0.811, 2 }, // 4
    };

    EXPECT_EQ(expected_output.size(), red.size());
    EXPECT_EQ(expected_output, red);
}

TEST(Prediction, reduceZeroThreshold) {
    const std::vector<Prediction> predictions = {
	{ Box(0.168703, 0.512449, 0.35275 , 0.303689), 0.895, 2 }, // 1
	{ Box(0.354891, 0.506814, 0.381235, 0.190528), 0.734, 2 }, // 2
	{ Box(0.51416 , 0.50779 , 0.386727, 0.199172), 0.000, 2 }, // 3
	{ Box(0.679756, 0.505654, 0.413894, 0.240368), 0.811, 2 }, // 4
	{ Box(0.687505, 0.504608, 0.401359, 0.246871), 0.612, 2 }, // 5 => replaced by 4
	{ Box(0.890024, 0.491709, 0.370772, 0.226073), 0.784, 2 }, // 6
	{ Box(0.897085, 0.4912  , 0.36957 , 0.215755), 0.601, 2 }  // 7 => replaced by 6
    };

    auto red = Prediction::reduce(predictions, 0.0);

    const std::vector<Prediction> expected_output = {
	{ Box(0.168703, 0.512449, 0.35275 , 0.303689), 0.895, 2 }, // 1
	{ Box(0.354891, 0.506814, 0.381235, 0.190528), 0.734, 2 }, // 2
	{ Box(0.51416 , 0.50779 , 0.386727, 0.199172), 0.000, 2 }, // 3
	{ Box(0.679756, 0.505654, 0.413894, 0.240368), 0.811, 2 }, // 4
	{ Box(0.890024, 0.491709, 0.370772, 0.226073), 0.784, 2 }, // 6
    };

    EXPECT_EQ(expected_output.size(), red.size());
    EXPECT_EQ(expected_output, red);
}

TEST(Prediction, reduceiBoxesNeverMatch) {
    const std::vector<Prediction> predictions = {
	{ Box(0.168703, 0.512449, 0.35275 , 0.303689), 0.895, 2 }, // 1
	{ Box(0.354891, 0.506814, 0.381235, 0.190528), 0.734, 2 }, // 2
	{ Box(0.51416 , 0.50779 , 0.386727, 0.199172), 0.900, 2 }, // 3
	{ Box(0.679756, 0.505654, 0.413894, 0.240368), 0.811, 2 }, // 4
	{ Box(0.687505, 0.504608, 0.401359, 0.246871), 0.612, 2 }, // 5 => replaced by 4
	{ Box(0.890024, 0.491709, 0.370772, 0.226073), 0.784, 2 }, // 6
	{ Box(0.897085, 0.4912  , 0.36957 , 0.215755), 0.601, 2 }  // 7 => replaced by 6
    };

    auto red = Prediction::reduce(predictions, 0.6, [] (const Box &, const Box &) { return false; });

    const std::vector<Prediction> expected_output = {
	{ Box(0.168703, 0.512449, 0.35275 , 0.303689), 0.895, 2 }, // 1
	{ Box(0.354891, 0.506814, 0.381235, 0.190528), 0.734, 2 }, // 2
	{ Box(0.51416 , 0.50779 , 0.386727, 0.199172), 0.900, 2 }, // 3
	{ Box(0.679756, 0.505654, 0.413894, 0.240368), 0.811, 2 }, // 4
	{ Box(0.687505, 0.504608, 0.401359, 0.246871), 0.612, 2 }, // 5 => replaced by 4
	{ Box(0.890024, 0.491709, 0.370772, 0.226073), 0.784, 2 }, // 6
	{ Box(0.897085, 0.4912  , 0.36957 , 0.215755), 0.601, 2 }  // 7 => replaced by 6
    };

    EXPECT_EQ(expected_output.size(), red.size());
    EXPECT_EQ(expected_output, red);
}

TEST(Prediction, reduceiBoxesAlwaysMatch) {
    const std::vector<Prediction> predictions = {
	{ Box(0.168703, 0.512449, 0.35275 , 0.303689), 0.895, 2 }, // 1
	{ Box(0.354891, 0.506814, 0.381235, 0.190528), 0.734, 2 }, // 2
	{ Box(0.51416 , 0.50779 , 0.386727, 0.199172), 0.900, 2 }, // 3
	{ Box(0.679756, 0.505654, 0.413894, 0.240368), 0.811, 2 }, // 4
	{ Box(0.687505, 0.504608, 0.401359, 0.246871), 0.612, 2 }, // 5 => replaced by 4
	{ Box(0.890024, 0.491709, 0.370772, 0.226073), 0.784, 2 }, // 6
	{ Box(0.897085, 0.4912  , 0.36957 , 0.215755), 0.601, 2 }  // 7 => replaced by 6
    };

    auto red = Prediction::reduce(predictions, 0.6, [] (const Box &, const Box &) { return true; });

    const std::vector<Prediction> expected_output = {
	{ Box(0.51416 , 0.50779 , 0.386727, 0.199172), 0.900, 2 }, // 3
    };

    EXPECT_EQ(expected_output.size(), red.size());
    EXPECT_EQ(expected_output, red);
}
