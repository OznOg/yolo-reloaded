#include "Layer.hpp"
#include <fstream>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"

std::vector<float> input = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,

                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

TEST(Layers, Convolutional) {
    yolo::ConvolutionalLayer cl(true, 2, 3, 1, 1, yolo::Activation::Leaky);

    cl.setInputFormat(yolo::Format(10, 10, 2, 1));

    const float values[] { 0.123, 0.163, -0.56, 0.022, .002, 0.33, 0.40, 0.7, 0.8, 0.9,
                           0.123, 0.163, -0.56, .13, .17, .15, .14, .17, .18, .19,
                           0.123, 0.163, 0.55, .33, .7, .115, .14, .1117, .1115, .12223,
                           0.123, 0.163, 0.56, .32, .7, .35, .24, .7, .11, .9,
                           0.155, 0.163, 0.56, .31
                          };

    std::stringstream ss((char *)values);
                        //  "I should definitly rework that bad API taking steam as"
                        //  " input... As a matter of fact this makes testing quite"
                        //  " hard as I don't know how to make a suitable stream of"
                        //  " serialized floats in here...");

    cl.loadWeights(ss);

    const std::vector<float> &output = cl.forward(input);

    std::vector<float> expected_output = {
            -0.0703112036,   -0.193209752,    -0.385703623 ,   -0.578197539,
            -0.770691335 ,   -0.963185191,    -1.15567911  ,   -1.34817302,
            -1.5406667   ,   -0.90527761 ,    -0.0980253667,   -0.344265193,
            -0.745987773 ,   -1.14771008 ,    -1.54943252  ,   -1.95115507,
            -2.35287762  ,   -2.75460005 ,    -3.15632296  ,   -2.4530201,
            -0.0980253667,   -0.344265193,    -0.745987773 ,   -1.14771008,
            -1.54943252  ,   -1.95115507 ,    -2.35287762  ,   -2.75460005,
            -3.15632296  ,   -2.4530201  ,    -0.0980253667,   -0.344265193,
            -0.745987773 ,   -1.14771008 ,    -1.54943252  ,   -1.95115507,
            -2.35287762  ,   -2.75460005 ,    -3.15632296  ,   -2.4530201,
            -0.0980253667,   -0.344265193,    -0.745987773 ,   -1.14771008,
            -1.54943252  ,   -1.95115507 ,    -2.35287762  ,   -2.75460005,
            -3.15632296  ,   -2.4530201  ,    -0.0980253667,   -0.344265193,
            -0.745987773 ,   -1.14771008 ,    -1.54943252  ,   -1.95115507,
            -2.35287762  ,   -2.75460005 ,    -3.15632296  ,   -2.4530201,
            -0.0980253667,   -0.344265193,    -0.745987773 ,   -1.14771008,
            -1.54943252  ,   -1.95115507 ,    -2.35287762  ,   -2.75460005,
            -3.15632296  ,   -2.4530201  ,    -0.0980253667,   -0.344265193,
            -0.745987773 ,   -1.14771008 ,    -1.54943252  ,   -1.95115507,
            -2.35287762  ,   -2.75460005 ,    -3.15632296  ,   -2.4530201,
            -0.0980253667,   -0.344265193,    -0.745987773 ,   -1.14771008,
            -1.54943252  ,   -1.95115507 ,    -2.35287762  ,   -2.75460005,
            -3.15632296  ,   -2.4530201  ,    -0.0754467472,   -0.223845884,
            -0.483455807 ,   -0.743065655,    -1.00267565  ,   -1.26228547,
            -1.52189529  ,   -1.78150535 ,    -2.04111552  ,   -1.42148697,
            0.189242408  ,   0.274201542 ,    0.384278476  ,   0.494355381,
            0.604432225  ,   0.71450913  ,    0.824586093  ,   0.934662938,
            1.04473984   ,   0.80561924  ,    0.210580796  ,   0.326126277,
            0.479674041  ,   0.633221865 ,    0.786769629  ,   0.940317452,
            1.09386516   ,   1.24741316  ,    1.4009608    ,   0.991927147,
            0.210580796  ,   0.326126277 ,    0.479674041  ,   0.633221865,
            0.786769629  ,   0.940317452 ,    1.09386516   ,   1.24741316,
            1.4009608    ,   0.991927147 ,    0.210580796  ,   0.326126277,
            0.479674041  ,   0.633221865 ,    0.786769629  ,   0.940317452,
            1.09386516   ,   1.24741316  ,    1.4009608    ,   0.991927147,
            0.210580796  ,   0.326126277 ,    0.479674041  ,   0.633221865,
            0.786769629  ,   0.940317452 ,    1.09386516   ,   1.24741316,
            1.4009608    ,   0.991927147 ,    0.210580796  ,   0.326126277,
            0.479674041  ,   0.633221865 ,    0.786769629  ,   0.940317452,
            1.09386516   ,   1.24741316  ,    1.4009608    ,   0.991927147,
            0.210580796  ,   0.326126277 ,    0.479674041  ,   0.633221865,
            0.786769629  ,   0.940317452 ,    1.09386516   ,   1.24741316,
            1.4009608    ,   0.991927147 ,    0.210580796  ,   0.326126277,
            0.479674041  ,   0.633221865 ,    0.786769629  ,   0.940317452,
            1.09386516   ,   1.24741316  ,    1.4009608    ,   0.991927147,
            0.210580796  ,   0.326126277 ,    0.479674041  ,   0.633221865,
            0.786769629  ,   0.940317452 ,    1.09386516   ,   1.24741316,
            1.4009608    ,   0.991927147 ,    0.184022859  ,   0.249870777,
            0.334709764  ,   0.41954875  ,    0.504387677  ,   0.589226604,
            0.67406559   ,   0.758904517 ,    0.843743622  ,   0.631580532 };

    EXPECT_EQ(expected_output.size(), output.size());

    size_t idx = 0;
    for (auto &f : expected_output)
        EXPECT_NEAR(f, output[idx++], 0.000001);
}

TEST(Layers, Maxpool) {
    yolo::MaxpoolLayer ml(2, 2, 0);

    ml.setInputFormat(yolo::Format(10, 10, 2, 1));


    std::vector<float> expected_output = { 1, 3, 5, 7, 9, 1, 3, 5, 7, 9,
                                           1, 3, 5, 7, 9, 1, 3, 5, 7, 9,
                                           1, 3, 5, 7, 9, 1, 3, 5, 7, 9,
                                           1, 3, 5, 7, 9, 1, 3, 5, 7, 9,
                                           1, 3, 5, 7, 9, 1, 3, 5, 7, 9 };

    const std::vector<float> &output = ml.forward(input);

    EXPECT_EQ(expected_output.size(), output.size());

    EXPECT_EQ(expected_output, output);
}

