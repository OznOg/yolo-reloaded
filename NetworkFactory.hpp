#pragma once

#include <Network.hpp>
#include <memory>

namespace yolo {

class NetworkFactory {
public:
    std::unique_ptr<Network> createFromFile(const std::string &fileName, bool training);
    std::unique_ptr<Network> createFromString(const std::string &config, bool training);
};

} // namespace yolo
