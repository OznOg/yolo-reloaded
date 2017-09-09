#pragma once

#include <Network.hpp>
#include <memory>

namespace yolo {

class NetworkFactory {
public:
    std::unique_ptr<Network> createFromFile(const std::string &fileName);
    std::unique_ptr<Network> createFromString(const std::string &config);
};

} // namespace yolo
