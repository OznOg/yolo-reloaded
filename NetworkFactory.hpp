#include <Network.hpp>

namespace yolo {

class NetworkFactory {
public:
    Network createFromFile(const std::string &fileName);
    Network createFromString(const std::string &config);
};

} // namespace yolo
