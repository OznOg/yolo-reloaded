#include <NetworkFactory.hpp>
#include <iostream>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <Size.hpp>

namespace yolo {

static std::string getFileContent(const std::string &fileName) {
    std::ifstream file(fileName);

    std::string file_content;

    while (file) {
	std::string line;

	std::getline(file, line);

	if (line[0] == '#') // comment line
	    continue;

	if (line.size() == 0) // blank line
	    continue;

	 file_content += line + "\n";
    }

    return file_content;
}

class ConfigHunk {
    struct Entry {
	Entry(const std::string &optionLine) {
	    auto equalPos = optionLine.find('=');

	    if (equalPos == std::string::npos)
		throw std::invalid_argument("Entry parsing error.");

	    key = optionLine.substr(0, equalPos);
	    key.erase(key.find_last_not_of(" ") + 1); //trim

	    std::string value;
	    std::stringstream valueStream(optionLine.substr(equalPos + 1, std::string::npos));
	    while (std::getline(valueStream, value, ',')) {
		values.push_back(value);
	    }
	    if (values.size() == 0)
		throw std::invalid_argument("missing value for entry: '" + key + "'");
	}

	std::string key;
	std::vector<std::string> values;
    };
public:
    ConfigHunk(const std::string &configHunkStr) {
	std::istringstream configStream(configHunkStr);

	std::getline(configStream, _name);

	for (std::string optionLine; std::getline(configStream, optionLine); ) {
	    _entries.push_back(Entry(optionLine));
	}
    }

    template <class T>
    std::optional<T> getScalar(const std::string &key) const {
	static_assert(std::is_scalar<T>::value || std::is_same<std::string, T>::value,
                      "Trying to pick a non scalar value; for non scalar, use vector specialization.");
	for (const auto &e : _entries)
	    if (e.key == key) {
		if (e.values.size() != 1) 
		    throw std::invalid_argument("Trying to pick a vector as scalar.");
		
		T numeric_val;
		std::istringstream ss(e.values[0]);
		ss >> numeric_val;
		return numeric_val;
	    }
        return {}; // not found
    };

    template <class T>
    std::optional<std::vector<T>> getVector(const std::string &key) const {
	for (const auto &e : _entries)
	    if (e.key == key) {
		std::vector<T> out;
		for (const auto &v : e.values) {
		    std::istringstream ss(v);
		    T numeric_val;
		    ss >> numeric_val;
		    out.push_back(numeric_val);
		}
		return out;
	    }
        return {}; // not found
    };

    const std::string &getName() const {
	return _name;
    }
private:
    std::string _name;
    std::vector<Entry> _entries;
};

static ConfigHunk popConfigHunk(std::string &remaining_content) {
    // A hunk MUST begin with '['
    if (remaining_content[0] != '[')
	throw std::invalid_argument("Invalid config content.");

    auto next_hunk_begin = remaining_content.find('[', 1);
    if (next_hunk_begin != std::string::npos) {
	auto hunk = remaining_content.substr(0, next_hunk_begin - 1);
	remaining_content = remaining_content.substr(next_hunk_begin, std::string::npos);
	return ConfigHunk(hunk);
    } else { // last hunk
        auto hunk = remaining_content;
	remaining_content = "";
	return ConfigHunk(hunk);
    }
}


static std::unique_ptr<Policy> makePolicy(const ConfigHunk &config) {
    if (config.getName() != "[net]" && config.getName() != "[network]")
	throw std::invalid_argument("Need a network configuration for creating policy.");
    
    std::unique_ptr<Policy> policy;

    const auto policy_name = config.getScalar<std::string>("policy");

    if (policy_name == "steps") {
	const auto ranks = config.getVector<int>("steps");
	const auto scales = config.getVector<float>("steps");
	if (ranks->size() != scales->size())
	    throw std::invalid_argument("Inconsistant policy configuration, scales count do not match steps count");

	std::vector<StepsPolicy::Step> steps;
	for (size_t i = 0; i < ranks->size(); i++) {
	    steps.push_back(StepsPolicy::Step(ranks.value()[i], scales.value()[i]));
	}
	policy.reset(new StepsPolicy(steps));


#if 0
    // FIXME  well I don't really know how to test thos policies as I have no config file using them...
    } else if (policy_name == "poly") {
    } else if (policy_name == "constant") {
    } else if (policy_name == "sigmoid") {
    } else if (policy_name == "step") {
    } else if (policy_name == "exp") {
    } else if (policy_name == "random") {
#endif
    } else {
	throw std::invalid_argument("Unhandled policy type '" + policy_name.value() + "'");
    }

    return policy;
}

static std::unique_ptr<Network> makeNetwork(const ConfigHunk &config) {
    if (config.getName() != "[net]" && config.getName() != "[network]")
	throw std::invalid_argument("Inconsistant configuration (network not found at top level)");


    std::unique_ptr<Network> net(new Network()); 
    net->_batch             = config.getScalar<int>("batch").value();
    net->_subdivisions      = config.getScalar<size_t>("subdivisions").value();
    net->_input_size.width  = config.getScalar<size_t>("width").value();
    net->_input_size.height = config.getScalar<size_t>("height").value();
    net->_channels          = config.getScalar<size_t>("channels").value();
    net->_momentum          = config.getScalar<float>("momentum").value();
    net->_decay             = config.getScalar<float>("decay").value();
    net->_angle             = config.getScalar<float>("angle").value();
    net->_saturation        = config.getScalar<float>("saturation").value();
    net->_exposure          = config.getScalar<float>("exposure").value();
    net->_hue               = config.getScalar<float>("hue").value();
    net->_learning_rate     = config.getScalar<float>("learning_rate").value();
    net->_burn_in           = config.getScalar<size_t>("burn_in").value();
    net->_max_batches       = config.getScalar<size_t>("max_batches").value();

    net->setPolicy(makePolicy(config));
    return net;
}

static std::unique_ptr<Layer> makeLayer(const ConfigHunk &config) {
    std::string layer_name = config.getName().substr(1, config.getName().size() - 2); // remove [] around name

    if (layer_name == "convolutional") {
	bool batch_normalize   = config.getScalar<bool>("batch_normalize").value_or(false);
	size_t filters         = config.getScalar<size_t>("filters").value();
	size_t size            = config.getScalar<size_t>("size").value();
	size_t stride          = config.getScalar<size_t>("stride").value();
	bool pad               = config.getScalar<bool>("pad").value();
	size_t padding         = pad ? size / 2 : 0;
	Activation activation  = activationFromString(config.getScalar<std::string>("activation").value());

	return std::unique_ptr<Layer>(new ConvolutionalLayer(batch_normalize, filters, size, stride, padding, activation));
    } else if (layer_name == "maxpool") {
	size_t stride          = config.getScalar<size_t>("stride").value();
	size_t size            = config.getScalar<size_t>("size").value();
	size_t padding = config.getScalar<size_t>("padding").value_or((size - 1) / 2);

	return std::unique_ptr<Layer>(new MaxpoolLayer(size, stride, padding));
    } else if (layer_name == "region") {
        int coords = config.getScalar<int>("coords").value_or(4);
        int classes = config.getScalar<int>("classes").value_or(20);
        int num = config.getScalar<int>("num").value_or(1);
        auto biases = config.getVector<float>("anchors").value_or(std::vector<float>());

        auto layer = new RegionLayer(num, classes, coords, biases);

#if 0
        //FIXME not used yet...
        auto log = config.getScalar<int>("log").value_or(0);
        auto sqrt = config.getScalar<int>("sqrt").value_or(0);

        auto softmax = config.getScalar<int>("softmax").value_or(0);
        auto background = config.getScalar<int>("background").value_or(0);
        auto max_boxes = config.getScalar<int>("max").value_or(30);
        auto jitter = config.getScalar<float>("jitter").value_or(.2);
        auto rescore = config.getScalar<int>("rescore").value_or(0);

        auto thresh = config.getScalar<float>("thresh").value_or(.5);
        auto classfix = config.getScalar<int>("classfix").value_or(0);
        auto absolute = config.getScalar<int>("absolute").value_or(0);
        auto random = config.getScalar<int>("random").value_or(0);

        auto coord_scale = config.getScalar<float>("coord_scale").value_or(1);
        auto object_scale = config.getScalar<float>("object_scale").value_or(1);
        auto noobject_scale = config.getScalar<float>("noobject_scale").value_or(1);
        auto mask_scale = config.getScalar<float>("mask_scale").value_or(1);
        auto class_scale = config.getScalar<float>("class_scale").value_or(1);
        auto bias_match = config.getScalar<int>("bias_match").value_or(0);

        auto tree_file = config.getScalar<std::string>("tree").value_or(nullptr);
#endif

        // FIXME tree parsing not handled (yet)
        return std::unique_ptr<Layer>(layer);

    } else if (layer_name == "reorg") {
	size_t stride = config.getScalar<size_t>("stride").value_or(1);
        bool reverse  = config.getScalar<bool>("reverse").value_or(false);
        bool flatten  = config.getScalar<bool>("flatten").value_or(false);
        bool extra    = config.getScalar<bool>("extra").value_or(false);
        return std::unique_ptr<Layer>(new ReorgLayer(stride, reverse, flatten, extra));

    } else if (layer_name == "route") {
        throw std::invalid_argument("Route are not standard layers, cannot be handled here.");
    } else {
	throw std::invalid_argument("Unhandled layer type '" + layer_name + "'");
    }
    return {};
}

static inline bool isRouteConfig(const ConfigHunk &config) {
    return config.getName() == "[route]";
}

std::unique_ptr<Network> NetworkFactory::createFromString(const std::string &content) {
    auto local_copy = content;

    //file Content MUST begin with network stuff
    const auto &networkConfig = popConfigHunk(local_copy);
    auto net = makeNetwork(networkConfig);

    while (!local_copy.empty()) {
	const auto &config = popConfigHunk(local_copy);
        if (isRouteConfig(config)) {
            const auto layers_idx = config.getVector<int>("layers").value();
            net->addRoute(layers_idx);
        } else
            net->addLayer(makeLayer(config));
    }
 
    return net;
}

std::unique_ptr<Network> NetworkFactory::createFromFile(const std::string &fileName) {
    std::string file_content = getFileContent(fileName);

    return createFromString(file_content);
}

} // namespace yolo

