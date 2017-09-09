#include <NetworkFactory.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <type_traits>

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
    T getScalar(const std::string &key) const {
	static_assert(std::is_scalar<T>::value, "Trying to pick a non scalar value; for non scalar, use vector specialization.");
	for (const auto &e : _entries)
	    if (e.key == key) {
		if (e.values.size() != 1) 
		    throw std::invalid_argument("Trying to pick a vector as scalar.");
		
		return e.values[0];
	    }
	throw std::invalid_argument("No matching key '" + key + "' found.");
    };

    template <class T>
    std::vector<T> getVector(const std::string &key) const {
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
	throw std::invalid_argument("No matching key '" + key + "' found.");
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


static Network makeNetwork(const ConfigHunk &config) {
    if (config.getName() != "[net]" && config.getName() != "[network]")
	throw std::invalid_argument("Inconsistant configuration (network not found at top level)");

    const auto &steps = config.getVector<int>("steps");
    for (const auto &s : steps)
	std::cout << "step ====> " << s << "<===" << std::endl;

    return {};
}

Network NetworkFactory::createFromString(const std::string &content) {
    auto local_copy = content;

    //file Content MUST begin with network stuff
    const auto &networkConfig = popConfigHunk(local_copy);
    Network net = makeNetwork(networkConfig);

    while (!local_copy.empty()) {
	const auto &head = popConfigHunk(local_copy);
	std::cout << "====> " << head.getName() << "<===" << std::endl;
    }
 
    return {};
}

Network NetworkFactory::createFromFile(const std::string &fileName) {
    std::string file_content = getFileContent(fileName);

    return createFromString(file_content);
}

} // namespace yolo

