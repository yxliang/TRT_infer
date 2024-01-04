#include "onnx2engine.hpp"
#include <fstream>

std::vector<uint8_t> loadFile(std::string const file)
{
    std::ifstream ifile(file, std::ios::binary);
    ifile.seekg(0, std::ifstream::end);
    int64_t fsize = ifile.tellg();
    ifile.seekg(0, std::ifstream::beg);

    std::vector<uint8_t> buffer(fsize);
    ifile.read(reinterpret_cast<char*>(buffer.data()), fsize);
    return buffer;
}

std::unordered_multimap<std::string, std::string> argsToArgumentsMap(int32_t argc, char* argv[])
{
    std::unordered_multimap<std::string, std::string> arguments;
    for (int32_t i = 1; i < argc; ++i)
    {
        auto valuePtr = strchr(argv[i], '=');
        if (valuePtr)
        {
            std::string value{ valuePtr + 1 };
            arguments.emplace(std::string(argv[i], valuePtr - argv[i]), value);
        }
        else
        {
            arguments.emplace(argv[i], "");
        }
    }
    return arguments;
}

int main(int argc, char** argv) {
	std::unordered_multimap<std::string, std::string> args;
    //args.emplace()
	args = argsToArgumentsMap(argc, argv);
	std::vector<uint8_t>& ort_buffer = loadFile("D:/Pinxin/DeepBCM/build/DeepBCM/bcmlow.onnx");
	std::vector<uint8_t> trt_buffer = buildEngineFromOrtBuffer(ort_buffer, args);
	return 0;
}