#ifndef __ONNX2ENGINE_HPP__
#define __ONNX2ENGINE_HPP__

#include <unordered_map>
#include <vector>
#include <string>
std::vector<uint8_t> buildEngineFromOrtBuffer(std::vector<uint8_t>& ort_buffer, std::unordered_multimap<std::string, std::string>& args);

#endif //__ONNX2ENGINE_HPP__