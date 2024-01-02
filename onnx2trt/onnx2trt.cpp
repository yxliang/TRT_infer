#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <NvOnnxParser.h>

class SimpleLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        switch (severity) {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "[INTERNAL_ERROR] " << msg << std::endl;
            break;
        case Severity::kERROR:
            std::cerr << "[ERROR] " << msg << std::endl;
            break;
        case Severity::kWARNING:
            std::cerr << "[WARNING] " << msg << std::endl;
            break;
        case Severity::kINFO:
            std::cout << "[INFO] " << msg << std::endl;
            break;
        default:
            break;
        }
    }
};

template<typename _T>
static void destroy_nvidia_pointer(_T* ptr) {
    if (ptr) ptr->destroy();
}

class InputDims {
public:
    InputDims() = default;

    // ��Ϊ-1ʱ����������ʱ������ṹ�ߴ�
    InputDims(const std::initializer_list<int>& dims);
    InputDims(const std::vector<int>& dims);

    const std::vector<int>& dims() const;

private:
    std::vector<int> dims_;
};

int main() {
    // ���� TensorRT �� Builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(SimpleLogger{});

    // ���� TensorRT ������
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

    // ���� TensorRT ��ONNX������
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, *builder->getLogger());

    // ��ȡONNXģ���ļ�
    const std::string onnxModelPath = "../workspace/yolov7.onnx";
    std::ifstream onnxFile(onnxModelPath, std::ios::binary);
    if (!onnxFile) {
        std::cerr << "Error opening ONNX file" << std::endl;
        return -1;
    }
    onnxFile.seekg(0, onnxFile.end);
    size_t size = onnxFile.tellg();
    onnxFile.seekg(0, onnxFile.beg);
    std::vector<char> onnxModelData(size);
    onnxFile.read(onnxModelData.data(), size);

    // ����ONNXģ��
    if (!parser->parse(onnxModelData.data(), size)) {
        std::cerr << "Error parsing ONNX model" << std::endl;
        return -1;
    }

    // ���� TensorRT ������
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

    builder->setMaxBatchSize(32); 
    config->setMaxWorkspaceSize(1 << 30);  // 1 GB

    // ʹ���µĹ�������ķ�ʽ
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    // ����TensorRT���浽�ļ�
    const std::string trtEnginePath = "../workspace/yolov7.trt";
    nvinfer1::IHostMemory* serializedEngine = engine->serialize();
    std::ofstream trtEngineFile(trtEnginePath, std::ios::binary);
    trtEngineFile.write(static_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    trtEngineFile.close();

    // �ͷ���Դ
    serializedEngine->destroy();
    engine->destroy();
    parser->destroy();
    network->destroy();
    builder->destroy();

    return 0;
}
