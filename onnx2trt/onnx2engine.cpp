#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"

#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "sampleDevice.h"
#include "sampleEngines.h"
#include "sampleInference.h"
#include "sampleOptions.h"
#include "sampleReporting.h"
#include "onnx2engine.hpp"

#define SMP_RETVAL_IF_FALSE(condition, msg, retval, err)                                                               \
    {                                                                                                                  \
        if ((condition) == false)                                                                                      \
        {                                                                                                              \
            (err) << (msg) << std::endl;                                                                               \
            return retval;                                                                                             \
        }                                                                                                              \
    }

using namespace nvinfer1;
using namespace sample;
using namespace samplesCommon;
namespace
{
	using LibraryPtr = std::unique_ptr<DynamicLibrary>;

#if !TRT_STATIC
#if defined(_WIN32)
	std::string const kNVINFER_PLUGIN_LIBNAME{ "nvinfer_plugin.dll" };
	std::string const kNVINFER_LIBNAME{ "nvinfer.dll" };
	std::string const kNVONNXPARSER_LIBNAME{ "nvonnxparser.dll" };
	std::string const kNVPARSERS_LIBNAME{ "nvparsers.dll" };
	std::string const kNVINFER_LEAN_LIBNAME{ "nvinfer_lean.dll" };
	std::string const kNVINFER_DISPATCH_LIBNAME{ "nvinfer_dispatch.dll" };

	std::string const kMANGLED_UFF_PARSER_CREATE_NAME{ "?createUffParser@nvuffparser@@YAPEAVIUffParser@1@XZ" };
	std::string const kMANGLED_CAFFE_PARSER_CREATE_NAME{ "?createCaffeParser@nvcaffeparser1@@YAPEAVICaffeParser@1@XZ" };
	std::string const kMANGLED_UFF_PARSER_SHUTDOWN_NAME{ "?shutdownProtobufLibrary@nvuffparser@@YAXXZ" };
	std::string const kMANGLED_CAFFE_PARSER_SHUTDOWN_NAME{ "?shutdownProtobufLibrary@nvcaffeparser1@@YAXXZ" };
#else
	std::string const kNVINFER_PLUGIN_LIBNAME = std::string{ "libnvinfer_plugin.so." } + std::to_string(NV_TENSORRT_MAJOR);
	std::string const kNVINFER_LIBNAME = std::string{ "libnvinfer.so." } + std::to_string(NV_TENSORRT_MAJOR);
	std::string const kNVONNXPARSER_LIBNAME = std::string{ "libnvonnxparser.so." } + std::to_string(NV_TENSORRT_MAJOR);
	std::string const kNVPARSERS_LIBNAME = std::string{ "libnvparsers.so." } + std::to_string(NV_TENSORRT_MAJOR);
	std::string const kNVINFER_LEAN_LIBNAME = std::string{ "libnvinfer_lean.so." } + std::to_string(NV_TENSORRT_MAJOR);
	std::string const kNVINFER_DISPATCH_LIBNAME
		= std::string{ "libnvinfer_dispatch.so." } + std::to_string(NV_TENSORRT_MAJOR);

	std::string const kMANGLED_UFF_PARSER_CREATE_NAME{ "_ZN11nvuffparser15createUffParserEv" };
	std::string const kMANGLED_CAFFE_PARSER_CREATE_NAME{ "_ZN14nvcaffeparser117createCaffeParserEv" };
	std::string const kMANGLED_UFF_PARSER_SHUTDOWN_NAME{ "_ZN11nvuffparser23shutdownProtobufLibraryEv" };
	std::string const kMANGLED_CAFFE_PARSER_SHUTDOWN_NAME{ "_ZN14nvcaffeparser123shutdownProtobufLibraryEv" };
#endif
#endif // !TRT_STATIC
	std::function<void* (void*, int32_t)>
		pCreateInferRuntimeInternal{};
	std::function<void* (void*, void*, int32_t)> pCreateInferRefitterInternal{};
	std::function<void* (void*, int32_t)> pCreateInferBuilderInternal{};
	std::function<void* (void*, void*, int)> pCreateNvOnnxParserInternal{};
	std::function<nvuffparser::IUffParser* ()> pCreateUffParser{};
	std::function<nvcaffeparser1::ICaffeParser* ()> pCreateCaffeParser{};
	std::function<void()> pShutdownUffLibrary{};
	std::function<void(void)> pShutdownCaffeLibrary{};

	//! Track runtime used for the execution of trtexec.
	//! Must be tracked as a global variable due to how library init functions APIs are organized.
	RuntimeMode gUseRuntime = RuntimeMode::kFULL;

#if !TRT_STATIC
	inline std::string const& getRuntimeLibraryName(RuntimeMode const mode)
	{
		switch (mode)
		{
		case RuntimeMode::kFULL: return kNVINFER_LIBNAME;
		case RuntimeMode::kDISPATCH: return kNVINFER_DISPATCH_LIBNAME;
		case RuntimeMode::kLEAN: return kNVINFER_LEAN_LIBNAME;
		}
		throw std::runtime_error("Unknown runtime mode");
	}

	template <typename FetchPtrs>
	bool initLibrary(LibraryPtr& libPtr, std::string const& libName, FetchPtrs fetchFunc)
	{
		if (libPtr != nullptr)
		{
			return true;
		}
		try
		{
			libPtr.reset(new DynamicLibrary{ libName });
			fetchFunc(libPtr.get());
		}
		catch (std::exception const& e)
		{
			libPtr.reset();
			sample::gLogError << "Could not load library " << libName << ": " << e.what() << std::endl;
			return false;
		}
		catch (...)
		{
			libPtr.reset();
			sample::gLogError << "Could not load library " << libName << std::endl;
			return false;
		}

		return true;
	}
#endif // !TRT_STATIC

	bool initNvinfer()
	{
#if !TRT_STATIC
		static LibraryPtr libnvinferPtr{};
		auto fetchPtrs = [](DynamicLibrary* l) {
			pCreateInferRuntimeInternal = l->symbolAddress<void* (void*, int32_t)>("createInferRuntime_INTERNAL");

			if (gUseRuntime == RuntimeMode::kFULL)
			{
				pCreateInferRefitterInternal
					= l->symbolAddress<void* (void*, void*, int32_t)>("createInferRefitter_INTERNAL");
				pCreateInferBuilderInternal = l->symbolAddress<void* (void*, int32_t)>("createInferBuilder_INTERNAL");
			}
			};
		return initLibrary(libnvinferPtr, getRuntimeLibraryName(gUseRuntime), fetchPtrs);
#else
		pCreateInferRuntimeInternal = createInferRuntime_INTERNAL;
		pCreateInferRefitterInternal = createInferRefitter_INTERNAL;
		pCreateInferBuilderInternal = createInferBuilder_INTERNAL;
		return true;
#endif // !TRT_STATIC
	}

	bool initNvonnxparser()
	{
#if !TRT_STATIC
		static LibraryPtr libnvonnxparserPtr{};
		auto fetchPtrs = [](DynamicLibrary* l) {
			pCreateNvOnnxParserInternal = l->symbolAddress<void* (void*, void*, int)>("createNvOnnxParser_INTERNAL");
			};
		return initLibrary(libnvonnxparserPtr, kNVONNXPARSER_LIBNAME, fetchPtrs);
#else
		pCreateNvOnnxParserInternal = createNvOnnxParser_INTERNAL;
		return true;
#endif // !TRT_STATIC
	}

	bool initNvparsers()
	{
#if !TRT_STATIC
		static LibraryPtr libnvparsersPtr{};
		auto fetchPtrs = [](DynamicLibrary* l) {
			// TODO: get equivalent Windows symbol names
			pCreateUffParser = l->symbolAddress<nvuffparser::IUffParser * ()>(kMANGLED_UFF_PARSER_CREATE_NAME.c_str());
			pCreateCaffeParser
				= l->symbolAddress<nvcaffeparser1::ICaffeParser * ()>(kMANGLED_CAFFE_PARSER_CREATE_NAME.c_str());
			pShutdownUffLibrary = l->symbolAddress<void()>(kMANGLED_UFF_PARSER_SHUTDOWN_NAME.c_str());
			pShutdownCaffeLibrary = l->symbolAddress<void(void)>(kMANGLED_CAFFE_PARSER_SHUTDOWN_NAME.c_str());
			};
		return initLibrary(libnvparsersPtr, kNVPARSERS_LIBNAME, fetchPtrs);
#else
		pCreateUffParser = nvuffparser::createUffParser;
		pCreateCaffeParser = nvcaffeparser1::createCaffeParser;
		pShutdownUffLibrary = nvuffparser::shutdownProtobufLibrary;
		pShutdownCaffeLibrary = nvcaffeparser1::shutdownProtobufLibrary;
		return true;
#endif // !TRT_STATIC
	}

} // namespace

IRuntime* createRuntime()
{
	if (!initNvinfer())
	{
		return {};
	}
	ASSERT(pCreateInferRuntimeInternal != nullptr);
	return static_cast<IRuntime*>(pCreateInferRuntimeInternal(&gLogger.getTRTLogger(), NV_TENSORRT_VERSION));
}

IBuilder* createBuilder()
{
	if (!initNvinfer())
	{
		return {};
	}
	ASSERT(pCreateInferBuilderInternal != nullptr);
	return static_cast<IBuilder*>(pCreateInferBuilderInternal(&gLogger.getTRTLogger(), NV_TENSORRT_VERSION));
}

IRefitter* createRefitter(ICudaEngine& engine)
{
	if (!initNvinfer())
	{
		return {};
	}
	ASSERT(pCreateInferRefitterInternal != nullptr);
	return static_cast<IRefitter*>(pCreateInferRefitterInternal(&engine, &gLogger.getTRTLogger(), NV_TENSORRT_VERSION));
}

nvonnxparser::IParser* createONNXParser(INetworkDefinition& network)
{
	if (!initNvonnxparser())
	{
		return {};
	}
	ASSERT(pCreateNvOnnxParserInternal != nullptr);
	return static_cast<nvonnxparser::IParser*>(
		pCreateNvOnnxParserInternal(&network, &gLogger.getTRTLogger(), NV_ONNX_PARSER_VERSION));
}

nvcaffeparser1::ICaffeParser* sampleCreateCaffeParser()
{
	if (!initNvparsers())
	{
		return {};
	}
	ASSERT(pCreateCaffeParser != nullptr);
	return pCreateCaffeParser();
}

void shutdownCaffeParser()
{
	if (!initNvparsers())
	{
		return;
	}
	ASSERT(pShutdownCaffeLibrary != nullptr);
	pShutdownCaffeLibrary();
}

nvuffparser::IUffParser* sampleCreateUffParser()
{
	if (!initNvparsers())
	{
		return {};
	}
	ASSERT(pCreateUffParser != nullptr);
	return pCreateUffParser();
}

void shutdownUffParser()
{
	if (!initNvparsers())
	{
		return;
	}
	ASSERT(pShutdownUffLibrary != nullptr);
	pShutdownUffLibrary();
}

using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;
using duration = std::chrono::duration<float>;

Parser modelToNetworkExt(ModelOptions const& model, std::vector<uint8_t> buffer, BuildOptions const& build, nvinfer1::INetworkDefinition& network,
	std::ostream& err, std::vector<std::string>* vcPluginLibrariesUsed)
{
	sample::gLogInfo << "Start parsing network model." << std::endl;
	auto const tBegin = std::chrono::high_resolution_clock::now();

	Parser parser;
	std::string const& modelName = model.baseModel.model;

	using namespace nvonnxparser;
	parser.onnxParser.reset(createONNXParser(network));
	ASSERT(parser.onnxParser != nullptr);
	// For version or hardware compatible engines, we must use TensorRT's native InstanceNorm implementation for
	// compatibility.
	if (build.versionCompatible
		|| (build.hardwareCompatibilityLevel != nvinfer1::HardwareCompatibilityLevel::kNONE))
	{
		auto parserflags = 1U << static_cast<uint32_t>(OnnxParserFlag::kNATIVE_INSTANCENORM);
		parser.onnxParser->setFlags(parserflags);
	}
	//if (!parser.onnxParser->parseFromFile(
	//	model.baseModel.model.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity())))
	if (!parser.onnxParser->parse(buffer.data(), buffer.size()))
	{
		err << "Failed to parse onnx file" << std::endl;
		parser.onnxParser.reset();
	}
	if (vcPluginLibrariesUsed && parser.onnxParser.get())
	{
		int64_t nbPluginLibs;
		char const* const* pluginLibArray = parser.onnxParser->getUsedVCPluginLibraries(nbPluginLibs);
		if (nbPluginLibs >= 0)
		{
			vcPluginLibrariesUsed->reserve(nbPluginLibs);
			for (int64_t i = 0; i < nbPluginLibs; ++i)
			{
				sample::gLogInfo << "Using VC plugin library " << pluginLibArray[i] << std::endl;
				vcPluginLibrariesUsed->emplace_back(std::string{ pluginLibArray[i] });
			}
		}
		else
		{
			sample::gLogWarning << "Failure to query VC plugin libraries required by parsed ONNX network"
				<< std::endl;
		}
	}
	auto const tEnd = std::chrono::high_resolution_clock::now();
	float const parseTime = std::chrono::duration<float>(tEnd - tBegin).count();

	sample::gLogInfo << "Finished parsing network model. Parse time: " << parseTime << std::endl;
	return parser;
}

bool setupNetworkAndConfig(BuildOptions const& build, SystemOptions const& sys, IBuilder& builder,
	INetworkDefinition& network, IBuilderConfig& config, std::unique_ptr<nvinfer1::IInt8Calibrator>& calibrator,
	std::ostream& err, std::vector<std::vector<int8_t>>& sparseWeights)
{
	IOptimizationProfile* profile{ nullptr };
	if (build.maxBatch)
	{
		builder.setMaxBatchSize(build.maxBatch);
	}
	else
	{
		profile = builder.createOptimizationProfile();
	}

	bool hasDynamicShapes{ false };

	bool broadcastInputFormats = broadcastIOFormats(build.inputFormats, network.getNbInputs());

	if (profile)
	{
		// Check if the provided input tensor names match the input tensors of the engine.
		// Throw an error if the provided input tensor names cannot be found because it implies a potential typo.
		for (auto const& shape : build.shapes)
		{
			bool tensorNameFound{ false };
			for (int32_t i = 0; i < network.getNbInputs(); ++i)
			{
				if (network.getInput(i)->getName() == shape.first)
				{
					tensorNameFound = true;
					break;
				}
			}
			if (!tensorNameFound)
			{
				sample::gLogError << "Cannot find input tensor with name \"" << shape.first << "\" in the network "
					<< "inputs! Please make sure the input tensor names are correct." << std::endl;
				return false;
			}
		}
	}

	for (uint32_t i = 0, n = network.getNbInputs(); i < n; i++)
	{
		// Set formats and data types of inputs
		auto* input = network.getInput(i);
		if (!build.inputFormats.empty())
		{
			int inputFormatIndex = broadcastInputFormats ? 0 : i;
			input->setType(build.inputFormats[inputFormatIndex].first);
			input->setAllowedFormats(build.inputFormats[inputFormatIndex].second);
		}
		else
		{
			switch (input->getType())
			{
			case DataType::kINT32:
			case DataType::kBOOL:
			case DataType::kHALF:
			case DataType::kUINT8:
				// Leave these as is.
				break;
			case DataType::kFLOAT:
			case DataType::kINT8:
				// User did not specify a floating-point format.  Default to kFLOAT.
				input->setType(DataType::kFLOAT);
				break;
			case DataType::kFP8: ASSERT(!"FP8 is not supported"); break;
			}
			input->setAllowedFormats(1U << static_cast<int>(TensorFormat::kLINEAR));
		}

		if (profile)
		{
			auto const dims = input->getDimensions();
			auto const isScalar = dims.nbDims == 0;
			auto const isDynamicInput = std::any_of(dims.d, dims.d + dims.nbDims, [](int32_t dim) { return dim == -1; })
				|| input->isShapeTensor();
			if (isDynamicInput)
			{
				hasDynamicShapes = true;
				auto shape = build.shapes.find(input->getName());
				ShapeRange shapes{};

				// If no shape is provided, set dynamic dimensions to 1.
				if (shape == build.shapes.end())
				{
					constexpr int DEFAULT_DIMENSION = 1;
					std::vector<int> staticDims;
					if (input->isShapeTensor())
					{
						if (isScalar)
						{
							staticDims.push_back(1);
						}
						else
						{
							staticDims.resize(dims.d[0]);
							std::fill(staticDims.begin(), staticDims.end(), DEFAULT_DIMENSION);
						}
					}
					else
					{
						staticDims.resize(dims.nbDims);
						std::transform(dims.d, dims.d + dims.nbDims, staticDims.begin(),
							[&](int dimension) { return dimension > 0 ? dimension : DEFAULT_DIMENSION; });
					}
					sample::gLogWarning << "Dynamic dimensions required for input: " << input->getName()
						<< ", but no shapes were provided. Automatically overriding shape to: "
						<< staticDims << std::endl;
					std::fill(shapes.begin(), shapes.end(), staticDims);
				}
				else
				{
					shapes = shape->second;
				}

				std::vector<int> profileDims{};
				if (input->isShapeTensor())
				{
					profileDims = shapes[static_cast<size_t>(OptProfileSelector::kMIN)];
					SMP_RETVAL_IF_FALSE(profile->setShapeValues(input->getName(), OptProfileSelector::kMIN,
						profileDims.data(), static_cast<int>(profileDims.size())),
						"Error in set shape values MIN", false, err);
					profileDims = shapes[static_cast<size_t>(OptProfileSelector::kOPT)];
					SMP_RETVAL_IF_FALSE(profile->setShapeValues(input->getName(), OptProfileSelector::kOPT,
						profileDims.data(), static_cast<int>(profileDims.size())),
						"Error in set shape values OPT", false, err);
					profileDims = shapes[static_cast<size_t>(OptProfileSelector::kMAX)];
					SMP_RETVAL_IF_FALSE(profile->setShapeValues(input->getName(), OptProfileSelector::kMAX,
						profileDims.data(), static_cast<int>(profileDims.size())),
						"Error in set shape values MAX", false, err);
				}
				else
				{
					profileDims = shapes[static_cast<size_t>(OptProfileSelector::kMIN)];
					SMP_RETVAL_IF_FALSE(
						profile->setDimensions(input->getName(), OptProfileSelector::kMIN, toDims(profileDims)),
						"Error in set dimensions to profile MIN", false, err);
					profileDims = shapes[static_cast<size_t>(OptProfileSelector::kOPT)];
					SMP_RETVAL_IF_FALSE(
						profile->setDimensions(input->getName(), OptProfileSelector::kOPT, toDims(profileDims)),
						"Error in set dimensions to profile OPT", false, err);
					profileDims = shapes[static_cast<size_t>(OptProfileSelector::kMAX)];
					SMP_RETVAL_IF_FALSE(
						profile->setDimensions(input->getName(), OptProfileSelector::kMAX, toDims(profileDims)),
						"Error in set dimensions to profile MAX", false, err);
				}
			}
		}
	}

	for (uint32_t i = 0, n = network.getNbOutputs(); i < n; i++)
	{
		auto* output = network.getOutput(i);
		if (profile)
		{
			auto const dims = output->getDimensions();
			// A shape tensor output with known static dimensions may have dynamic shape values inside it.
			auto const isDynamicOutput = std::any_of(dims.d, dims.d + dims.nbDims, [](int32_t dim) { return dim == -1; })
				|| output->isShapeTensor();
			if (isDynamicOutput)
			{
				hasDynamicShapes = true;
			}
		}
	}

	if (!hasDynamicShapes && !build.shapes.empty())
	{
		sample::gLogError << "Static model does not take explicit shapes since the shape of inference tensors will be "
			"determined by the model itself"
			<< std::endl;
		return false;
	}

	if (profile && hasDynamicShapes)
	{
		SMP_RETVAL_IF_FALSE(profile->isValid(), "Required optimization profile is invalid", false, err);
		SMP_RETVAL_IF_FALSE(
			config.addOptimizationProfile(profile) != -1, "Error in add optimization profile", false, err);
	}

	return true;
}

bool networkToSerializedEngine(
	BuildOptions const& build, SystemOptions const& sys, IBuilder& builder, BuildEnvironment& env, std::ostream& err)
{
	std::unique_ptr<IBuilderConfig> config{ builder.createBuilderConfig() };
	std::unique_ptr<nvinfer1::IInt8Calibrator> calibrator;
	std::vector<std::vector<int8_t>> sparseWeights;
	SMP_RETVAL_IF_FALSE(config != nullptr, "Config creation failed", false, err);
	SMP_RETVAL_IF_FALSE(setupNetworkAndConfig(build, sys, builder, *env.network, *config, calibrator, err, sparseWeights), "Network And Config setup failed", false, err);

	std::unique_ptr<ITimingCache> timingCache{ nullptr };
	// Try to load cache from file. Create a fresh cache if the file doesn't exist
	if (build.timingCacheMode == TimingCacheMode::kGLOBAL)
	{
		std::vector<char> loadedCache = samplesCommon::loadTimingCacheFile(build.timingCacheFile);
		timingCache.reset(config->createTimingCache(static_cast<const void*>(loadedCache.data()), loadedCache.size()));
		SMP_RETVAL_IF_FALSE(timingCache != nullptr, "TimingCache creation failed", false, err);
		config->setTimingCache(*timingCache, false);
	}

	// CUDA stream used for profiling by the builder.
	auto profileStream = samplesCommon::makeCudaStream();
	SMP_RETVAL_IF_FALSE(profileStream != nullptr, "Cuda stream creation failed", false, err);
	config->setProfileStream(*profileStream);

	std::unique_ptr<IHostMemory> serializedEngine{ builder.buildSerializedNetwork(*env.network, *config) };
	SMP_RETVAL_IF_FALSE(serializedEngine != nullptr, "Engine could not be created from network", false, err);

	env.engine.setBlob(serializedEngine->data(), serializedEngine->size());

	if (build.safe && build.consistency)
	{
		checkSafeEngine(serializedEngine->data(), serializedEngine->size());
	}

	if (build.timingCacheMode == TimingCacheMode::kGLOBAL)
	{
		auto timingCache = config->getTimingCache();
		samplesCommon::updateTimingCacheFile(build.timingCacheFile, timingCache);
	}

	return true;
}

bool modelToBuildEnvExt(ModelOptions const& model, std::vector<uint8_t> buffer, BuildOptions const& build, SystemOptions& sys, BuildEnvironment& env, std::ostream& err)
{
	env.builder.reset(createBuilder());
	SMP_RETVAL_IF_FALSE(env.builder != nullptr, "Builder creation failed", false, err);
	IErrorRecorder* ptr_gRecorder = (IErrorRecorder*)(&gRecorder);
	env.builder->setErrorRecorder(ptr_gRecorder);
	auto networkFlags
		= (build.maxBatch) ? 0U : 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

	for (auto const& pluginPath : sys.dynamicPlugins)
	{
		env.builder->getPluginRegistry().loadLibrary(pluginPath.c_str());
	}
	env.network.reset(env.builder->createNetworkV2(networkFlags));

	std::vector<std::string> vcPluginLibrariesUsed;
	SMP_RETVAL_IF_FALSE(env.network != nullptr, "Network creation failed", false, err);
	env.parser = modelToNetworkExt(model, buffer, build, *env.network, err, build.versionCompatible ? &vcPluginLibrariesUsed : nullptr);
	SMP_RETVAL_IF_FALSE(env.parser.operator bool(), "Parsing model failed", false, err);

	if (build.versionCompatible && !sys.ignoreParsedPluginLibs && !vcPluginLibrariesUsed.empty())
	{
		sample::gLogInfo << "The following plugin libraries were identified by the parser as required for a "
			"version-compatible engine:"
			<< std::endl;
		for (auto const& lib : vcPluginLibrariesUsed)
		{
			sample::gLogInfo << "    " << lib << std::endl;
		}
		if (!build.excludeLeanRuntime)
		{
			sample::gLogInfo << "These libraries will be added to --setPluginsToSerialize since --excludeLeanRuntime "
				"was not specified."
				<< std::endl;
			std::copy(vcPluginLibrariesUsed.begin(), vcPluginLibrariesUsed.end(),
				std::back_inserter(sys.setPluginsToSerialize));
		}
		sample::gLogInfo << "These libraries will be added to --dynamicPlugins for use at inference time." << std::endl;
		std::copy(vcPluginLibrariesUsed.begin(), vcPluginLibrariesUsed.end(), std::back_inserter(sys.dynamicPlugins));

		// Implicitly-added plugins from ONNX parser should be loaded into plugin registry as well.
		for (auto const& pluginPath : vcPluginLibrariesUsed)
		{
			env.builder->getPluginRegistry().loadLibrary(pluginPath.c_str());
		}

		sample::gLogInfo << "Use --ignoreParsedPluginLibs to disable this behavior." << std::endl;
	}

	SMP_RETVAL_IF_FALSE(networkToSerializedEngine(build, sys, *env.builder, env, err), "Building engine failed", false, err);
	return true;
}

std::vector<uint8_t> buildEngineFromOrtBuffer(std::vector<uint8_t>& ort_buffer, std::unordered_multimap<std::string, std::string>& args) {
	std::vector<uint8_t> trt_buffer;
	AllOptions options;
	try {
		options.parse(args);
	}
	catch (std::invalid_argument const& arg) {
		sample::gLogError << arg.what() << std::endl;
	}

	if (options.reporting.verbose)
	{
		sample::setReportableSeverity(ILogger::Severity::kVERBOSE);
	}

	setCudaDevice(options.system.device, sample::gLogInfo);
	sample::gLogInfo << std::endl;
	sample::gLogInfo << "TensorRT version: " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "."
		<< NV_TENSORRT_PATCH << std::endl;

	// Record specified runtime
	gUseRuntime = options.build.useRuntime;
	try
	{
#if !TRT_STATIC
		LibraryPtr nvinferPluginLib{};
#endif
		std::vector<LibraryPtr> pluginLibs;
		if (gUseRuntime == RuntimeMode::kFULL)
		{
			if (!options.build.versionCompatible)
			{
				sample::gLogInfo << "Loading standard plugins" << std::endl;
#if !TRT_STATIC
				nvinferPluginLib = loadLibrary(kNVINFER_PLUGIN_LIBNAME);
				auto pInitLibNvinferPlugins
					= nvinferPluginLib->symbolAddress<bool(void*, char const*)>("initLibNvInferPlugins");
#else
				auto pInitLibNvinferPlugins = initLibNvInferPlugins;
#endif
				ASSERT(pInitLibNvinferPlugins != nullptr);
				pInitLibNvinferPlugins(&sample::gLogger.getTRTLogger(), "");
			}
			else
			{
				sample::gLogInfo << "Not loading standard plugins since --versionCompatible is specified." << std::endl;
			}
			for (auto const& pluginPath : options.system.plugins)
			{
				sample::gLogInfo << "Loading supplied plugin library: " << pluginPath << std::endl;
				pluginLibs.emplace_back(loadLibrary(pluginPath));
			}
		}
		else if (!options.system.plugins.empty())
		{
			throw std::runtime_error("TRT-18412: Plugins require --useRuntime=full.");
		}

		if (options.build.safe && !sample::hasSafeRuntime())
		{
			sample::gLogError << "Safety is not supported because safety runtime library is unavailable." << std::endl;
			throw std::runtime_error("Safety is not supported because safety runtime library is unavailable.");
		}

		if (!options.build.safe && options.build.consistency)
		{
			sample::gLogInfo << "Skipping consistency checker on non-safety mode." << std::endl;
			options.build.consistency = false;
		}

		// Start engine building phase.
		std::unique_ptr<BuildEnvironment> bEnv(new BuildEnvironment(options.build.safe, options.build.versionCompatible,
			options.system.DLACore, options.build.tempdir, options.build.tempfileControls, options.build.leanDLLPath));

		time_point const buildStartTime{ std::chrono::high_resolution_clock::now() };

		//std::vector<uint8_t> buffer = loadFile(options.model.baseModel.model);
		bool buildPass = modelToBuildEnvExt(options.model, ort_buffer, options.build, options.system, *bEnv, sample::gLogError);

		//bool buildPass = getEngineBuildEnv(options.model, options.build, options.system, *bEnv, sample::gLogError);
		time_point const buildEndTime{ std::chrono::high_resolution_clock::now() };
		trt_buffer = bEnv->engine.getBlob();

		if (!buildPass)
		{
			sample::gLogError << "Engine set up failed" << std::endl;
			throw std::runtime_error("Engine set up failed.");
		}


		// dynamicPlugins may have been updated by getEngineBuildEnv above
		bEnv->engine.setDynamicPlugins(options.system.dynamicPlugins);

		sample::gLogInfo << "Engine " << (options.build.load ? "loaded" : "built") << " in "
			<< duration(buildEndTime - buildStartTime).count() << " sec." << std::endl;

		if (!options.build.safe && options.build.refittable)
		{
			auto* engine = bEnv->engine.get();
			if (options.reporting.refit)
			{
				dumpRefittable(*engine);
			}
			if (options.inference.timeRefit)
			{
				if (bEnv->network.operator bool())
				{
					bool const success = timeRefit(*bEnv->network, *engine, options.inference.threads);
					if (!success)
					{
						sample::gLogError << "Engine refit failed." << std::endl;
						throw std::runtime_error("Engine refit failed.");
					}
				}
				else
				{
					sample::gLogWarning << "Network not available, skipped timing refit." << std::endl;
				}
			}
		}

		if (options.build.skipInference)
		{
			if (!options.build.safe)
			{
				printLayerInfo(options.reporting, bEnv->engine.get(), nullptr);
			}
			sample::gLogInfo << "Skipped inference phase since --skipInference is added." << std::endl;
			throw std::runtime_error("Skipped inference phase since --skipInference is added.");
		}

		// Start inference phase.
		std::unique_ptr<InferenceEnvironment> iEnv(new InferenceEnvironment(*bEnv));

		// Delete build environment.
		bEnv.reset();

		if (options.inference.timeDeserialize)
		{
			if (timeDeserialize(*iEnv, options.system))
			{
				throw std::runtime_error("Deserialize failed.");
			}
			throw std::runtime_error("Deserialize failed.");
		}

		if (options.build.safe && options.system.DLACore >= 0)
		{
			sample::gLogInfo << "Safe DLA capability is detected. Please save DLA loadable with --saveEngine option, "
				"then use dla_safety_runtime to run inference with saved DLA loadable, "
				"or alternatively run with your own application"
				<< std::endl;
			throw std::runtime_error("Safe DLA capability is detected. Please save DLA loadable with --saveEngine option.");
		}

		bool const profilerEnabled = options.reporting.profile || !options.reporting.exportProfile.empty();

		if (iEnv->safe && profilerEnabled)
		{
			sample::gLogError << "Safe runtime does not support --dumpProfile or --exportProfile=<file>, please use "
				"--verbose to print profiling info."
				<< std::endl;
			throw std::runtime_error("Safe runtime does not support.");
		}

		if (profilerEnabled && !options.inference.rerun)
		{
			iEnv->profiler.reset(new Profiler);
			if (options.inference.graph && (getCudaDriverVersion() < 11010 || getCudaRuntimeVersion() < 11000))
			{
				options.inference.graph = false;
				sample::gLogWarning
					<< "Graph profiling only works with CUDA 11.1 and beyond. Ignored --useCudaGraph flag "
					"and disabled CUDA graph."
					<< std::endl;
			}
		}

		if (!setUpInference(*iEnv, options.inference, options.system))
		{
			sample::gLogError << "Inference set up failed" << std::endl;
			throw std::runtime_error("Inference set up failed.");
		}

		if (!options.build.safe)
		{
			printLayerInfo(options.reporting, iEnv->engine.get(), iEnv->contexts.front().get());
		}

		std::vector<InferenceTrace> trace;
		sample::gLogInfo << "Starting inference" << std::endl;

		if (!runInference(options.inference, *iEnv, options.system.device, trace))
		{
			sample::gLogError << "Error occurred during inference" << std::endl;
			throw std::runtime_error("Error occurred during inference.");
		}

		if (profilerEnabled && !options.inference.rerun)
		{
			sample::gLogInfo << "The e2e network timing is not reported since it is inaccurate due to the extra "
				<< "synchronizations when the profiler is enabled." << std::endl;
			sample::gLogInfo
				<< "To show e2e network timing report, add --separateProfileRun to profile layer timing in a "
				<< "separate run or remove --dumpProfile to disable the profiler." << std::endl;
		}
		else
		{
			printPerformanceReport(trace, options.reporting, options.inference, sample::gLogInfo, sample::gLogWarning,
				sample::gLogVerbose);
		}

		printOutput(options.reporting, *iEnv, options.inference.batch);

		if (profilerEnabled && options.inference.rerun)
		{
			auto* profiler = new Profiler;
			iEnv->profiler.reset(profiler);
			iEnv->contexts.front()->setProfiler(profiler);
			iEnv->contexts.front()->setEnqueueEmitsProfile(false);
			if (options.inference.graph && (getCudaDriverVersion() < 11010 || getCudaRuntimeVersion() < 11000))
			{
				options.inference.graph = false;
				sample::gLogWarning
					<< "Graph profiling only works with CUDA 11.1 and beyond. Ignored --useCudaGraph flag "
					"and disabled CUDA graph."
					<< std::endl;
			}
			if (!runInference(options.inference, *iEnv, options.system.device, trace))
			{
				sample::gLogError << "Error occurred during inference" << std::endl;
				throw std::runtime_error("Error occurred during inference.");
			}
		}
		printPerformanceProfile(options.reporting, *iEnv);

		//return -1;
	}
	catch (std::exception const& e)
	{
		sample::gLogError << "Uncaught exception detected: " << e.what() << std::endl;
	}

	return trt_buffer;
}

