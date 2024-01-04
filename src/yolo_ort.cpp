#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "yolo_ort.hpp"
#include <fstream>

namespace yolo_ort {
	using namespace std;
	using namespace cv;

	typedef struct MY_Buffer {
		const void* data;
		size_t length;
		void (*data_deallocator)(void* data);
	} MY_Buffer;

	void DeallocateBuffer(void* data) {
		std::free(data);
	}

	MY_Buffer* ReadBufferFromFile(const char* file) {
		std::ifstream f(file, std::ios::binary);
		if (f.fail() || !f.is_open()) {
			return nullptr;
		}

		if (f.seekg(0, std::ios::end).fail()) {
			return nullptr;
		}
		auto fsize = f.tellg();
		if (f.seekg(0, std::ios::beg).fail()) {
			return nullptr;
		}

		if (fsize <= 0) {
			return nullptr;
		}

		auto data = static_cast<char*>(std::malloc(fsize));
		if (f.read(data, fsize).fail()) {
			return nullptr;
		}

		MY_Buffer* buf = new MY_Buffer;
		buf->data = data;
		buf->length = fsize;
		buf->data_deallocator = DeallocateBuffer;

		return buf;
	}

	class ORTWrapper {
	public:
		ORTWrapper(const char* model_file, int device_id, int n_thread);
		ORTWrapper(const void* model_data, size_t model_data_length, int device_id, int n_thread);
		std::vector<Ort::Value> forward(std::vector<float>& input, std::vector<int64_t>& input_dims);

		virtual bool has_dynamic_dim() {
			// check if any input have dynamic shapes
			for (size_t i = 0; i < input_dims.size(); ++i) {
				for (size_t j = 0; j < input_dims[i].size(); ++j) {
					if (input_dims[i][j] == -1)
						return true;
				}
			}
			return false;
		}
		virtual void print();

		virtual std::vector<int64_t> static_dims(int idx) {
			return input_dims[idx];
		}

	protected:
		Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "Default" };
		Ort::Session session{ nullptr };
		Ort::MemoryInfo memory_info{ nullptr };
		std::vector<const char*> input_names, output_names;
		std::vector< std::vector<int64_t>> input_dims, output_dims;
		bool isDynamicInputShape{ false };
	};

	ORTWrapper::ORTWrapper(const char* model_file, int device_id, int n_thread) {
		MY_Buffer* buffer = ReadBufferFromFile(model_file);
		new(this)ORTWrapper(buffer->data, buffer->length, device_id, n_thread);
		buffer->data_deallocator((void*)buffer->data);
	}

	ORTWrapper::ORTWrapper(const void* model_data, size_t model_data_length, int device_id, int n_thread) {

		Ort::SessionOptions session_options;
		this->isDynamicInputShape = false;
		session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_ERROR);

		std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
#if defined(USE_CUDA)
		std::cout << "first try cuda" << std::endl;
		//auto cuda_available = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
		//if (cuda_available != availableProviders.end() && device_id >= 0) {
		bool cuda_available = (std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider")) != availableProviders.end();
		if (cuda_available && device_id >= 0) {
			std::cout << "Try GPU" << std::endl;

			OrtCUDAProviderOptions cuda_options;
			cuda_options.device_id = device_id;
			cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
			cuda_options.gpu_mem_limit = SIZE_MAX;
			try {
				session_options.AppendExecutionProvider_CUDA(cuda_options);
			}
			catch (Ort::Exception& e) {
				std::cout << e.what();
				std::cout << "Inference device: CPU" << std::endl;
			}
		}
		else
			std::cout << "Inference device: CPU" << std::endl;
#endif
#if defined(USE_TENSORRT)
		auto tensorrt_available = std::find(availableProviders.begin(), availableProviders.end(), "TensorrtExecutionProvider");
		if (tensorrt_available != availableProviders.end()) {
			//OrtTensorRTProviderOptions trt_options{};
			//trt_options.device_id = device_id;
			//trt_options.trt_max_workspace_size = 2147483648;
			//trt_options.trt_max_partition_iterations = 10;
			//trt_options.trt_min_subgraph_size = 5;
			//trt_options.trt_fp16_enable = 1;
			//trt_options.trt_int8_enable = 1;
			//trt_options.trt_int8_use_native_calibration_table = 1;
			const char* trt_engine_cache_path = "./data";
			const char* trt_profile_min_shapes = "images:1x2x1280x1280";
			const char* trt_profile_max_shapes = "images:16x2x1280x1280";
			const char* trt_profile_opt_shapes = "images:8x2x1280x1280";
			const char* pchar = const_cast<char*>(std::to_string(device_id).c_str());
			std::vector<const char*> keys{ "trt_profile_min_shapes", "trt_profile_max_shapes", "trt_profile_opt_shapes",
			"trt_fp16_enable", "trt_int8_enable", "trt_engine_cache_enable", "trt_max_partition_iterations",
			"trt_min_subgraph_size", "trt_engine_decryption_enable", "device_id" };
			std::vector<const char*> values{ trt_profile_min_shapes, trt_profile_max_shapes, trt_profile_opt_shapes,
			"1", "0", "0","10",	"5", "0", pchar };

			const auto& api = Ort::GetApi();
			OrtTensorRTProviderOptionsV2* trt_options;
			CHECK(api.CreateTensorRTProviderOptions(&trt_options) == nullptr);
			std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)>
				rel_trt_options(trt_options, api.ReleaseTensorRTProviderOptions);
			CHECK(api.UpdateTensorRTProviderOptions(rel_trt_options.get(), keys.data(), values.data(), keys.size()) == nullptr);
			try {
				//Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sf, device_id));
				session_options.AppendExecutionProvider_TensorRT_V2(*trt_options);
				//session_options.AppendExecutionProvider_TensorRT(trt_options);
			}
			catch (Ort::Exception& e) {
				LOG(WARNING) << e.what();
			}
			LOG(INFO) << "Inference device: TENSORRT" << std::endl;
		}
		else {
			LOG(WARNING) << "Inference device: CPU" << std::endl;
		}
#endif
		// Sets graph optimization level
		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
		//session_options.SetExecutionMode(ORT_SEQUENTIAL);
		//session_options.DisableMemPattern();
		//session_options.DisableCpuMemArena();
		uint32_t capacity = std::thread::hardware_concurrency();
		if (n_thread >= capacity) {
			std::cout << "Out of capacity! Claimed: " << n_thread << "; Capacity: " << capacity << std::endl;
			n_thread = capacity;
		}

		std::cout << "Thread Count: " << n_thread << "; Ort version: " << Ort::GetVersionString() << std::endl;
		session_options.SetIntraOpNumThreads(n_thread);

		//load model
		try {
			session = Ort::Session(env, model_data, model_data_length, session_options);
		}
		catch (Ort::Exception& e) {
			std::cout << e.what();
		}
		memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

		//get the input & output names and dims
		Ort::AllocatorWithDefaultOptions allocator;
		auto num_input_nodes = session.GetInputCount();
		input_dims.resize(num_input_nodes);
		for (int i = 0; i < num_input_nodes; i++) {
			char* temp_buf = new char[50];
			Ort::AllocatedStringPtr input_node_name = session.GetInputNameAllocated(i, allocator);
			strcpy(temp_buf, input_node_name.get());
			input_names.push_back(temp_buf);

			Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
			auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
			ONNXTensorElementDataType type = tensor_info.GetElementType();
			std::vector<int64_t> input_node_dims = tensor_info.GetShape();
			input_dims[i] = input_node_dims;

			if (input_node_dims[2] == -1 && input_node_dims[3] == -1) {
				std::cout << "Dynamic input shape" << std::endl;
				this->isDynamicInputShape = true;
			}
		}

		auto num_output_nodes = session.GetOutputCount();
		output_dims.resize(num_output_nodes);
		for (int i = 0; i < num_output_nodes; i++) {
			Ort::AllocatedStringPtr output_node_name = session.GetOutputNameAllocated(i, allocator);
			char* temp_buf = new char[10];
			strcpy(temp_buf, output_node_name.get());
			output_names.push_back(temp_buf);

			Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
			auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
			ONNXTensorElementDataType type = tensor_info.GetElementType();
			std::vector<int64_t> output_node_dims = tensor_info.GetShape();
			output_dims[i] = output_node_dims;
		}

#ifdef _DEBUG  
		print();
#endif
	}

	//Print Model Info
	void ORTWrapper::print() {
		auto num_input_nodes = session.GetInputCount();
		auto num_output_nodes = session.GetOutputCount();
		std::cout << "Number of input node is:" << num_input_nodes << std::endl;
		std::cout << "Number of output node is:" << num_output_nodes << std::endl;

		for (auto i = 0; i < num_input_nodes; i++) {
			std::vector<int64_t> input_dims = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
			std::cout << std::endl << "input " << i << " dim is: ";
			for (auto j = 0; j < input_dims.size(); j++)
				std::cout << input_dims[j] << " ";
		}
		for (auto i = 0; i < num_output_nodes; i++)
		{
			std::vector<int64_t> output_dims = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
			std::cout << std::endl << "output " << i << " dim is: ";
			for (auto j = 0; j < output_dims.size(); j++)
				std::cout << output_dims[j] << " ";
		}

		std::cout << std::endl;
		Ort::AllocatorWithDefaultOptions allocator;
		for (auto i = 0; i < num_input_nodes; i++)
			std::cout << "The input op-name " << i << " is:" << session.GetInputNameAllocated(i, allocator) << std::endl;
		for (auto i = 0; i < num_output_nodes; i++)
			std::cout << "The output op-name " << i << " is:" << session.GetInputNameAllocated(i, allocator) << std::endl;
	}

	std::vector<Ort::Value> ORTWrapper::forward(std::vector<float>& input, std::vector<int64_t>& input_dims) {
		int64_t input_size = 1;// input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3];
		for (int ii = 0; ii < input_dims.size(); ii++) {
			input_size *= input_dims[ii];
		}
		auto input_tensor = Ort::Value::CreateTensor<float>(
			memory_info, input.data(), input_size, input_dims.data(), input_dims.size());

		std::vector<Ort::Value>  results;
		try {
			results = session.Run(
				Ort::RunOptions{ nullptr },
				input_names.data(), &input_tensor, input_names.size(),
				output_names.data(), output_names.size()
			);
		}
		catch (Ort::Exception e) {
			std::cout << e.what() << std::endl;
		}

		input_tensor.release();
		return results;


	}

	class InferImpl : public Infer {
	public:
		shared_ptr<ORTWrapper> ort_;
		Type type_;
		float confidence_threshold_;
		float nms_threshold_;
		//vector<shared_ptr<trt::Memory<unsigned char>>> preprocess_buffers_;
		//trt::Memory<float> input_buffer_, bbox_predict_, output_boxarray_;
		//trt::Memory<float> segment_predict_;
		int network_input_width_, network_input_height_;
		//Norm normalize_;
		vector<int> bbox_head_dims_;
		//vector<int> segment_head_dims_;
		int num_classes_ = 0;
		bool has_segment_ = false;
		bool isdynamic_model_ = false;
		//vector<shared_ptr<trt::Memory<unsigned char>>> box_segment_cache_;

		virtual ~InferImpl() = default;

		bool load(const void* model_data, size_t model_data_length, Type type, int device_id, int n_thread, float confidence_threshold, float nms_threshold) {
			ort_ = std::shared_ptr<ORTWrapper>(new ORTWrapper(model_data, model_data_length, device_id, n_thread));
			if (ort_ == nullptr) return false;

			//ort_->print();

			this->type_ = type;
			this->confidence_threshold_ = confidence_threshold;
			this->nms_threshold_ = nms_threshold;

			auto input_dim = ort_->static_dims(0);
			has_segment_ = (type == Type::V8Seg);
			network_input_width_ = input_dim[3];
			network_input_height_ = input_dim[2];
			isdynamic_model_ = ort_->has_dynamic_dim();

			return true;
		}

		void letter_box(const cv::Mat& img, cv::Size new_shape, cv::Mat& dst, float& r, cv::Point& d, cv::Scalar color = cv::Scalar(114, 114, 114),
			bool auto_mode = false, bool scaleup = true, int stride = 32) {
			//# Resizeand pad image while meeting stride - multiple constraints
			float width = float(img.cols), height = float(img.rows);

			//# Scale ratio(new / old)
			r = std::min(new_shape.width / width, new_shape.height / height);
			if (!scaleup) // # only scale down, do not scale up (for better val mAP)
				r = std::min(r, 1.0f);

			//# Compute padding
			int new_unpadW = int(round(width * r));
			int new_unpadH = int(round(height * r));
			int dw = new_shape.width - new_unpadW;  //# wh padding
			int dh = new_shape.height - new_unpadH;
			if (auto_mode) { //# minimum rectangle, wh padding
				dw %= stride;
				dh %= stride;
			}
			dw /= 2, dh /= 2; //# divide padding into 2 sides
			d.x = dw, d.y = dh;

			resize(img, dst, cv::Size(new_unpadW, new_unpadH), 0, 0, cv::INTER_LINEAR);
			int top = int(round(dh - 0.1));
			int bottom = int(round(dh + 0.1));
			int left = int(round(dw - 0.1));
			int right = int(round(dw + 0.1));
			copyMakeBorder(dst, dst, top, bottom, left, right, cv::BORDER_CONSTANT, color); //# add border
		}

		virtual BoxArray forward(const cv::Mat& image) {
			auto output = forwards({ image });
			if (output.empty()) return {};
			return output[0];
		}

		virtual vector<BoxArray> forwards(const vector<cv::Mat>& images) {
			int num_image = images.size();
			if (num_image == 0) return {};

			auto input_dims = ort_->static_dims(0);
			int infer_batch_size = input_dims[0];
			if (infer_batch_size != num_image) {
				if (isdynamic_model_) {
					infer_batch_size = num_image;
					input_dims[0] = num_image;
					//if (!ort_->set_run_dims(0, input_dims)) return {};
				}
				else {
					if (infer_batch_size < num_image) {
						printf(
							"When using static shape model, number of images[%d] must be "
							"less than or equal to the maximum batch[%d].",
							num_image, infer_batch_size);
						return {};
					}
				}
			}

			//pre-processing
			int height = this->network_input_height_, width = this->network_input_width_;
			//int channels = (input_dims[1] != -1) ? input_dims[1] : 3;
			std::vector<float> input_data;
			float ratio = 1.0f; // share fixed resolution
			std::vector<float> ratios;
			std::vector<cv::Point> dxys;
			cv::Point dxy;
			for (auto image : images) {
				cv::Mat dst;
				letter_box(image, cv::Size(width, height), dst, ratio, dxy);
				cv::Mat blob = cv::dnn::blobFromImage(dst, 1 / 255.0, cv::Size(width, height), cv::Scalar(0, 0, 0), true, false).reshape(1, 1);
				std::vector<float> value = (std::vector<float>)(blob);
				input_data.insert(input_data.end(), value.begin(), value.end());
				ratios.push_back(ratio);
				dxys.push_back(dxy);
			}

			//inference
			//int batch_size = static_cast<int>(images.size());
			//input_dims = { batch_size, channels, height, width };
			auto results = ort_->forward(input_data, input_dims);

			//post-processing & output
			auto* rawOutput = results[0].GetTensorMutableData<float>();
			std::vector<int64_t> outputShape = results[0].GetTensorTypeAndShapeInfo().GetShape();
			size_t count = results[0].GetTensorTypeAndShapeInfo().GetElementCount();
			std::vector<float> output(rawOutput, rawOutput + count);
	
			vector<BoxArray> arrout(num_image);
			if (outputShape.size() <= 2) { // end-to-end inference
				for (auto it = output.begin(); it != output.begin() + count; it += outputShape[1]) {  //
					Box box;
					int batch_id = (int)it[0];
					box.left = (it[1] - dxys[batch_id].x) / ratios[batch_id];
					box.top  = (it[2] - dxys[batch_id].y) / ratios[batch_id];
					box.right = (it[3] - dxys[batch_id].x) / ratios[batch_id];
					box.bottom = (it[4] - dxys[batch_id].y) / ratios[batch_id];
					box.left = std::clamp(box.left, 0.f, float(images[batch_id].cols));
					box.top = std::clamp(box.top, 0.f, float(images[batch_id].rows));
					box.right = std::clamp(box.right, 0.f, float(images[batch_id].cols));
					box.bottom = std::clamp(box.bottom, 0.f, float(images[batch_id].rows));
					box.confidence = it[6];
					box.class_label = (int)it[5];
					arrout[batch_id].push_back(box);
				}
			}

			return arrout;
		}
	};

	Infer* loadraw(const void* model_data, size_t model_data_length, Type type, int device_id, int n_thread, float confidence_threshold,
		float nms_threshold) {
		InferImpl* impl = new InferImpl();
		if (!impl->load(model_data, model_data_length, type, device_id, n_thread, confidence_threshold, nms_threshold)) {
			delete impl;
			impl = nullptr;
		}
		return impl;
	}

	std::shared_ptr<Infer> load(const std::string& model_file, Type type, int device_id, int n_thread,
		float confidence_threshold, float nms_threshold) {
		MY_Buffer* buffer = ReadBufferFromFile(model_file.c_str());
		std::shared_ptr<Infer> infer = load(buffer->data, buffer->length, type, device_id, n_thread, confidence_threshold, nms_threshold);
		buffer->data_deallocator((void*)buffer->data);
		return infer;
	}

	shared_ptr<Infer> load(const void* model_data, size_t model_data_length, Type type, int device_id, int n_thread, float confidence_threshold,
		float nms_threshold) {
		return std::shared_ptr<InferImpl>(
			(InferImpl*)loadraw(model_data, model_data_length, type, device_id, n_thread, confidence_threshold, nms_threshold));
	}
}; // namespace yolo_ort