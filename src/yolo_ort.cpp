#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "yolo_ort.hpp"
#include <fstream>
#include <numeric> 

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

	template <typename T>
	T VectorProduct(const std::vector<T>& v) {
		return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
	};


	std::map<int, std::string> parseString(const std::string& input) {
		std::map<int, std::string> result;
		std::istringstream iss(input);
		char brace;
		iss >> brace;// 忽略开头的'{'
		while (true) {
			int key;
			char colon;
			std::string value, line;
			if (!std::getline(iss, line, ',')) 	break;  // 无法读取整行，可能是因为已经到达字符串末尾

			std::istringstream lineStream(line);
			if (!(lineStream >> key))	break;  // 无法解析键，可能是因为已经到达字符串末尾
			if (!(lineStream >> colon) || colon != ':')  break;  // 无法解析':'，可能是因为已经到达字符串末尾
			std::getline(lineStream >> std::ws, value);		// 读取剩余部分作为值
			result[key] = value;// 将键值对添加到结果中
		}

		if (!result.empty()) {	// 移除末尾可能的 '}' 字符
			auto lastEntry = result.rbegin();
			lastEntry->second.erase(std::remove(lastEntry->second.begin(), lastEntry->second.end(), '}'), lastEntry->second.end());
		}

		return result;
	}

	struct AffineMatrix {
		float i2d[6];  // image to dst(network), 2x3 matrix
		float d2i[6];  // dst to image, 2x3 matrix

		void compute(const std::tuple<int, int>& from, const std::tuple<int, int>& to) {
			float scale_x = get<0>(to) / (float)get<0>(from);
			float scale_y = get<1>(to) / (float)get<1>(from);
			float scale = std::min(scale_x, scale_y);
			i2d[0] = scale;
			i2d[1] = 0;
			i2d[2] = -scale * get<0>(from) * 0.5 + get<0>(to) * 0.5 + scale * 0.5 - 0.5;
			i2d[3] = 0;
			i2d[4] = scale;
			i2d[5] = -scale * get<1>(from) * 0.5 + get<1>(to) * 0.5 + scale * 0.5 - 0.5;

			double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
			D = D != 0. ? double(1.) / D : double(0.);
			double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D, A21 = -i2d[3] * D;
			double b1 = -A11 * i2d[2] - A12 * i2d[5];
			double b2 = -A21 * i2d[2] - A22 * i2d[5];

			d2i[0] = A11;
			d2i[1] = A12;
			d2i[2] = b1;
			d2i[3] = A21;
			d2i[4] = A22;
			d2i[5] = b2;
		}
	};

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
		void load_meta_data();

		virtual std::vector<int64_t> static_dims(int idx) {
			return input_dims[idx];
		}

		virtual std::vector<int64_t> get_intput_dims(int idx) {
			return input_dims[idx];
		}

		virtual std::vector<int64_t> get_output_dims(int idx) {
			return output_dims[idx];
		}

	protected:
		int width{ 640 };
		int height{ 640 };
		Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "Default" };
		Ort::Session session{ nullptr };
		Ort::MemoryInfo memory_info{ nullptr };
		std::vector<const char*> input_names, output_names;
		std::vector< std::vector<int64_t>> input_dims, output_dims;
		bool isDynamicInputShape{ false };
		std::map<int, std::string> classes;
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
			std::cout << "Inference device: TENSORRT" << std::endl;
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
	}

	//Print Model Info
	void ORTWrapper::print() {
		std::cout << "*****************************************************************" << std::endl;
		auto num_input_nodes = session.GetInputCount();
		auto num_output_nodes = session.GetOutputCount();
		std::cout << "Number of input node is:" << num_input_nodes << std::endl;
		std::cout << "Number of output node is:" << num_output_nodes;

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
			std::cout << "The output op-name " << i << " is:" << session.GetOutputNameAllocated(i, allocator) << std::endl;
		std::cout << "*****************************************************************" << std::endl;
	}

	void ORTWrapper::load_meta_data() {
		Ort::AllocatorWithDefaultOptions allocator;
		Ort::ModelMetadata model_metadata = session.GetModelMetadata();
		Ort::AllocatedStringPtr search = model_metadata.LookupCustomMetadataMapAllocated("date", allocator);
		if (search != nullptr) {
			std::cout << "Model version: " << model_metadata.GetVersion() << "; date: " << search.get() << std::endl;
		}
		search = model_metadata.LookupCustomMetadataMapAllocated("imgsz", allocator);
		if (search != nullptr) {
			std::istringstream iss(search.get());
			char bracket, comma;
			iss >> bracket >> this->width >> comma >> this->height >> bracket;
			std::cout << "image size: " << search.get() << std::endl;
		}
		search = model_metadata.LookupCustomMetadataMapAllocated("names", allocator);
		if (search != nullptr) {
			classes = parseString(search.get());
			for (const auto& entry : classes) {
				std::cout << entry.first << " : " << entry.second << std::endl;
			}
		}
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
		vector<int64_t> bbox_head_dims_;
		vector<int64_t> segment_head_dims_;
		int num_classes_ = 0;
		bool has_segment_ = false;
		bool isdynamic_model_ = false;
		//vector<shared_ptr<trt::Memory<unsigned char>>> box_segment_cache_;

		virtual ~InferImpl() = default;

		bool load(const void* model_data, size_t model_data_length, Type type, int device_id, int n_thread, float confidence_threshold, float nms_threshold) {
			ort_ = std::shared_ptr<ORTWrapper>(new ORTWrapper(model_data, model_data_length, device_id, n_thread));
			if (ort_ == nullptr) return false;

#ifdef _DEBUG  
			ort_->print();
			ort_->load_meta_data();
#endif	

			this->type_ = type;
			this->confidence_threshold_ = confidence_threshold;
			this->nms_threshold_ = nms_threshold;

			auto input_dim = ort_->static_dims(0);
			bbox_head_dims_ = ort_->get_output_dims(0);
			has_segment_ = (type == Type::V8Seg);
			if (has_segment_) {
				segment_head_dims_ = ort_->get_output_dims(1);;
			}
			network_input_width_ = input_dim[3];
			network_input_height_ = input_dim[2];
			isdynamic_model_ = ort_->has_dynamic_dim();

			if (type == Type::V5 || type == Type::V3 || type == Type::V7) {
				num_classes_ = bbox_head_dims_[2] - 5;
			}
			else if (type == Type::V8) {
				num_classes_ = bbox_head_dims_[2] - 4;
			}
			else if (type == Type::V8Seg) {
				num_classes_ = bbox_head_dims_[2] - 4 - segment_head_dims_[1];
			}
			else if (type == Type::X) {
				num_classes_ = bbox_head_dims_[2] - 5;
			}
			else {
				std::cout << "Unsupport type " << std::endl;
			}

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

			std::vector<AffineMatrix> affine_matrixs(num_image);

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
			std::vector<cv::Vec4d> params;
			int height = this->network_input_height_, width = this->network_input_width_;

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
				cv::Vec4d param;
				param[0] = (float)width / (float)(image.cols);
				param[1] = (float)height / (float)(image.rows);
				param[2] = dxy.x;
				param[3] = dxy.y;
				params.push_back(param);
			}

			//inference
			auto output_tensors = ort_->forward(input_data, input_dims);

			//post-processing & output
			vector<BoxArray> arrout(num_image);

			float* pdata = output_tensors[0].GetTensorMutableData<float>();
			std::vector<int64_t> _outputTensorShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape(), _outputMaskTensorShape;
			vector<int> mask_protos_shape;
			int mask_protos_length;
			if (has_segment_) {
				_outputMaskTensorShape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
				mask_protos_shape = { 1,(int)_outputMaskTensorShape[1],(int)_outputMaskTensorShape[2],(int)_outputMaskTensorShape[3] };
				mask_protos_length = VectorProduct(mask_protos_shape);
			}
			int64_t one_output_length = VectorProduct(_outputTensorShape) / _outputTensorShape[0];
			int net_width = _outputTensorShape[2];
			int net_height = _outputTensorShape[1];

			for (int img_index = 0; img_index < images.size(); ++img_index) {
				std::vector<int> class_ids;
				std::vector<float> confidences;
				std::vector<cv::Rect> boxes;
				std::vector<vector<float>> picked_proposals;

				for (int r = 0; r < net_height; r++) {    //stride
					cv::Mat scores(1, this->num_classes_, CV_32FC1, pdata + 4);
					Point classIdPoint;
					double max_class_socre;
					minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
					max_class_socre = (float)max_class_socre;

					if (max_class_socre >= this->confidence_threshold_) {
						if (has_segment_) {
							vector<float> temp_proto(pdata + 4 + this->num_classes_, pdata + net_width);
							picked_proposals.push_back(temp_proto);
						}
						float x = (pdata[0] - params[img_index][2]) / params[img_index][0];
						float y = (pdata[1] - params[img_index][3]) / params[img_index][1];
						float w = pdata[2] / params[img_index][0];
						float h = pdata[3] / params[img_index][1];
						int left = MAX(int(x - 0.5 * w + 0.5), 0);
						int top = MAX(int(y - 0.5 * h + 0.5), 0);
						class_ids.push_back(classIdPoint.x);
						confidences.push_back(max_class_socre);
						boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
					}

					pdata += net_width;
				}

				//NMS
				vector<int> nms_result;
				cv::dnn::NMSBoxes(boxes, confidences, this->confidence_threshold_, this->nms_threshold_, nms_result);
				std::vector<vector<float>> temp_mask_proposals;
				yolo_ort::BoxArray box_array;
				Rect holeImgRect(0, 0, images[img_index].cols, images[img_index].rows);
				for (int i = 0; i < nms_result.size(); ++i) {
					int idx = nms_result[i];
					yolo_ort::Box result;
					result.class_label = class_ids[idx];
					result.confidence = confidences[idx];
					boxes[idx] = boxes[idx] & holeImgRect;
					result.top = boxes[idx].tl().y;
					result.left = boxes[idx].tl().x;
					result.bottom = boxes[idx].br().y;
					result.right = boxes[idx].br().x;
					temp_mask_proposals.push_back(picked_proposals[idx]);
					box_array.push_back(result);
				}
				arrout[img_index] = box_array;
				//MaskParams mask_params;
				//mask_params.params = params[img_index];
				//mask_params.srcImgShape = srcImgs[img_index].size();
				//mask_params.netHeight = _netHeight;
				//mask_params.netWidth = _netWidth;
				//mask_params.maskThreshold = _maskThreshold;
				//Mat mask_protos = Mat(mask_protos_shape, CV_32F, output_tensors[1].GetTensorMutableData<float>() + img_index * mask_protos_length);
				//for (int i = 0; i < temp_mask_proposals.size(); ++i) {
				//	GetMask2(Mat(temp_mask_proposals[i]).t(), mask_protos, temp_output[i], mask_params);
				//}


				////******************** ****************
				//// 老版本的方案，如果上面在开启我注释的部分之后还一直报错，建议使用这个。
				//// If the GetMask2() still reports errors , it is recommended to use GetMask().
				//// Mat mask_proposals;
				//// for (int i = 0; i < temp_mask_proposals.size(); ++i) {
				////	mask_proposals.push_back(Mat(temp_mask_proposals[i]).t());
				////}
				////GetMask(mask_proposals, mask_protos, temp_output, mask_params);
				////*****************************************************/
				//output.push_back(temp_output);

			}
			//auto* rawOutput = results[0].GetTensorMutableData<float>();
			//std::vector<int64_t> outputShape = results[0].GetTensorTypeAndShapeInfo().GetShape();
			//size_t count = results[0].GetTensorTypeAndShapeInfo().GetElementCount();
			//std::vector<float> output(rawOutput, rawOutput + count);

			//if (has_segment_) {
			//	auto _outputMaskTensorShape = results[1].GetTensorTypeAndShapeInfo().GetShape();
			//	vector<int> mask_protos_shape = { 1,(int)_outputMaskTensorShape[1],(int)_outputMaskTensorShape[2],(int)_outputMaskTensorShape[3] };
			//	int mask_protos_length = VectorProduct(mask_protos_shape);
			//}

			//vector<BoxArray> arrout(num_image);
			//if (outputShape.size() <= 2) { // end-to-end inference
			//	for (auto it = output.begin(); it != output.begin() + count; it += outputShape[1]) {  //
			//		Box box;
			//		int batch_id = (int)it[0];
			//		box.left = (it[1] - dxys[batch_id].x) / ratios[batch_id];
			//		box.top = (it[2] - dxys[batch_id].y) / ratios[batch_id];
			//		box.right = (it[3] - dxys[batch_id].x) / ratios[batch_id];
			//		box.bottom = (it[4] - dxys[batch_id].y) / ratios[batch_id];
			//		box.left = std::clamp(box.left, 0.f, float(images[batch_id].cols));
			//		box.top = std::clamp(box.top, 0.f, float(images[batch_id].rows));
			//		box.right = std::clamp(box.right, 0.f, float(images[batch_id].cols));
			//		box.bottom = std::clamp(box.bottom, 0.f, float(images[batch_id].rows));
			//		box.confidence = it[6];
			//		box.class_label = (int)it[5];
			//		arrout[batch_id].push_back(box);
			//	}
			//}

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