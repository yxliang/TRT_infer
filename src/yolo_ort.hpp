#ifndef __YOLO_ORT_HPP__
#define __YOLO_ORT_HPP__

#include <future>
#include <memory>
#include <string>
#include <vector>

namespace yolo_ort {

	enum class Type : int {
		V5 = 0,
		X = 1,
		V3 = 2,
		V7 = 3,
		V8 = 5,
		V8Seg = 6  // yolov8 instance segmentation
	};

	struct InstanceSegmentMap {
		int width = 0, height = 0;      // width % 8 == 0
		unsigned char* data = nullptr;  // is width * height memory

		InstanceSegmentMap(int width, int height);
		virtual ~InstanceSegmentMap();
	};

	struct Box {
		float left, top, right, bottom, confidence;
		int class_label;
		std::shared_ptr<InstanceSegmentMap> seg;  // valid only in segment task

		Box() = default;
		Box(float left, float top, float right, float bottom, float confidence, int class_label)
			: left(left),
			top(top),
			right(right),
			bottom(bottom),
			confidence(confidence),
			class_label(class_label) {}
	};

	//struct Image {
	//	const void* bgrptr = nullptr;
	//	int width = 0, height = 0;

	//	Image() = default;
	//	Image(const void* bgrptr, int width, int height) : bgrptr(bgrptr), width(width), height(height) {}
	//};

	typedef std::vector<Box> BoxArray;

	class Infer {
	public:
		virtual BoxArray forward(const cv::Mat& image) = 0;
		virtual std::vector<BoxArray> forwards(const std::vector<cv::Mat>& images) = 0;
	};

	std::shared_ptr<Infer> load(const std::string& engine_file, Type type, int device_id = 0, int n_thread = 1,
		float confidence_threshold = 0.25f, float nms_threshold = 0.5f);

	std::shared_ptr<Infer> load(const void* model_data, size_t model_data_length, Type type, int device_id = 0, int n_thread = 1,
		float confidence_threshold = 0.25f, float nms_threshold = 0.5f);

};  // namespace yolo_ort

#endif  // __YOLO_ORT_HPP__