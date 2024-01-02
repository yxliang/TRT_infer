
#include <opencv2/opencv.hpp>

#include "cpm.hpp"
#include "infer.hpp"
#include "yolo.hpp"
#include "yolo_ort.hpp"
#include <filesystem>

using namespace std;

static const char* cocolabels[] = { "person",        "bicycle",      "car",
								   "motorcycle",    "airplane",     "bus",
								   "train",         "truck",        "boat",
								   "traffic light", "fire hydrant", "stop sign",
								   "parking meter", "bench",        "bird",
								   "cat",           "dog",          "horse",
								   "sheep",         "cow",          "elephant",
								   "bear",          "zebra",        "giraffe",
								   "backpack",      "umbrella",     "handbag",
								   "tie",           "suitcase",     "frisbee",
								   "skis",          "snowboard",    "sports ball",
								   "kite",          "baseball bat", "baseball glove",
								   "skateboard",    "surfboard",    "tennis racket",
								   "bottle",        "wine glass",   "cup",
								   "fork",          "knife",        "spoon",
								   "bowl",          "banana",       "apple",
								   "sandwich",      "orange",       "broccoli",
								   "carrot",        "hot dog",      "pizza",
								   "donut",         "cake",         "chair",
								   "couch",         "potted plant", "bed",
								   "dining table",  "toilet",       "tv",
								   "laptop",        "mouse",        "remote",
								   "keyboard",      "cell phone",   "microwave",
								   "oven",          "toaster",      "sink",
								   "refrigerator",  "book",         "clock",
								   "vase",          "scissors",     "teddy bear",
								   "hair drier",    "toothbrush" };

yolo::Image cvimg(const cv::Mat& image) { return yolo::Image(image.data, image.cols, image.rows); }

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

// letter box
void letter_box(const cv::Mat& image, cv::Size new_shape, cv::Mat& dst, cv::Scalar color = cv::Scalar(114, 114, 114)) {

	float width = float(image.cols), height = float(image.rows);
	float* input_data_host = (float*)malloc(3 * new_shape.width * new_shape.height * sizeof(float));

	float scale_x = new_shape.width / (float)image.cols;
	float scale_y = new_shape.height / (float)image.rows;
	float scale = std::min(scale_x, scale_y);
	float i2d[6], d2i[6];

	// d2i 是为了后续坐标映射回去
	i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * image.cols + new_shape.width + scale - 1) * 0.5;
	i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * image.rows + new_shape.height + scale - 1) * 0.5;

	cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);

	dst = cv::Mat(new_shape.height, new_shape.width, CV_8UC3);
	cv::warpAffine(image, dst, m2x3_i2d, dst.size(), cv::INTER_LINEAR,
		cv::BORDER_CONSTANT, color);
}

void perf() {
	int max_infer_batch = 8;
	int batch = 8;
	std::vector<cv::Mat> images{ cv::imread("../workspace/inference/car.jpg"), cv::imread("../workspace/inference/gril.jpg"),
								cv::imread("../workspace/inference/group.jpg") };

	for (int i = images.size(); i < batch; ++i) images.push_back(images[i % 3]);
	int64 t;

	//for (int i = 0; i < 1000; ++i) {
	//	for (auto image : images) {
	//		cv::Mat dst;
	//		t = cv::getTickCount();
	//		letter_box(image, cv::Size(1280, 1280), dst);
	//		std::cout << "\t warpAffine time elapse: " << (double)(cv::getTickCount() - t) / cv::getTickFrequency() * 1000 << "ms." << endl;

	//		cv::Mat dst2; float r; cv::Point d;
	//		t = cv::getTickCount();
	//		letter_box(image, cv::Size(1280, 1280), dst2, r, d);
	//		std::cout << "\t resize time elapse: " << (double)(cv::getTickCount() - t) / cv::getTickFrequency() * 1000 << "ms." << endl;
	//		dst.release(); dst2.release();
	//	}
	//}
	//cv::waitKey(0);

	cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;
	std::string engine_file = "../workspace/yolov8x.transd.engine"; //../workspace/yolov8n.transd.engine ;"../workspace/yolov7.engine";  //
	bool ok = cpmi.start([engine_file]() { return yolo::load(engine_file, yolo::Type::V7); },
		max_infer_batch);

	//cpm::Instance<yolo_ort::BoxArray, cv::Mat, yolo_ort::Infer> cpmi;
	//std::string onnx_file = "../workspace/yolov7.onnx";
	//int device_id = 0, n_thread = 8;
	//bool ok = cpmi.start([onnx_file, device_id, n_thread]() { return yolo_ort::load(onnx_file, yolo_ort::Type::V7, device_id, n_thread); },
	//	max_infer_batch);

	if (!ok) return;

	std::vector<yolo::Image> yoloimages(images.size());
	std::transform(images.begin(), images.end(), yoloimages.begin(), cvimg);

	//trt::Timer timer;		
	std::vector<std::shared_future<yolo::BoxArray>> results;
#ifdef _DEBUG
	for (int i = 0; i < 10; ++i) {
#else
	for (int i = 0; i < 1000; ++i) {
#endif
		//timer.start();
		t = cv::getTickCount();
		//cpmi.commits(images).back().get();
		//std::vector<std::shared_future<yolo_ort::BoxArray>> batched_result = cpmi.commits(images);
		auto batched_result = cpmi.commits(yoloimages);
		auto last_result = batched_result.back().get();
		std::cout << "\t BATCH16 time elapse: " << (double)(cv::getTickCount() - t) / cv::getTickFrequency() * 1000 << "ms." << endl;
		for (auto result : batched_result) {
			results.emplace_back(result);
		}
		//timer.stop("BATCH16");
	}

	for (int ib = 0; ib < (int)results.size(); ++ib) {
		int idx = ib % batch;
		auto& objs = results[idx].get();
		auto image = images[idx].clone();
		for (auto& obj : objs) {
			uint8_t b, g, r;
			tie(b, g, r) = yolo::random_color(obj.class_label);
			cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
				cv::Scalar(b, g, r), 5);

			auto name = cocolabels[obj.class_label];
			auto caption = cv::format("%s %.2f", name, obj.confidence);
			int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
			cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
				cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
			cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2,
				16);
		}
	}

	for (int i = 0; i < 50; ++i) {
		//timer.start();
		t = cv::getTickCount();
		auto result = cpmi.commit(yoloimages[0]).get();
		//cpmi.commit(images[0]).get();
		std::cout << "\t BATCH1 time elapse: " << (double)(cv::getTickCount() - t) / cv::getTickFrequency() * 1000 << "ms." << endl;
		//timer.stop("BATCH1");
	}
	}

void batch_inference() {
	std::vector<cv::Mat> images{ cv::imread("../workspace/inference/car.jpg"), cv::imread("../workspace/inference/gril.jpg"),
								cv::imread("../workspace/inference/group.jpg") };
	auto yolo = yolo::load("../workspace/yolov8n.transd.engine", yolo::Type::V8);
	if (yolo == nullptr) return;

	std::vector<yolo::Image> yoloimages(images.size());
	std::transform(images.begin(), images.end(), yoloimages.begin(), cvimg);
	auto batched_result = yolo->forwards(yoloimages);
	for (int ib = 0; ib < (int)batched_result.size(); ++ib) {
		auto& objs = batched_result[ib];
		auto& image = images[ib];
		for (auto& obj : objs) {
			uint8_t b, g, r;
			tie(b, g, r) = yolo::random_color(obj.class_label);
			cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
				cv::Scalar(b, g, r), 5);

			auto name = cocolabels[obj.class_label];
			auto caption = cv::format("%s %.2f", name, obj.confidence);
			int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
			cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
				cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
			cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2,
				16);
		}
		printf("Save result to Result.jpg, %d objects\n", (int)objs.size());
		cv::imwrite(cv::format("Result%d.jpg", ib), image);
	}
}

void single_inference() {
	cv::Mat image = cv::imread("../workspace/inference/car.jpg");
	auto yolo = yolo::load("../workspace/yolov8n-seg.b1.transd.engine", yolo::Type::V8Seg);
	if (yolo == nullptr) return;

	auto objs = yolo->forward(cvimg(image));
	int i = 0;
	for (auto& obj : objs) {
		uint8_t b, g, r;
		tie(b, g, r) = yolo::random_color(obj.class_label);
		cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
			cv::Scalar(b, g, r), 5);

		auto name = cocolabels[obj.class_label];
		auto caption = cv::format("%s %.2f", name, obj.confidence);
		int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
		cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33),
			cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
		cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);

		if (obj.seg) {
			cv::imwrite(cv::format("%d_mask.jpg", i),
				cv::Mat(obj.seg->height, obj.seg->width, CV_8U, obj.seg->data));
			i++;
		}
	}

	printf("Save result to Result.jpg, %d objects\n", (int)objs.size());
	cv::imwrite("Result.jpg", image);
}

namespace fs = std::filesystem;
int main() {
	std::cout << "Current path is " << fs::current_path() << '\n';
	perf();
	//batch_inference();
	//single_inference();
	return 0;
}