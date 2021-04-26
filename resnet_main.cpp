// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/5/6
#include "resnet.h"
#include <io.h>
void initInputParams(common::InputParams &inputParams) {
	inputParams.ImgH = 224;
	inputParams.ImgW = 224;
	inputParams.ImgC = 3;
	inputParams.workers = 8;
	inputParams.BatchSize = 8;
	inputParams.IsPadding = false;
	inputParams.InputTensorNames = std::vector<std::string>{ "input_1" };
	inputParams.OutputTensorNames = std::vector<std::string>{ "dense_1/Softmax:0" };
	inputParams.pFunction = [](const unsigned char &x) {return static_cast<float>(x) / 255; };
}

void initTrtParams(common::TrtParams &trtParams) {
	trtParams.ExtraWorkSpace = 0;
	trtParams.FP32 = true;
	trtParams.FP16 = false;
	trtParams.Int32 = false;
	trtParams.Int8 = false;
	trtParams.MaxBatch = 100;
	trtParams.MinTimingIteration = 1;
	trtParams.AvgTimingIteration = 2;
	trtParams.CalibrationTablePath = "D:/tensorRT/resnet_main/resnet_main/data/resnet50_opset9_0401_batch16_fixinput_optimizer.calibration";
	trtParams.CalibrationImageDir = "";
	trtParams.OnnxPath = "D:/tensorRT/resnet_main/resnet_main/data/onnx/resnet50_opset9_0401_batch16_fixinput_optimizer.onnx";
	trtParams.SerializedPath = "D:/tensorRT/resnet_main/resnet_main/data/onnx/resnet50_opset9_0401_batch16_fixinput_optimizer.serialized";
}

void initClassificationParams(common::ClassificationParams &classifactionParams) {
	classifactionParams.NumClass = 5;
}

int getMaxProb(const std::vector<float> &prob) {
	int cid = 0;
	float max_prob = 0;
	for (auto i = 0; i < prob.size(); ++i) {
		//printf("cid ===> %d   prob ===> %f\n", i, prob[i]);
		if (max_prob < prob[i]) {
			max_prob = prob[i];
			cid = i;
		}
	}
	//printf("Cid is %d, Prob is %f\n", cid, max_prob);
	return 0;
}
void getFiles(std::string path, std::vector<std::string>& files)
{
	/*files存储文件的路径及名称(eg.   C:\Users\WUQP\Desktop\test_devided\data1.txt)
	 ownname只存储文件的名称(eg.     data1.txt)*/

	 //文件句柄
	long long  hFile = 0;
	//文件信息
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之
			//如果不是,加入列表
			if ((fileinfo.attrib &  _A_SUBDIR))
			{  
				if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)
					getFiles( p.assign(path).append("\\").append(fileinfo.name), files); 
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

int main(int args, char **argv) {
	common::InputParams inputParams;
	common::TrtParams trtParams;
	common::ClassificationParams classifactionParams;
	initInputParams(inputParams);
	initTrtParams(trtParams);
	initClassificationParams(classifactionParams);

	Resnet resnet(inputParams, trtParams, classifactionParams);
	resnet.initSession(0);
	/*
	// inference one image
	cv::Mat image = cv::imread("D:\\tensorRT\\resnet_main\\resnet_main\\data\\20191210_083147.imgs\\0000_0000.jpg");

	const auto start_t = std::chrono::high_resolution_clock::now();
	std::vector<float> prob = resnet.predOneImage(image);
	const auto end_t = std::chrono::high_resolution_clock::now();
	std::cout
		<< "Wall clock time passed: "
		<< std::chrono::duration<double, std::milli>(end_t - start_t).count() << "ms"
		<< std::endl;
	getMaxProb(prob);
	*/
	// batch image inference 
	// input image preprocess
	std::vector<std::string> files;
    char *filePath = (char*)"D:\\tensorRT\\resnet_main\\resnet_main\\data\\20191210_083147.imgs";
	////获取该路径下的所有文件
	getFiles(filePath, files);

	char str[30];
	int size = files.size();
	std::vector<cv::Mat> imagevector;
	for (int i = 0; i < size; i = i+ inputParams.BatchSize)
	{
		for (int j = 0; j < inputParams.BatchSize; j++)
		{
			//std::cout << files[i+j].c_str() << std::endl;
			cv::Mat image = cv::imread(files[i+j].c_str());

			imagevector.push_back(image);
		}
		
		const auto start_t = std::chrono::high_resolution_clock::now();
		std::vector<float> prob = resnet.batchImage(imagevector);
		const auto end_t = std::chrono::high_resolution_clock::now();
		std::cout
			<< "image preprocessing time passed: "
			<< std::chrono::duration<double, std::milli>(end_t - start_t).count() << "ms"
			<< std::endl;
		std::vector<float> prob_m;
		for (int j = 0; j < inputParams.BatchSize; j++)
		{
			for (int i = 0; i < classifactionParams.NumClass; i++)
			{
				int index = classifactionParams.NumClass * j + i;
				prob_m.push_back(prob[index]);
			}
			getMaxProb(prob_m);
			prob_m.clear();
		}

		
		imagevector.clear();
	}


	return 0;
}
