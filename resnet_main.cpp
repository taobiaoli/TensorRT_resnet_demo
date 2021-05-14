// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/5/6
#include "resnet.h"
#include <io.h>

#include "imageResize.h"
#include <opencv2/core.hpp>
#include<opencv2/opencv.hpp>

nvjpegBackend_t impl = NVJPEG_BACKEND_GPU_HYBRID; //NVJPEG_BACKEND_DEFAULT;
nvjpegHandle_t nvjpeg_handle;
nvjpegJpegStream_t nvjpeg_jpeg_stream;
nvjpegDecodeParams_t nvjpeg_decode_params;
nvjpegJpegState_t nvjpeg_decoder_state;
nvjpegEncoderParams_t nvjpeg_encode_params;
nvjpegEncoderState_t nvjpeg_encoder_state;

void initInputParams(common::InputParams &inputParams) {
	inputParams.ImgH = 224;
	inputParams.ImgW = 224;
	inputParams.ImgC = 3;
	inputParams.workers = 8;
	inputParams.BatchSize = 16;
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
	trtParams.CalibrationTablePath = "D:/tensorRT/resnet_main/resnet_main/data/resnet50_opset9_0401_optimizer_sim.calibration";
	trtParams.CalibrationImageDir = "D:/tensorRT/resnet_main/resnet_main/data/20191210_083147.imgs";
	trtParams.OnnxPath = "D:/tensorRT/resnet_main/resnet_main/data/onnx/resnet50_opset9_0401_batch16_fixinput_optimizer_output_shape.onnx";
	trtParams.SerializedPath = "D:/tensorRT/resnet_main/resnet_main/data/onnx/resnet50_opset9_0401_batch16_fixinput_optimizer_output_shape.serialized";
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
	printf("Cid is %d, Prob is %f\n", cid, max_prob);
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

std::vector<float> decodeResizeOneImage(std::string sImagePath, double &time, int resizeWidth, int resizeHeight, int resize_quality)
{
	// Decode, Encoder format
	nvjpegOutputFormat_t oformat = NVJPEG_OUTPUT_BGR;
	nvjpegInputFormat_t iformat = NVJPEG_INPUT_BGR;

	cv::Mat img2(resizeWidth, resizeWidth, CV_8UC3);
	std::vector<float> result(resizeWidth*resizeWidth * 3);
	// timing for resize
	time = 0.;
	float resize_time = 0.;
	cudaEvent_t start, stop;
	CHECK_CUDA(cudaEventCreate(&start));
	CHECK_CUDA(cudaEventCreate(&stop));

	// Image reading section
	// Get the file name, without extension.
	// This will be used to rename the output file.
	size_t position = sImagePath.rfind("/");
	std::string sFileName = (std::string::npos == position) ? sImagePath : sImagePath.substr(position + 1, sImagePath.size());
	position = sFileName.rfind(".");
	sFileName = (std::string::npos == position) ? sFileName : sFileName.substr(0, position);

#ifndef _WIN64
	position = sFileName.rfind("/");
	sFileName = (std::string::npos == position) ? sFileName : sFileName.substr(position + 1, sFileName.length());
#else
	position = sFileName.rfind("\\");
	sFileName = (std::string::npos == position) ? sFileName : sFileName.substr(position + 1, sFileName.length());
#endif

	// Read an image from disk.
	std::ifstream oInputStream(sImagePath.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
	if (!(oInputStream.is_open()))
	{
		std::cerr << "Cannot open image: " << sImagePath << std::endl;
		return result;
	}

	// Get the size.
	std::streamsize nSize = oInputStream.tellg();
	oInputStream.seekg(0, std::ios::beg);

	// Image buffers. 
	unsigned char * pBuffer = NULL;
	unsigned char * pResizeBuffer = NULL;

	std::vector<char> vBuffer(nSize);

	//std::vector<unsigned char*> filedata;
	if (oInputStream.read(vBuffer.data(), nSize))
	{
		unsigned char * dpImage = (unsigned char *)vBuffer.data();

		// Retrieve the componenet and size info.
		int nComponent = 0;
		nvjpegChromaSubsampling_t subsampling;
		int widths[NVJPEG_MAX_COMPONENT];
		int heights[NVJPEG_MAX_COMPONENT];
		int nReturnCode = 0;
		if (NVJPEG_STATUS_SUCCESS != nvjpegGetImageInfo(nvjpeg_handle, dpImage, nSize, &nComponent, &subsampling, widths, heights))
		{
			std::cerr << "Error decoding JPEG header: " << sImagePath << std::endl;
			return result;
		}

		if (resizeWidth == 0 || resizeHeight == 0)
		{
			resizeWidth = widths[0] / 2;
			resizeHeight = heights[0] / 2;
		}

		// image resize
		size_t pitchDesc, pitchResize;
		NppiSize srcSize = { (int)widths[0], (int)heights[0] };
		NppiRect srcRoi = { 0, 0, srcSize.width, srcSize.height };
		NppiSize dstSize = { (int)resizeWidth, (int)resizeHeight };
		NppiRect dstRoi = { 0, 0, dstSize.width, dstSize.height };
		NppStatus st;
		NppStreamContext nppStreamCtx;
		nppStreamCtx.hStream = NULL; // default stream

		// device image buffers.
		nvjpegImage_t imgDesc;
		nvjpegImage_t imgResize;

		if (is_interleaved(oformat))
		{
			pitchDesc = NVJPEG_MAX_COMPONENT * widths[0];
			pitchResize = NVJPEG_MAX_COMPONENT * resizeWidth;
		}
		else
		{
			pitchDesc = 3 * widths[0];
			pitchResize = 3 * resizeWidth;
		}

		cudaError_t eCopy = cudaMalloc(&pBuffer, pitchDesc * heights[0]);
		if (cudaSuccess != eCopy)
		{
			std::cerr << "cudaMalloc failed : " << cudaGetErrorString(eCopy) << std::endl;
			return result;
		}
		cudaError_t eCopy1 = cudaMalloc(&pResizeBuffer, pitchResize * resizeHeight);
		if (cudaSuccess != eCopy1)
		{
			std::cerr << "cudaMalloc failed : " << cudaGetErrorString(eCopy) << std::endl;
			return result;
		}


		imgDesc.channel[0] = pBuffer;
		imgDesc.channel[1] = pBuffer + widths[0] * heights[0];
		imgDesc.channel[2] = pBuffer + widths[0] * heights[0] * 2;
		imgDesc.pitch[0] = (unsigned int)(is_interleaved(oformat) ? widths[0] * NVJPEG_MAX_COMPONENT : widths[0]);
		imgDesc.pitch[1] = (unsigned int)widths[0];
		imgDesc.pitch[2] = (unsigned int)widths[0];

		imgResize.channel[0] = pResizeBuffer;
		imgResize.channel[1] = pResizeBuffer + resizeWidth * resizeHeight;
		imgResize.channel[2] = pResizeBuffer + resizeWidth * resizeHeight * 2;
		imgResize.pitch[0] = (unsigned int)(is_interleaved(oformat) ? resizeWidth * NVJPEG_MAX_COMPONENT : resizeWidth);;
		imgResize.pitch[1] = (unsigned int)resizeWidth;
		imgResize.pitch[2] = (unsigned int)resizeWidth;

		if (is_interleaved(oformat))
		{
			imgDesc.channel[3] = pBuffer + widths[0] * heights[0] * 3;
			imgDesc.pitch[3] = (unsigned int)widths[0];
			imgResize.channel[3] = pResizeBuffer + resizeWidth * resizeHeight * 3;
			imgResize.pitch[3] = (unsigned int)resizeWidth;
		}
		// nvJPEG encoder parameter setting
		CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(nvjpeg_encode_params, resize_quality, NULL));

#ifdef OPTIMIZED_HUFFMAN  // Optimized Huffman
		CHECK_NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(nvjpeg_encode_params, 1, NULL));
#endif
		CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(nvjpeg_encode_params, subsampling, NULL));


		// Timing start
		CHECK_CUDA(cudaEventRecord(start, 0));

#ifdef CUDA10U2 // This part needs CUDA 10.1 Update 2
		//parse image save metadata in jpegStream structure
		CHECK_NVJPEG(nvjpegJpegStreamParse(nvjpeg_handle, dpImage, nSize, 1, 0, nvjpeg_jpeg_stream));
#endif

		// decode by stages
		nReturnCode = nvjpegDecode(nvjpeg_handle, nvjpeg_decoder_state, dpImage, nSize, oformat, &imgDesc, NULL);
		if (nReturnCode != 0)
		{
			std::cerr << "Error in nvjpegDecode." << nReturnCode << std::endl;
			return result;
		}

		// image resize
		/* Note: this is the simplest resizing function from NPP. */
		if (is_interleaved(oformat))
		{
			st = nppiResize_8u_C3R_Ctx(imgDesc.channel[0], imgDesc.pitch[0], srcSize, srcRoi,
				imgResize.channel[0], imgResize.pitch[0], dstSize, dstRoi, NPPI_INTER_LANCZOS, nppStreamCtx);
		}
		else
		{
			st = nppiResize_8u_C1R_Ctx(imgDesc.channel[0], imgDesc.pitch[0], srcSize, srcRoi,
				imgResize.channel[0], imgResize.pitch[0], dstSize, dstRoi, NPPI_INTER_LANCZOS, nppStreamCtx);
			st = nppiResize_8u_C1R_Ctx(imgDesc.channel[1], imgDesc.pitch[1], srcSize, srcRoi,
				imgResize.channel[1], imgResize.pitch[1], dstSize, dstRoi, NPPI_INTER_LANCZOS, nppStreamCtx);
			st = nppiResize_8u_C1R_Ctx(imgDesc.channel[2], imgDesc.pitch[2], srcSize, srcRoi,
				imgResize.channel[2], imgResize.pitch[2], dstSize, dstRoi, NPPI_INTER_LANCZOS, nppStreamCtx);
		}

		if (st != NPP_SUCCESS)
		{
			std::cerr << "NPP resize failed : " << st << std::endl;
			return result;
		}


		cv::Mat img_b(dstSize.width, dstSize.height, CV_8UC1);
		cv::Mat img_g(dstSize.width, dstSize.height, CV_8UC1);
		cv::Mat img_r(dstSize.width, dstSize.height, CV_8UC1);
		cv::Mat img2(dstSize.width, dstSize.height, CV_8UC3);
		std::vector<cv::Mat> bgrChannels;
		//std::vector<unsigned char>image2(imgResize.pitch[0] * imgResize.pitch[1]);

		cudaMemcpy(img_b.data, imgResize.channel[0], imgResize.pitch[0] * imgResize.pitch[1], cudaMemcpyDeviceToHost);
		bgrChannels.push_back(img_b);

		cudaMemcpy(img_g.data, imgResize.channel[1], imgResize.pitch[0] * imgResize.pitch[1], cudaMemcpyDeviceToHost);
		bgrChannels.push_back(img_g);

		cudaMemcpy(img_r.data, imgResize.channel[2], imgResize.pitch[0] * imgResize.pitch[1], cudaMemcpyDeviceToHost);
		bgrChannels.push_back(img_r);

		//if return cv::mat 
		//hwc to chw and normalize return std::vector<float>
		cv::merge(bgrChannels, img2);
		cv::Mat img_float;
		img2.convertTo(img_float, CV_32FC3, 1 / 255.0);//normalize
		std::vector<cv::Mat> input_channels(3);
		cv::split(img_float, input_channels);
		
		float *data = result.data();
		int channelLength = dstSize.height*dstSize.width;
		for (int i = 0; i < 3; ++i) {
			memcpy(data, input_channels[i].data, channelLength * sizeof(float));
			data+= channelLength;
	    }
		//std::cout << img2.channels() << std::endl;
		//cv::imwrite("./rzImage_npp02.jpg", img2);

		// get encoding from the jpeg stream and copy it to the encode parameters
#ifdef CUDA10U2 // This part needs CUDA 10.1 Update 2 for copy the metadata other information from base image.
		CHECK_NVJPEG(nvjpegJpegStreamGetJpegEncoding(nvjpeg_jpeg_stream, &nvjpeg_encoding));
		CHECK_NVJPEG(nvjpegEncoderParamsSetEncoding(nvjpeg_encode_params, nvjpeg_encoding, NULL));
		CHECK_NVJPEG(nvjpegEncoderParamsCopyQuantizationTables(nvjpeg_encode_params, nvjpeg_jpeg_stream, NULL));
		CHECK_NVJPEG(nvjpegEncoderParamsCopyHuffmanTables(nvjpeg_encoder_state, nvjpeg_encode_params, nvjpeg_jpeg_stream, NULL));
		CHECK_NVJPEG(nvjpegEncoderParamsCopyMetadata(nvjpeg_encoder_state, nvjpeg_encode_params, nvjpeg_jpeg_stream, NULL));
#endif
		// Timing stop
		CHECK_CUDA(cudaEventRecord(stop, 0));
		CHECK_CUDA(cudaEventSynchronize(stop));

	}
	// Free memory
	CHECK_CUDA(cudaFree(pBuffer));
	CHECK_CUDA(cudaFree(pResizeBuffer));

	// get timing
	CHECK_CUDA(cudaEventElapsedTime(&resize_time, start, stop));
	time = (double)resize_time;

	return result;
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
	cv::Mat image = cv::imread("D:\\tensorRT\\resnet_main\\resnet_main\\data\\20191210_083147.imgs\\0000_0026.jpg");

	const auto start_t = std::chrono::high_resolution_clock::now();
	std::vector<float> prob = resnet.predOneImage(std::vector<cv::Mat>{image});
	const auto end_t = std::chrono::high_resolution_clock::now();
	std::cout
		<< "Wall clock time passed: "
		<< std::chrono::duration<double, std::milli>(end_t - start_t).count() << "ms"
		<< std::endl;
	getMaxProb(prob);
	*/
	nvjpegDevAllocator_t dev_allocator = { &dev_malloc, &dev_free };
	CHECK_NVJPEG(nvjpegCreate(impl, &dev_allocator, &nvjpeg_handle));
	CHECK_NVJPEG(nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_decoder_state));

	// create bitstream object
	CHECK_NVJPEG(nvjpegJpegStreamCreate(nvjpeg_handle, &nvjpeg_jpeg_stream));
	CHECK_NVJPEG(nvjpegDecodeParamsCreate(nvjpeg_handle, &nvjpeg_decode_params));
	CHECK_NVJPEG(nvjpegEncoderStateCreate(nvjpeg_handle, &nvjpeg_encoder_state, NULL));
	CHECK_NVJPEG(nvjpegEncoderParamsCreate(nvjpeg_handle, &nvjpeg_encode_params, NULL));

	// batch image inference 
	// input image preprocess
	std::vector<std::string> files;
    char *filePath = (char*)"D:\\tensorRT\\resnet_main\\resnet_main\\data\\20191210_083147.imgs";
	////获取该路径下的所有文件
	getFiles(filePath, files);

	char str[30];
	int size = files.size();
	std::vector<std::vector<float>> imagevector;
	for (int i = 0; i < size; i = i+ inputParams.BatchSize)
	{
		const auto start_t = std::chrono::high_resolution_clock::now();
		for (int j = 0; j < inputParams.BatchSize; j++)
		{
			std::cout << files[i+j].c_str() << std::endl;
			//opencv decode
			//cv::Mat image = cv::imread(files[i+j].c_str());

			//nvjpeg docode and resize
			double time = 0.;
			std::vector<float> image = decodeResizeOneImage(files[i + j].c_str(), time, inputParams.ImgH, inputParams.ImgW, 85);
			imagevector.push_back(image);
		}
		//const auto end_t = std::chrono::high_resolution_clock::now();
		//std::cout << "read image cost:" << std::chrono::duration<double, std::milli>(end_t - start_t).count() << "ms" << std::endl;
		//const auto start_t = std::chrono::high_resolution_clock::now();
		//std::vector<float> prob = resnet.batchImage(imagevector);
		std::vector<float> prob = resnet.batchImage(imagevector);
		//const auto end_t = std::chrono::high_resolution_clock::now();
		//std::cout
		//	<< "image preprocessing time passed: "
		//	<< std::chrono::duration<double, std::milli>(end_t - start_t).count() << "ms"
		//	<< std::endl;
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
		const auto end_t = std::chrono::high_resolution_clock::now();
		std::cout << "inference image cost:" << std::chrono::duration<double, std::milli>(end_t - start_t).count() << "ms" << std::endl;
	}
	
	CHECK_NVJPEG(nvjpegEncoderParamsDestroy(nvjpeg_encode_params));
	CHECK_NVJPEG(nvjpegDecodeParamsDestroy(nvjpeg_decode_params));
	CHECK_NVJPEG(nvjpegEncoderStateDestroy(nvjpeg_encoder_state));
	CHECK_NVJPEG(nvjpegJpegStateDestroy(nvjpeg_decoder_state));
	CHECK_NVJPEG(nvjpegDestroy(nvjpeg_handle));
	return 0;
}
