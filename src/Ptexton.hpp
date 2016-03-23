/**
* @file Ptexton.hpp
* @brief This class includes the whole process related to Patch-based textons framework (for PolSAR images)
		- reading image data
		- extract patches
		- random projection
		- training patch-based textons (clustering textons, visualizing texton histogram)
		- testing patch-based textons
		- evaluation of textons	
* @author Eunbi Park
* @date 2015.11.21
*/

#ifndef PTEXTON_HPP
#define PTEXTON_HPP

#include <iostream>
#include <opencv2/opencv.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>

#include "Tools.hpp"
#include "ImgData.h"


using namespace std;
using namespace cv;

class PTexton{
public:
	//! the constructor
	PTexton(string fname, int patchSize, int K, string RP);

	//! the destructor
	~PTexton(void);

	enum TYPE { GRAY = 1, COLOR, POLSAR = 4 };

	//! learning textons
	void learningTexton(void);

	//! training patch-based textons
	void train();
	
	//! testing patch-based textons
	void test(void);

	//! evaluate textons
	void evaluate();

	//! visualize center matrix
	void printCenter(Mat& input);

	//! show and write outputImage and apply colormap
	void showImg(Mat img, const char* win, bool show, bool save);

	//! load image data
	bool loadImageData(void);

	//! load reference data
	void loadReferenceData();

	//! training image type for processing
	int imgType;

	//! file path 
	string fname;

	//! patch size
	int pSize;

	//! K for K-means
	int K;
private:
	
	//! extract feature vector
	Mat extractFVec(string mode,string RP);

	//! calculate feature vector using RP
	void calculateFVecRP(Mat tImg, Mat& featureVec);

	//! calculate feature vector without RP
	void calculateFVecNRP(Mat tImg, Mat& featureVec);

	//! generate random matrix for random projection
	void generateRandomMat(Mat& randomMat, int highD, int lowD, string rMode);

	//! clustering textons using train image
	void clusterTextons(Mat featureVec);	

	//! map textons to each pixel
	void textonMapping(Mat featureVec, Mat& textonMap, string mode);

	//! calculate histogram
	vector<int> trainHist(Mat textonMap);

	//! classification by mapping histogram
	void classification(Mat textonMap);

	//! the image data during training and evaluation
	vector<ImgData*> imageData;
	
	//! the reference data for text (classfication)
	vector<Mat> referenceData;

	//TODO: hist+nPatches => histogram class
	//! number of patches in histogram database of training step
	vector<int> nPatches;

	//! histogram of training
	Mat histDB;

	//! feature vector size
	int vSize;

	//! centers from K-means (k x vector size)
	Mat textons;

	//! current used image
	Mat currentImg;

	//! classification (test) result
	Mat resultImg;

	//! ROI for training
	cv::Rect trainRect;

	//! ROI for testing
	cv::Rect testRect;

	int half_patch;

	string RP;
};
#endif /* PTEXTON_HPP*/