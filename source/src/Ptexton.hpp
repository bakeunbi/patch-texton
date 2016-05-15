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
#include <fstream>
#include <thread>
#include <opencv2/opencv.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>

#include "Tools.hpp"
#include "ImgData.h"
#include "ClassificationError.h"
//#include "Evaluation.h"


using namespace std;
using namespace cv;

class PTexton{
public:
	//! the constructor
	PTexton(){};
	void initialize(string fname, int patchSize, int K, int knn);

	//! the destructor
	~PTexton(void);

	enum TYPE { GRAY = 1, COLOR, POLSAR = 4 };

	//! load image data
	bool loadImageData(void);

	//! load reference data
	bool loadReferenceData();

	//! select current image and type
	void selectCurrentImage();

	//! fold data generation
	void foldGeneration();

	//! learning textons
	void learningTexton(void);

	//! training patch-based textons
	void train();
	
	//! only include test stage
	void test();

	//! evaluate all folds
	void evaluate();
	
	//! read image and compute classification errors
	void errorAssessment();
	
	//! visualize center matrix
	void printCenter(Mat& input);

	void printTextonMap();
	void printResult();

	//! show and write outputImage and apply colormap
	void showImg(Mat img, string win, bool show, bool save);

	//! fold data vector
	vector<cv::Rect> foldRect;

	//! training image type for processing
	int imgType;
	
	//! file path 
	string fname;

	//! patch size
	int pSize;

	//! K for K-means
	int K;

	int knn;

	int half_patch;

	int nfolds;

	string RP;

	int nclass;

private:
	//! clustering textons using train image
	void clusterTextons(vector<vector<Mat>> fVectors, int fold);

	//! extract feature vectors
	vector<vector<Mat>> generateFVectors(cv::Rect region);
	vector<vector<Mat>> generateFVectors(cv::Rect region,int c);

	//! clustering textons using train image by pre-calculation
	void initializeCenters(int sampling);

	//! testing patch-based textons with knn(opencv-Euclidean distance)
	void histMatching(Mat textonMap, vector<Mat> histDB, int fold);

	//! histogram mapping with nearest neigbor using chi-square ditribution
	void histMatchingI(Mat textonMap, vector<Mat> histDB, int fold);

	//! measurement of wishart distance
	float wishartDistance(Mat center, Mat comp);

	//! measurement of wishart distance (first term and inverse of center is already calculated)
	float wishartDistance(float firstTerm, Mat invCenter, Mat comp);

	//! map textons to each pixel
	void textonMapping(vector<vector<Mat>> tfvectors, vector<vector<Mat>> testonDic,int fold, int trainfold);

	//! map textons to each pixel for thread
	void textonMappingT(vector<vector<Mat>> testonDic, int fold, int trainfold);

	//! generate random matrix for random projection
	void generateRandomMat(Mat& randomMat, int highD, int lowD, string rMode);

	//! apply Random Prjection to feature vector
	void RandomProjection(int foldN);
	Mat RandomProjection(Mat target);

	//! calculate histogram
	//! input: vector<Mat> histogram DB of a fold (Mat is histograms in a class)
	vector<Mat> learnHist(Mat textonMap, int fold);
	
	//! the image data during training and evaluation
	ImgData imageData;
	
	//! the reference data for text (classfication)
	vector<Mat> referenceData;

	//TODO: hist+nPatches => histogram class
	//! number of patches in histogram database of training step
	vector<int> nPatches;

	//! histogram of training
	vector<Mat> globalhistDB[5];

	/*
	* fVectors includes feature vectors(vector<Mat> fVec)
	* Mat: 3x3 covariance matrix in a pixel if image type is PolSAR
	*		1x3 RGB channel in a pixel if image type is color
	*		1x1 intensity of a pixel if image type is grayscale
	*/
	//vector<vector<Mat>> fVectors;

	//! centers from K-means (K textons)
	vector<vector<Mat>> textons[5];

	//! current used image
	Mat currentImg;

	//! file outstream
	ofstream ofile;

	thread textonT[5];
	thread mapT[5];
	thread matchT[5];
};
#endif /* PTEXTON_HPP*/
