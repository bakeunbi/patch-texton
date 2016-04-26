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
#include <opencv2/opencv.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>

#include "Tools.hpp"
#include "ImgData.h"
//#include "Evaluation.h"


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
	vector<Mat> test(Mat textonMap, vector<Mat> histDB, int fold);

	//! evaluate textons
	void evaluate();
	
	void classification();
	
	//! visualize center matrix
	void printCenter(Mat& input);

	//! show and write outputImage and apply colormap
	void showImg(Mat img, string win, bool show, bool save);

	//! load image data
	bool loadImageData(void);

	//! load reference data
	bool loadReferenceData();

	//! select current image and type
	void selectCurrentImage();

	//! fold data generation
	void foldGeneration();

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

	int half_patch;

	int nfolds;

	string RP;

	int nclass;
private:
	
	//! extract feature vectors
	void generateFVectors(cv::Rect region, int foldN);
	
	//! generate random matrix for random projection
	void generateRandomMat(Mat& randomMat, int highD, int lowD, string rMode);

	//! apply Random Prjection to feature vector
	void RandomProjection(int foldN);
	Mat RandomProjection(Mat target);

	//! clustering textons using train image
	void clusterTextons();

	//! clustering textons using train image by random sampling
	void clusterTextons(int sampling);

	//! measurement of wishart distance
	float wishartDistance(Mat center, Mat comp);

	//! measurement of wishart distance (first term and inverse of center is already calculated)
	float wishartDistance(float firstTerm, Mat invCenter, Mat comp);

	//! map textons to each pixel
	Mat textonMapping(vector<vector<Mat>> tfvectors, vector<vector<Mat>> testonDic);
	
	//! calculate histogram
	//! input: vector<Mat> histogram DB of a fold (Mat is histograms in a class)
	vector<Mat> trainHist(Mat textonMap, int fold);
	
	//! the image data during training and evaluation
	ImgData imageData;
	
	//! the reference data for text (classfication)
	vector<Mat> referenceData;

	//TODO: hist+nPatches => histogram class
	//! number of patches in histogram database of training step
	vector<int> nPatches;

	//! histogram of training
	//Mat histDB;

	/*
	* fVectors includes feature vectors(vector<Mat> fVec)
	* Mat: 3x3 covariance matrix in a pixel if image type is PolSAR
	*		1x3 RGB channel in a pixel if image type is color
	*		1x1 intensity of a pixel if image type is grayscale
	*/
	vector<vector<Mat>> fVectors;

	//! centers from K-means (K textons)
	vector<vector<Mat>> textons;

	//! current used image
	Mat currentImg;

	//! file outstream
	ofstream ofile;

};
#endif /* PTEXTON_HPP*/
