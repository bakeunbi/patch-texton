/*
* @file Ptexton.cpp
* @brief This class includes the whole process related to Patch-based textons framework (for PolSAR images)
* @author Eunbi Park
* @date 2015.11.21
*/

#include "Ptexton.hpp"
#include <direct.h>
#include <Windows.h>

std::vector<std::string> GetFileNamesInDirectory(std::string directory);

/*
* author: Eunbi Park
* date	: 07.04.2016
* input	: Region
* output: Mat wMap
* contents
*		implement Wishart distance measure and save distance map to tImg
*/
//calculate determinant of complex matrix
float determinant_comp(Mat complex){
	float det = 0;

	Mat com[2];
	split(complex, com);

	det = complex.at<Vec2f>(0, 0)[0] * complex.at<Vec2f>(1, 1)[0] * complex.at<Vec2f>(2, 2)[0]
		+ 2 * complex.at<Vec2f>(0, 1)[0] * complex.at<Vec2f>(0, 2)[0] * complex.at<Vec2f>(1, 2)[0]
		- 2 * complex.at<Vec2f>(0, 1)[1] * complex.at<Vec2f>(0, 2)[0] * complex.at<Vec2f>(1, 2)[1]
		+ 2 * complex.at<Vec2f>(0, 1)[1] * complex.at<Vec2f>(0, 2)[1] * complex.at<Vec2f>(1, 2)[0]
		+ 2 * complex.at<Vec2f>(0, 1)[0] * complex.at<Vec2f>(0, 2)[1] * complex.at<Vec2f>(1, 2)[1]
		- complex.at<Vec2f>(1, 1)[0] * (complex.at<Vec2f>(0, 2)[0] * complex.at<Vec2f>(0, 2)[0] + complex.at<Vec2f>(0, 2)[1] * complex.at<Vec2f>(0, 2)[1])
		- complex.at<Vec2f>(2, 2)[0] * (complex.at<Vec2f>(0, 1)[0] * complex.at<Vec2f>(0, 1)[0] + complex.at<Vec2f>(0, 1)[1] * complex.at<Vec2f>(0, 1)[1])
		- complex.at<Vec2f>(0, 0)[0] * (complex.at<Vec2f>(1, 2)[0] * complex.at<Vec2f>(1, 2)[0] + complex.at<Vec2f>(1, 2)[1] * complex.at<Vec2f>(1, 2)[1]);
	
	return det;
}

cv::Mat invComplex(const cv::Mat& m)
{
	//Create matrix with twice the dimensions of original
	cv::Mat twiceM(m.rows * 2, m.cols * 2, CV_MAKE_TYPE(m.type(), 1));

	//Separate real & imaginary parts
	std::vector<cv::Mat> components;
	cv::split(m, components);

	cv::Mat real = components[0], imag = components[1];

	//Copy values in quadrants of large matrix
	real.copyTo(twiceM({ 0, 0, m.cols, m.rows })); //top-left
	real.copyTo(twiceM({ m.cols, m.rows, m.cols, m.rows })); //bottom-right
	imag.copyTo(twiceM({ m.cols, 0, m.cols, m.rows })); //top-right
	cv::Mat(-imag).copyTo(twiceM({ 0, m.rows, m.cols, m.rows })); //bottom-left

	//Invert the large matrix
	cv::Mat twiceInverse = twiceM.inv();

	cv::Mat inverse(m.cols, m.rows, m.type());

	//Copy back real & imaginary parts
	twiceInverse({ 0, 0, inverse.cols, inverse.rows }).copyTo(real);
	twiceInverse({ inverse.cols, 0, inverse.cols, inverse.rows }).copyTo(imag);

	//Merge real & imaginary parts into complex inverse matrix
	cv::merge(components, inverse);
	return inverse;
}
//! the constructor
void PTexton::initialize(string fname, int patchSize, int K, int knn){
	
	this->nfolds = 5;
	this->imgType = POLSAR;
	this->knn = knn;
	this->fname = fname;
	this->pSize = patchSize;
	this->half_patch = this->pSize / 2;
	this->K = K;
	this->RP = "no";

//	this->vSize = 0;
	ofile.open(to_string(K) + "K" + to_string(pSize) + "p" + "Experimental data.txt");

	if (this->loadImageData()){
		cout << "loadImageData successfully" << endl;
		ofile << "loadImageData successfully" << endl;
	}
	else{
		cout << "loadImageData error" << endl; ofile << "loadImageData error" << endl;
	}

	foldGeneration();
	loadReferenceData();
}

//! the destructor
PTexton::~PTexton(void){
	ofile.close();
}
void PTexton::generateRandomMat(Mat& randomMat, int highD, int lowD, string rMode){
	//Random Projection//
	//cout << "generate Random Matrix" << endl;

	//Random Matrix (k x d), k << d
	//k = lowD, d = highD
	randomMat.create(lowD, highD, CV_32F);
	
	if (rMode == "gaussian"){
		//gaussian normal distribution	
		RNG rng;
		rng.fill(randomMat, RNG::NORMAL, 0, 1);
		//cout << "random MAt"<<endl<<randomMat << endl;
	}
	else if (rMode == "achlioptas"){
		//Achlioptas's sparse random projection
		for (int i = 0; i < lowD; i++){
			for (int j = 0; j < highD; j++){
				int prob = rand() % 6;
				if (prob < 1){
					randomMat.at<float>(i, j) = 1;
				}
				else if (prob < 5){
					randomMat.at<float>(i, j) = 0;
				}
				else{
					randomMat.at<float>(i, j) = -1;
				}
			}
		}
		randomMat *= 1.732;	//multiply root 3
	}
	//cout << "generate Random Matrix - success" << endl;

}

vector<vector<Mat>> PTexton::generateFVectors(cv::Rect region){
	vector<vector<Mat>> fVectors;
	ofile << "generateFVectors()" << endl;

	//! current polsar data
	vector<Mat> curPolSAR = imageData.getPolSARData();

	for (int r = region.y; r < region.y + region.height; r++){
		for (int c = region.x; c < region.x + region.width; c++){
			vector<Mat> fVec;
			cv::Point2i pos(c,r);

			//trace each pixel in a patch
			for (int j = -half_patch; j <= half_patch; j++){
				for (int i = -half_patch; i <= half_patch; i++){
					cv::Point2i curPos(pos.x + i, pos.y + j);

					Mat aPixel;
					if (this->imgType == POLSAR){
						aPixel = Mat(3, 3, curPolSAR.at(0).type());	//two channel for complex value of PolSAR data (real, imaginary)

						//TODO: transformation of matrix can be done before patch extraction for reducing complexity
						int n = 0;
						//9 entries to 3x3 covariance matrix
						for (int k = 0; k < 3; k++){
							for (int l = 0; l < 3; l++){
								aPixel.at<Vec2f>(k, l)[0] = curPolSAR.at(n).at<Vec2f>(curPos)[0];
								aPixel.at<Vec2f>(k, l)[1] = curPolSAR.at(n).at<Vec2f>(curPos)[1];
								n++;
							}
						}
					}
					/*else if(this->imgType==GRAY){
						aPixel = Mat(1, 1, currentImg.type());
						aPixel.at<int>(0) = currentImg.at<int>(curPos);
					}*/
					fVec.push_back(aPixel.clone());
				}
			}
		fVectors.push_back(fVec);
		vector<Mat>().swap(fVec);
		}
	}
	vector<Mat>().swap(curPolSAR);
	curPolSAR.clear();

	//size of feature vectors
	cout << "current fVectors size:" << fVectors.size() << endl; 
	ofile << "current fVectors size:" << fVectors.size() << endl;

	return fVectors;
}

vector<vector<Mat>> PTexton::generateFVectors(cv::Rect region, int c){
	vector<vector<Mat>> fVectors;
	ofile << "generateFVectors()" << endl;

	//! current polsar data
	vector<Mat> curPolSAR = imageData.getPolSARData();
	/*if (imgType == GRAY){
		currentImg = imageData.getData(GRAY, 0);
	}*/

	//fold generation in reference data
	Mat refFold = referenceData.at(c);

	for (int r = region.y; r < region.y + region.height; r++){
		for (int c = region.x; c < region.x + region.width; c++){
			vector<Mat> fVec;

			int val = (int)refFold.at<uchar>(r, c);
			//if the pixel doesn't included the class, then discard it
			if (val == 0) {
				continue;
			}

			//trace each pixel in a patch
			for (int j = -half_patch; j <= half_patch; j++){
				for (int i = -half_patch; i <= half_patch; i++){
					cv::Point2i curPos(c + i, r + j);

					Mat aPixel;
					if (this->imgType == POLSAR){
						aPixel = Mat(3, 3, curPolSAR.at(0).type());	//two channel for complex value of PolSAR data (real, imaginary)

						//TODO: transformation of matrix can be done before patch extraction for reducing complexity
						int n = 0;
						//9 entries to 3x3 covariance matrix
						for (int k = 0; k < 3; k++){
							for (int l = 0; l < 3; l++){
								aPixel.at<Vec2f>(k, l)[0] = curPolSAR.at(n).at<Vec2f>(curPos)[0];
								aPixel.at<Vec2f>(k, l)[1] = curPolSAR.at(n).at<Vec2f>(curPos)[1];
								n++;
							}
						}
					}
					/*else if (this->imgType == GRAY){
						aPixel = Mat(1, 1, currentImg.type());
						aPixel.at<int>(0) = currentImg.at<int>(curPos);
					}*/
					fVec.push_back(aPixel.clone());
				}
			}
			fVectors.push_back(fVec);
			vector<Mat>().swap(fVec);
		}
	}
	vector<Mat>().swap(curPolSAR);
	curPolSAR.clear();

	//size of feature vectors
	cout << c<<"class fVectors size:" << fVectors.size() << endl;
	ofile << c << "class fVectors size:" << fVectors.size() << endl;

	return fVectors;
}


void PTexton::textonMapping(vector<vector<Mat>> tfvectors, vector<vector<Mat>> tcenters, int fold, int trainfold){
	cout << "textonMapping" << endl;
	ofile << "textonMapping" << endl;

	int vsSize = tfvectors.size();
	int vSize = tfvectors.at(0).size();
	int cSize = tcenters.size();

	cout << "center size=" << cSize << endl; ofile << "center size=" << cSize << endl;
	Mat label(vsSize, 1, CV_32S);


	//pre-calculation about center//
	vector<vector<Mat>> invCenters;
	vector<vector<float>> firstTerms;

	for (int k = 0; k < cSize; k++){	//cluster number
		vector<Mat> invCenter_k;
		vector<float> firstTerms_k;
		for (int m = 0; m < vSize; m++){	//order in a patch
			if (imgType == POLSAR){

				//first term calculation
				float det = determinant_comp(tcenters.at(k).at(m).clone());
				if (det < 0){
					cerr << "Error:det is less than 0" << endl;
				}
				firstTerms_k.push_back(log(det));

				//second term calculation
				invCenter_k.push_back(invComplex(tcenters.at(k).at(m).clone()).clone());
			}
		}
		invCenters.push_back(invCenter_k);
		firstTerms.push_back(firstTerms_k);
	}

	//assign labels to dataset
	for (int i = 0; i < vsSize; i++){
		float dist = 0, minDist = 0;

		//find min_distance with assigning label
		for (int k = 0; k < cSize; k++){	//cluster number
			dist = 0;
			for (int m = 0; m < vSize; m++){	//order in a patch
				if (imgType == POLSAR){
					dist += wishartDistance(firstTerms.at(k).at(m), invCenters.at(k).at(m).clone(), tfvectors.at(i).at(m).clone());
				}
				else if (imgType == GRAY){
					dist += abs(tfvectors.at(i).at(m).at<int>(0) - tcenters.at(k).at(m).at<int>(0));
				}
			}
			dist /= (float)vSize;
			//cout << "distance=" << dist << ",";
			if (k == 0){
				label.at<int>(i) = 0;
				minDist = dist;
			}
			else{
				if (minDist>dist){
					label.at<int>(i) = k;
					minDist = dist;
				}
			}
		}
	}

	Mat textonMap(foldRect[0].height,foldRect[0].width,CV_32S);
	
	int n = 0;
	for (int i = 0; i < textonMap.rows; i++){
		for (int j = 0; j < textonMap.cols; j++){
			textonMap.at<int>(i, j) = label.at<int>(n);
			n++;
		}
	}

	imwrite("testfold" + to_string(fold) + "trainfold" + to_string(trainfold) + "textonMap.png", textonMap);
	showImg(textonMap.clone(), "testfold" + to_string(fold) + "trainfold" + to_string(trainfold) + "textonMap_color.png", false, true);
	//return textonMap.clone();
}



void PTexton::textonMappingT(vector<vector<Mat>> tcenters, int fold, int trainfold){
	cout << "textonMapping" << endl;
	ofile << "textonMapping" << endl;

	int vSize = tcenters.at(0).size();
	int cSize = tcenters.size();

	cout << "center size=" << cSize << endl; ofile << "center size=" << cSize << endl;
	

	//pre-calculation about center//
	vector<vector<Mat>> invCenters;
	vector<vector<float>> firstTerms;

	for (int k = 0; k < cSize; k++){	//cluster number
		vector<Mat> invCenter_k;
		vector<float> firstTerms_k;
		for (int m = 0; m < vSize; m++){	//order in a patch
			if (imgType == POLSAR){

				//first term calculation
				float det = determinant_comp(tcenters.at(k).at(m).clone());
				if (det < 0){
					cerr << "Error:det is less than 0" << endl;
				}
				firstTerms_k.push_back(log(det));

				//second term calculation
				invCenter_k.push_back(invComplex(tcenters.at(k).at(m).clone()).clone());
			}
		}
		invCenters.push_back(invCenter_k);
		firstTerms.push_back(firstTerms_k);
	}


	//! current polsar data
	vector<Mat> curPolSAR = imageData.getPolSARData();
	/*if (imgType == GRAY){
		currentImg = imageData.getData(GRAY, 0);
	}*/

	Rect region = this->foldRect.at(trainfold);
	Mat textonMap(region.height, region.width, CV_32S);

	for (int r = region.y; r < region.y + region.height; r++){
		for (int c = region.x; c < region.x + region.width; c++){
			vector<Mat> fVec;
			cv::Point2i pos(c, r);			
			
			//trace each pixel in a patch
			for (int j = -half_patch; j <= half_patch; j++){
				for (int i = -half_patch; i <= half_patch; i++){
					cv::Point2i curPos(pos.x + i, pos.y + j);

					Mat aPixel = Mat(3, 3, curPolSAR.at(0).type());	//two channel for complex value of PolSAR data (real, imaginary)

					//TODO: transformation of matrix can be done before patch extraction for reducing complexity
					int n = 0;
					//9 entries to 3x3 covariance matrix
					for (int k = 0; k < 3; k++){
						for (int l = 0; l < 3; l++){
							aPixel.at<Vec2f>(k, l)[0] = curPolSAR.at(n).at<Vec2f>(curPos)[0];
							aPixel.at<Vec2f>(k, l)[1] = curPolSAR.at(n).at<Vec2f>(curPos)[1];
							n++;
						}
					}
					fVec.push_back(aPixel);
				}
			}

			float minDist = 0; 
			int label;
			//find min_distance with assigning label
			for (int k = 0; k < cSize; k++){	//cluster number
				float dist = 0;
				int m = 0;

				for (int m = 0; m < vSize; m++){
					dist += wishartDistance(firstTerms.at(k).at(m), invCenters.at(k).at(m).clone(), fVec.at(m).clone());
				}

				dist /= (float)vSize;
				//cout << "distance=" << dist << ",";
					
				if (k == 0){
					label = 0;
					minDist = dist;
				}
				else{
					if (minDist > dist){
						label = k;
						minDist = dist;
					}
				}
			}
			textonMap.at<int>(r-region.y, c-region.x) = label;
		}
	}
	vector<Mat>().swap(curPolSAR);
	curPolSAR.clear();

	imwrite("testfold" + to_string(fold) + "trainfold" + to_string(trainfold) + "textonMap.png", textonMap);
	showImg(textonMap.clone(), "testfold" + to_string(fold) + "trainfold" + to_string(trainfold) + "textonMap_color.png", false, true);
	//return textonMap.clone();
}


void PTexton::textonMappingG(vector<Mat> tcenters, int fold, int trainfold){
	cout << "textonMapping" << endl;
	ofile << "textonMapping" << endl;

	//int vSize = tcenters.at(0).size();
	int cSize = tcenters.size();

	cout << "center size=" << cSize << endl; ofile << "center size=" << cSize << endl;

	
	Mat currentImg = imageData.getData(GRAY, 0);
	Mat newImg;
	currentImg.convertTo(newImg, CV_32S);
	Rect region = this->foldRect.at(trainfold);
	Mat textonMap(region.height, region.width, CV_32S);

	for (int r = region.y; r < region.y + region.height; r++){
		for (int c = region.x; c < region.x + region.width; c++){
			Mat fVec(1,pSize*pSize,CV_32S);
			cv::Point2i pos(c, r);

			//trace each pixel in a patch
			int n = 0;
			for (int j = -half_patch; j <= half_patch; j++){
				for (int i = -half_patch; i <= half_patch; i++,n++){
					cv::Point2i curPos(pos.x + i, pos.y + j);

					fVec.at<int>(0, n) = newImg.at<int>(curPos);
				}
			}

			float minDist = 0;
			int label;
			//find min_distance with assigning label
			for (int k = 0; k < cSize; k++){	//cluster number
				float dist = 0;
				dist = norm(fVec,tcenters.at(k));
				
				if (k == 0){
					label = 0;
					minDist = dist;
				}
				else{
					if (minDist > dist){
						label = k;
						minDist = dist;
					}
				}
			}
			textonMap.at<int>(r - region.y, c - region.x) = label;
		}
	}

	imwrite("testfold" + to_string(fold) + "trainfold" + to_string(trainfold) + "textonMap.png", textonMap);
	showImg(textonMap.clone(), "testfold" + to_string(fold) + "trainfold" + to_string(trainfold) + "textonMap_color.png", false, true);
	//return textonMap.clone();
}
//! training patch-based textons
void PTexton::train(){

	cout << "train stage" << endl;
	ofile << "train stage" << endl;

	//learningTexton();
	
	for (int fold = 0; fold < this->nfolds; fold++){

		// training //// 	

		////construct texton Database ////
		vector<vector<Mat>> textonDB;
		for (int j = 0; j < nfolds; j++){
			if (fold != j){
				textonDB.insert(textonDB.end(), this->textons[j].begin(), this->textons[j].end());
			}
		}

		cout << "texton size=" << textonDB.size() << endl;
		ofile << "texton size=" << textonDB.size() << endl;

		//// texton mapping ////
		for (int trainfold = 0; trainfold < nfolds; trainfold++){		
			mapT[trainfold] = thread(&PTexton::textonMappingT, this, textonDB, fold, trainfold);
		}

		for (int trainfold = 0; trainfold < nfolds; trainfold++){
			mapT[trainfold].join();
		}

		//// learning histogram ////
		/*vector<Mat> thistDB;
		for (int j = 0; j < nfolds; j++){
			if (fold != j){
				Mat textonMap = imread("testfold" + to_string(fold) + "trainfold" + to_string(j) + "textonMap.png", 0);

				vector<Mat> tempHistDB = learnHist(textonMap.clone(), j);

				if (thistDB.size() == 0){
					thistDB = tempHistDB;
				}
				else{
					for (int k = 0; k < nclass; k++){
						Mat newHist;
						vconcat(thistDB.at(k), tempHistDB.at(k), newHist);
						thistDB.at(k) = newHist.clone();
					}
				}

			}
		}
		this->globalhistDB[fold] = thistDB;*/
	}

	/*for (int i = 0; i < this->nfolds; i++){
		vector<vector<Mat>>().swap(this->textons[i]);
		textons[i].clear();
	}*/

	cout << "train-success" << endl;
}

//! testing patch-based textons
void PTexton::histMatching(Mat textonMap, vector<Mat> histDBi, int fold){
	cout << "test start" << endl;
	ofile << "test start" << endl;

	Mat trainData;
	//Random Projection
	if (this->RP == "yes"){
		ofile << "random projection applied" << endl;
		cout << "random projection applied" << endl;
		Mat oneDB;
		vconcat(histDBi, oneDB);

		vector<Mat> reduced;	//reduced
		for (int i = 0; i < oneDB.rows; i++){
			Mat r = oneDB.row(i);
			transpose(r, r);
			Mat temprp = RandomProjection(r);
			reduced.push_back(temprp);
		}
		vconcat(reduced, trainData);
	}
	else{
		vconcat(histDBi, trainData);
	}
	//K-nn training
	trainData.convertTo(trainData, CV_32F);
	cout << "traindata size" << trainData.size() << endl;

	Mat trainClass(trainData.rows,1,CV_32SC1);

	int n = 0;
	for (int i = 0; i < histDBi.size(); i++){
		for (int j = 0; j < histDBi.at(i).rows; j++){
			trainClass.at<int>(n) = i;
			n++;
		}
	}
	cout << "Knn train success" << endl;
	ofile << "Knn train success" << endl;

	int minus = 0;
	cv::KNearest *knn = new KNearest(trainData, trainClass);

	Mat histograms((textonMap.rows - minus - (pSize - 1))*(textonMap.cols - (pSize - 1)),trainData.cols, CV_32F);
	cout << "hist create" << endl;

	int histSize = histDBi.at(0).cols;
	float range[] = { 0, histSize };
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;

	//trace test map
	n = 0; 
	for (int i = half_patch; i < textonMap.rows - minus - half_patch; i++){
		for (int j = half_patch; j < textonMap.cols - half_patch; j++,n++){

			//patch extraction in textonMap
			cv::Rect patchR(j-half_patch, i-half_patch, pSize, pSize);

			Mat patch = textonMap(patchR).clone();
			patch.convertTo(patch, CV_32F);

			//calculation of histogram in each patch
			//cout << "patch size=" << patch.size() << endl;
			//cout << patch << endl;

			Mat hist;
			calcHist(&patch, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
			//cout << "hist size=" << hist.size() << endl;
//			cout << hist << endl;

			if (this->RP == "yes"){
				hist = RandomProjection(hist);
				//hist.convertTo(hist, CV_32F);
			}
			else{
				transpose(hist, hist);
				hist.convertTo(hist, CV_32F);
			}

			//cout << hist << endl;
			hist.copyTo(histograms.row(n));
			//cout << cHist.row(i) << endl;
			
		}
		if (i%100==0)
			cout << "i=" << i << endl;
	}
	//TODO:revise Knn: a hist -> hists for find_nearest
	int knn_k=this->knn;
	//Mat nearests;// = new Mat(1, knn_k, CV_32FC1);
	Mat results;
	cout << "histogram size" << histograms.size()<< endl;

	//find nearest neighbor
	float response = knn->find_nearest(histograms, knn_k, &results);// , results, nearests);
	cout << "results size" << results.size() << endl;
	
	vector<Mat> classoutput;
	for (int i = 0; i < nclass; i++){
		classoutput.push_back(Mat::zeros(textonMap.rows - minus - (pSize - 1), textonMap.cols - (pSize - 1), CV_8U));
	}
	
	int m= 0;
	for (int i = 0; i < textonMap.rows - minus - (pSize - 1); i++){
		for (int j = 0; j < textonMap.cols - (pSize - 1); j++, m++){
			classoutput.at((int)results.at<float>(m)).at<uchar>(i, j) = 255;
		}
	}

	cout << "test end" << endl;
	ofile << "test end" << endl;
	

	for (int c = 0; c < nclass; c++){
		imwrite("test" + to_string(fold) + "fold_class" + to_string(c) + "output.png", classoutput.at(c));
	}
	//return classoutput;
}
//! testing patch-based textons (individual point) using chi-square distance
void PTexton::histMatchingI(Mat textonMap, vector<Mat> histDBi, int fold){
	cout << "test start" << endl;
	ofile << "test start" << endl;

	Mat trainData;
	//Random Projection
	if (this->RP == "yes"){
		Mat oneDB;
		vconcat(histDBi, oneDB);

		vector<Mat> reduced;	//reduced
		for (int i = 0; i < oneDB.rows; i++){
			Mat r = oneDB.row(i);
			transpose(r, r);
			reduced.push_back(RandomProjection(r));
		}
		vconcat(reduced, trainData);
	}
	else{
		vconcat(histDBi, trainData);
	}
	//K-nn training
	trainData.convertTo(trainData, CV_32F);
	Mat trainClass(trainData.rows, 1, CV_32SC1);

	int n = 0;
	for (int i = 0; i < histDBi.size(); i++){
		for (int j = 0; j < histDBi.at(i).rows; j++){
			trainClass.at<int>(n) = i;
			n++;
		}
	}

	int minus = 0;
	//cv::KNearest *knn = new KNearest(trainData, trainClass);

	cout << "Knn train success" << endl;
	ofile << "Knn train success" << endl;

	vector<Mat> classoutput;
	for (int i = 0; i < nclass; i++){
		classoutput.push_back(Mat::zeros(textonMap.rows - minus - (pSize - 1), textonMap.cols - (pSize - 1), CV_8U));
	}

	//trace test map
	n = 0;
	for (int i = half_patch; i < textonMap.rows - minus - half_patch; i++){
		for (int j = half_patch; j < textonMap.cols - half_patch; j++, n++){

			//patch extraction in textonMap
			cv::Rect patchR(j - half_patch, i - half_patch, pSize, pSize);

			Mat patch = textonMap(patchR).clone();
			patch.convertTo(patch, CV_32F);

			//calculation of histogram in each patch
			int histSize = histDBi.at(0).cols;
			float range[] = { 0, histSize };
			const float* histRange = { range };
			bool uniform = true; bool accumulate = false;
			//cout << "patch size=" << patch.size() << endl;
			//cout << patch << endl;

			Mat hist;
			calcHist(&patch, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
			//cout << "hist size=" << hist.size() << endl;
			//cout << hist << endl;

			if (this->RP == "yes"){
				//hist = RandomProjection(hist);
			}
			else{
				transpose(hist, hist);
				hist.convertTo(hist, CV_32F);
			}
			
			normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());
			//find nearest neighbor
			//int knn_k = this->knn;
			//float response = knn->find_nearest(hist, knn_k);// , results, nearests);
			double mindist = compareHist(hist, trainData.row(0), CV_COMP_CHISQR);
			int minIndex = 0;
			for (int t = 0; t < trainData.rows; t++){
				Mat row = trainData.row(t);
				double dist = compareHist(hist, row, CV_COMP_CHISQR);
				if (mindist >dist){
					mindist = dist;
					minIndex = t;
				}
			}
			int minClass = trainClass.at<int>(minIndex);
			classoutput.at(minClass).at<uchar>(i - half_patch, j - half_patch) = 255;

		}
		if (i % 100 == 0)
			cout << "i=" << i << endl;
	}	
	
	for (int c = 0; c < nclass; c++){
		imwrite("test" + to_string(fold) + "fold_class" + to_string(c) + "output.png", classoutput.at(c));
	}
	cout << "test end" << endl;
	ofile << "test end" << endl;

}
//! calculate histogram
vector<Mat> PTexton::learnHist(Mat textonMap, int fold){
	cout << "calculateHist start" << endl;
	ofile << "calculateHist start" << endl;

	//fold generation in reference data
	vector<Mat> refFolds;
	for (int c = 0; c < nclass; c++){
		Mat refFold = referenceData.at(c)(foldRect.at(fold));
		//imshow("refFold", refFold);
		//waitKey(0);
		refFolds.push_back(refFold.clone());
	}

	vector<Mat> histDB[5];

	for (int i = half_patch; i < foldRect[fold].height - half_patch; i++){
		for (int j = half_patch; j < foldRect[fold].width - half_patch; j++){
			//define the class in a pixel (center of patch) by searching reference data
			int curClass = -1;

			for (int c = 0; c < nclass; c++){
				int val = (int)refFolds.at(c).at<uchar>(i, j);
				//each class is exclusive
				if (val>0) {
					curClass = c;
					break;
				}
			}
			//if the pixel doesn't included any classes, then discard it
			if (curClass == -1) continue;

			//patch extraction in textonMap
			cv::Rect patchR(j-half_patch, i-half_patch, pSize, pSize);
			//cout << "patchR:" << patchR << endl;
			//cout << "current class:" << curClass << endl;

			Mat patch = textonMap(patchR);
			patch.convertTo(patch, CV_32F);
			//cout << patch << endl;
			//calculation of histogram in each patch
			int histSize = this->K*20;
			float range[] = { 0, histSize };
			const float* histRange = { range };
			bool uniform = true; bool accumulate = false;
			//cout << "patch size=" << patch.size() << endl;
			//cout << patch << endl;			

			Mat hist;
			calcHist(&patch, 1, 0, Mat(), hist, 1,&histSize, &histRange, uniform, accumulate);
			//cout << "hist size=" << hist.size() << endl;
			transpose(hist, hist);
			normalize(hist, hist,0,1,NORM_MINMAX,-1,Mat());

			//insert the histogram into histDB
			histDB[curClass].push_back(hist.clone());
			
		}
		//cout << i << endl;
	}

	vector<Mat> newDB;
	//transform type of histDB
	for (int c = 0; c < nclass; c++){
		int dbSize = histDB[c].size();
		cout << "DB size:" << dbSize << endl;
		ofile << "DB size:" << dbSize << endl;
		if (dbSize == 0){
			cout << "There are no training data in class " << c << endl;
			continue;
		}
		Mat cHist(dbSize,histDB[c].at(0).cols,CV_32F);
		for (int i = 0; i < dbSize; i++){
			Mat h = histDB[c].at(i).row(0);
			//cout << h << endl;
			h.copyTo(cHist.row(i));
			//cout << cHist.row(i) << endl;
		}
		newDB.push_back(cHist);
	}
	//return histDB in current fold		
	return newDB;
}

bool PTexton::loadReferenceData(){
	//if already done
	if (referenceData.size() > 0) return true;

	cout << "load reference Data";
	ofile << "load reference Data";
	//read files in a directory
	string dir = "reference";
	std::vector<std::string> vFileNames = GetFileNamesInDirectory(dir);

	for (int i = 0; i < vFileNames.size(); i++){

		//define file name for saving result images
		string fname = string(dir) + string("\\") + vFileNames[i];

		cout << "load image:" << fname << endl;
		ofile << "load image:" << fname << endl;

		Mat img = imread(fname, 0);

		//adjust size of reference image for matching position of each pixel in input image
		//cout << "ref size:" << img.size() << endl;
		//cout << "image size:" << this->imageData.getSize() << endl;

		resize(img, img, this->imageData.getSize());
		//normalize(img, img, 0, 255, CV_MINMAX);
		//cout << "resized ref size:" << img.size() << endl;
		this->referenceData.push_back(img);

		if (!img.data){
			cout << "ERROR: original image not specified" << endl;
			cout << "Press enter to exit" << endl;
			cin.get();
			return false;
		}
	}
	this->nclass = (int)referenceData.size();
	cout << "There are " << (int)referenceData.size() << " classes" << endl;
	ofile << "There are " << (int)referenceData.size() << " classes" << endl;
	
	return true;
}

void PTexton::selectCurrentImage()
{
	
	

	//TODO: solve the problem of insufficient memory
	switch (this->imgType){
	case GRAY:
		//cvtColor(colData, grayData, CV_BGR2GRAY);
		//this->currentImg = imageData.getData(GRAY, 0); break;
	case COLOR:
		// project it down (is the identiy function here, because multi-pol is the highest data-level, 
		//but the projections to lower data-levels will be performed here, too, so they are available later)
		imageData.project();
		//this->currentImg = imageData.getData(COLOR, 0);// (*img)->getData(COLOR, rep);	
		break;
	case POLSAR:
		//this->curPolSAR = imageData.getPolSARData();
		//this->currentImg = curPolSAR.at(0);
		return;
	}
			
}
void PTexton::foldGeneration(){
	//if already done
	if (this->foldRect.size() > 0) return;

	//// 5-fold cross-validation ////
	std::cout << "There are " << this->nfolds << " folds" << endl;
	ofile << "There are " << this->nfolds << " folds" << endl;
	std::cout << "There are less images than folds" << endl;
	
	cv::Size curSize = imageData.getSize();

	int height = curSize.height;
	int width = curSize.width;
	
	//use vertical folds
	for (int i = 0; i < this->nfolds; i++){
		int boundary = 10;
		int w = (width - 2*boundary) / nfolds;
		int h = height - 2*boundary;
		int x = boundary + w*i;
		int y = boundary;


		cv::Rect tempRect(x, y, w, h);
		this->foldRect.push_back(tempRect);

		ofile << i << "th fold=" << tempRect << endl;
	}

	// TODO:if training foldType = 0 else if testing foldType = 1
	int foldType = 0;

}

void PTexton::test(){

	//// learning histogram ////
	if (globalhistDB[0].size() == 0){
		for (int fold = 0; fold < nfolds; fold++){
			vector<Mat> thistDB;
			for (int j = 0; j < nfolds; j++){
				if (fold != j){
					Mat textonMap = imread("testfold" + to_string(fold) + "trainfold" + to_string(j) + "textonMap.png", 0);

					vector<Mat> tempHistDB = learnHist(textonMap.clone(), j);

					if (thistDB.size() == 0){
						thistDB = tempHistDB;
					}
					else{
						for (int k = 0; k < nclass; k++){
							Mat newHist;
							vconcat(thistDB.at(k), tempHistDB.at(k), newHist);
							thistDB.at(k) = newHist.clone();
						}
					}

				}
			}
			this->globalhistDB[fold] = thistDB;
		}
	}


	ofile<<"euclide dist is used"<<endl;
	for (int fold = 0; fold < nfolds; fold++){	
		Mat test_map = imread("testfold" + to_string(fold) + "trainfold" + to_string(fold) + "textonMap.png", 0);
		histMatching(test_map, globalhistDB[fold], fold);
	}
	/*ofile << "chi_dist is used" << endl;
	for (int fold = 0; fold < nfolds; fold++){
		Mat test_map = imread("testfold" + to_string(fold) + "trainfold" + to_string(fold) + "textonMap.png", 0);
		matchT[fold] = thread(&PTexton::histMatchingI, this, test_map, globalhistDB[fold], fold);
	}
	for (int fold = 0; fold < nfolds; fold++){
		matchT[fold].join();
	}*/
}
void PTexton::errorAssessment(){
	printResult();
	vector<vector<Mat>> estimate;

	for (int fold = 0; fold < nfolds; fold++){
		vector<Mat> outputc;
		for (int c = 0; c < nclass; c++){
			Mat img = imread("test" + to_string(fold) + "fold_class" + to_string(c) + "output.png", 0);
			
			img.convertTo(img, CV_32F);img /= 255.0;
			outputc.push_back(img);
		}
		estimate.push_back(outputc);
	}
	vector<vector<Mat>> reference;
	for (int fold = 0; fold < nfolds; fold++){
		vector<Mat> ref;
		for (int c = 0; c < nclass; c++){
			Mat img = referenceData.at(c)(Rect(foldRect.at(fold).x + half_patch, foldRect.at(fold).y + half_patch, foldRect.at(fold).width - (pSize - 1), foldRect.at(fold).height-(pSize-1)));
			
			img.convertTo(img, CV_32F);img /= 255.0;
			ref.push_back(img);
		}
		reference.push_back(ref);
	}
	ClassificationError error(reference, estimate);
	error.print();

}
//! evaluate textons
void PTexton::evaluate(){

	std::cout << "Evaluation start" << endl;

	learningTexton();

	train();

	test();

	errorAssessment();

	//concatenate output
}

Mat PTexton::RandomProjection(Mat target)
{
	int highD = target.rows;
	int lowD = highD / 3;
	Mat rMat;

	generateRandomMat(rMat, highD, lowD, "gaussian");//Rmode = "gaussian" or "achlioptas"
	//cout << rMat << endl;
	
	Mat reducedMat = rMat*target;
	transpose(reducedMat, reducedMat);

	return reducedMat.clone();
}
void PTexton::RandomProjection(int foldN)
{
	vector<vector<Mat>>fVectors;
	int highD;// = fVectors.at(0).size();
	int lowD = highD / 3;
	Mat rMat;

	generateRandomMat(rMat, highD, lowD, "gaussian");//Rmode = "gaussian" or "achlioptas"
	//cout << rMat << endl;

	//cout << "original vector size = " << fVectors[foldN].at(0).size() << endl;
	for (int j = 0; j < fVectors[foldN].size(); j++){
		vector<Mat> highVector = fVectors[foldN].at(j);
		vector<Mat> lowVector;

		//matrix multiplication
		for (int k = 0; k < lowD; k++){
			Mat sum = Mat::zeros(highVector.at(0).size(), highVector.at(0).type());

			for (int l = 0; l < highD; l++){
				sum += rMat.at<float>(k, l)*highVector.at(l);
			}
			lowVector.push_back(sum);
		}
		fVectors.at(j) = lowVector;
	}
	//cout << "after RP, vector size = " << fVectors[foldN].at(0).size() << endl;
}
//! load image data
bool PTexton::loadImageData(void){
	ImgData img;

	cout << "file name:" << this->fname << endl;
	ofile << "file name:" << this->fname << endl;

	if (this->fname.size() > 0){
		img = ImgData(this->fname, POLSAR);
		img.load();
		this->imageData = img;
		return true;
	}
	else{
		cout << "ERROR: File path doesn't exist" << endl;
		return false;
	}

}

//! visualize center matrix	(visualize K textons)
void PTexton::printCenter(Mat& centers){

	if (this->K != centers.rows){
		cerr << "centers' row is not equal to K" << endl;
		return;
	}
	int nChannel = 0;
	int type = CV_16U;

	switch (this->imgType){
	case GRAY: break;
	case COLOR:	type = CV_64FC3; break;
	case POLSAR: break;	//tranform PolSAR to color image
	}

	float* p;
	for (int i = 0; i < this->K; i++){
		p = centers.ptr<float>(i);

		Mat visTexton(this->pSize, this->pSize, type);

		for (int k = 0, n2 = 0; k < this->pSize; k++){
			for (int l = 0; l < this->pSize; l++,n2++){
				switch (this->imgType){
				case GRAY:
					visTexton.at<float>(k, l) = p[n2]; break;
				case COLOR:
					visTexton.at<Vec3f>(k, l)[0] = p[n2]; n2++;
					visTexton.at<Vec3f>(k, l)[1] = p[n2]; n2++;
					visTexton.at<Vec3f>(k, l)[2] = p[n2]; break;
				}
			}
		}
		imshow("texton" + to_string(i), visTexton);
		waitKey(0);
	}
	cout << endl;


}
//! clustering textons by random sampling
void PTexton::initializeCenters(int sampling){
	ofile << "initialize centers" << endl;
	vector<vector<Mat>> fVectors;
	int fsize = fVectors.size();
	int vSize = fVectors.at(0).size();

	vector<vector<Mat>> samples;
	if (sampling == 0){
		samples = fVectors;
	}
	else{
		//RAMDOM SAMPLING FOR DIMENSIONALITY REDUCTION
		srand((unsigned)time(NULL));
		
		for (int i = 0; i < sampling; i++){
			int r = rand() % fsize;
			samples.push_back(fVectors.at(r));
		}
		fsize = samples.size();
	}

	ofile << "sampled feature vectors size= " << fsize << endl;
	// initialize centers
	vector<Mat> init;
	int random = rand() % fsize;
	ofile << "random number = " << random << endl;
	init = samples.at(random);

	vector<Mat> invCenter;
	vector<float> firstTerms;
	for (int m = 0; m < vSize; m++){	//order in a patch
		if (imgType == POLSAR){
			//first term calculation
			float det = determinant_comp(init.at(m).clone());
			if (det < 0){
				cerr << "Error:det is less than 0" << endl;
				return;
			}
			firstTerms.push_back(log(det));

			//second term calculation
			invCenter.push_back(invComplex(init.at(m).clone()).clone());
		}
	}

	Mat distMap(fsize, 1, CV_32F);
	//assign labels to dataset
	for (int i = 0; i < fsize; i++){
		float dist = 0;

		//find min_distance with assigning label
		for (int m = 0; m < vSize; m++){	//order in a patch
			float wDist = wishartDistance(firstTerms.at(m), invCenter.at(m).clone(), samples.at(i).at(m).clone());
			dist += wDist;			
		}
		dist /= (float)vSize;

		distMap.at<float>(i) = dist;			
	}
	//cout << distMap << endl;
	Mat sortedMap;
	sortIdx(distMap, sortedMap, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
	//cout << sortedMap << endl;

	vector<vector<Mat>> newcenters;
	for (int i = 0; i < K; i++){
		vector<Mat> newcenter;
		for (int j = 0; j < vSize; j++){
			newcenter.push_back(Mat::zeros(init.at(j).size(),init.at(j).type()));
		}
		newcenters.push_back(newcenter);
	}
	
	//integration of feature vector
	for (int k = 0; k < this->K; k++){
		int index = sortedMap.at<int>(k*(fsize / K));
		cout << index << endl;
		for (int j = 0; j < vSize; j++){ //in a patch vector
			newcenters.at(k).at(j) = samples.at(index).at(j).clone();
		}
	}
	//this->textons = newcenters;
	cout << "initialize centers success" << endl;

	vector<vector<Mat>>().swap(samples);
	samples.clear();

}
//! clustering textons by random sampling
void PTexton::clusterTextons(vector<vector<Mat>> fVectors, int fold){

	cout << "kmeans clustering / K=" << this->K << endl;
	ofile << "kmeans clustering / K=" << this->K << endl;
	

	int fsize = fVectors.size();
	int vSize = fVectors.at(0).size();
	Mat label(fsize, 1, CV_32S);
	cout << "original feature vectors size= " << fsize << endl;
	ofile << "original feature vectors size= " << fsize << endl;
	
	vector<vector<Mat>> samples = fVectors;

	//// random sampling ////
	//vector<vector<Mat>> samples;
	//for (int i = 0; i < fsize/100; i++){
	//	int r = rand() % fsize;
	//	samples.push_back(fVectors.at(r));
	//}
	//fsize = samples.size();

	////// initialize centers ////
	//int sampling = K * 3000;
	//this->clusterTextons(fold, c, sampling);
	
	vector<vector<Mat>> centers;// = textons[fold];
	int centerSize = this->K;

	for (int i = 0; i < centerSize; i++){
		int random = rand() % fsize;
		centers.push_back(samples.at(random));
	}

	const int maxIter = 20;
	int iter = 0;
	// loop if stop criteria isn't satisfied
	while (iter < maxIter){
		Mat nlabel = Mat::zeros(centerSize, 1, CV_32S);

		//pre-calculation about center//
		vector<vector<Mat>> invCenters;
		vector<vector<float>> firstTerms;

		for (int k = 0; k < centerSize; k++){	//cluster number
			vector<Mat> invCenter_k;
			vector<float> firstTerms_k;
			for (int m = 0; m < vSize; m++){	//order in a patch
				if (imgType == POLSAR){
					//first term calculation
					float det = determinant_comp(centers.at(k).at(m).clone());
					if (det < 0){
						cerr << "Error:det is less than 0" << endl;
						return;
					}
					firstTerms_k.push_back(log(det));

					//second term calculation
					invCenter_k.push_back(invComplex(centers.at(k).at(m).clone()).clone());
				}
			}
			invCenters.push_back(invCenter_k);
			firstTerms.push_back(firstTerms_k);
		}
		//cout << "center pre-calculation" << endl;

		//assign labels to dataset
		for (int i = 0; i < fsize; i++){
			float dist = 0, minDist = 0;

			//find min_distance with assigning label
			for (int k = 0; k < centerSize; k++){	//cluster number
				dist = 0;
				for (int m = 0; m < vSize; m++){	//order in a patch
					float wDist = wishartDistance(firstTerms.at(k).at(m), invCenters.at(k).at(m).clone(), samples.at(i).at(m).clone());
					dist += wDist;
				}

				dist /= (float)vSize;
				//cout << "distance=" << dist << ",";
				if (k == 0){
					label.at<int>(i) = 0;
					minDist = dist;
				}
				else{
					if (minDist>dist){
						label.at<int>(i) = k;
						minDist = dist;
					}
				}
			}
			nlabel.at<int>(label.at<int>(i)) += 1;
			//if (i % 1000 == 0)
			//cout << i << "th vector label=" << label.at<int>(i) << endl;

		}
		//cout << "assign label -complete" << endl;

		//ofile << "nlabel=" << nlabel << endl;
		//cout << "nlabel=" << nlabel << endl;

		//improve intialization//
		bool one = false;
		for (int i = 0; i <centerSize; i++){
			if (nlabel.at <int>(i) < 3){
				one = true; break;
			}
		}
		if (one){
			for (int i = 0; i < centerSize; i++){
				int random = rand() % (fsize );
				centers.at(i) = samples.at(random);
			}

			continue;
		}
		
		////calculate new centers////
		//initialize centers
		vector<vector<Mat>> newcenters;
		for (int i = 0; i < centerSize; i++){
			vector<Mat> newcenter;
			for (int j = 0; j < vSize; j++){
				newcenter.push_back(Mat::zeros(centers.at(i).at(j).size(), centers.at(i).at(j).type()));
			}
			newcenters.push_back(newcenter);
		}
		//cout << "initialize centers vector-complete" << endl;

		//integration of feature vector
		for (int i = 0; i < fsize; i++){
			for (int j = 0; j < vSize; j++){ //in a patch vector
				newcenters.at(label.at<int>(i)).at(j) += samples.at(i).at(j).clone();
			}
		}
		//cout << "add feature matrix-complete" << endl;

		//mean of feature vector (centers)
		for (int i = 0; i < centerSize; i++){
			for (int j = 0; j < vSize; j++){	//in a patch vector
				if (nlabel.at<int>(i)>1){
					newcenters.at(i).at(j) /= (float)nlabel.at<int>(i);
				}
			}
			//cout << "K=" << i << "center matrix" << endl<<newcenters.at(i).at(0) << endl;
		}
		//cout << "make centers - complete" << endl;

		centers = newcenters;
		newcenters.clear();

		iter++;
		////stop criteria////	
		//oldCenters != centers

	}
	for (int i = 0; i < centerSize; i++){
		this->textons[fold].push_back(centers.at(i));
	}
	
	vector<vector<Mat>>().swap(samples);
	samples.clear();

	cout << "cluster Textons success" << endl;

}
void PTexton::printTextonMap(){
	for (int fold = 0; fold < this->nfolds; fold++){
		vector<Mat> temp;
		for (int trainfold = 0; trainfold < nfolds; trainfold++){
			//cout << "??" << endl;
			if (fold != trainfold){
				Mat textonMap = imread("testfold" + to_string(fold) + "trainfold" + to_string(trainfold) + "textonMap.png", 0);
				temp.push_back(textonMap);
			}
		}
		Mat output;
		hconcat(temp, output);
		imwrite("testfold" + to_string(fold) + "traintextonMap.png", output);
		showImg(output.clone(), to_string(this->K)+"k_"+to_string(pSize)+"p_testfold" + to_string(fold) + "traintextonMap_color.png", false, true);
	}	
}


void PTexton::printResult(){
	//print ref
	Mat a = Mat::zeros(referenceData.at(0).size(), referenceData.at(0).type());
	Mat a2 = Mat::zeros(referenceData.at(0).size(), referenceData.at(0).type());
	for (int i = 0; i < 5; i++){
		Mat ref = referenceData.at(i);
		a += ref;
		normalize(ref, ref, 0, i + 1, NORM_MINMAX);
		a2 += ref;
	}
	a2.convertTo(a2, CV_8U);
	normalize(a2, a2, 0, 255, CV_MINMAX);
	applyColorMap(a2, a2, COLORMAP_RAINBOW);
	imwrite("referenceClasses.png", a2);

	//print each class
	vector<Mat> outputc[5];
	vector<Mat> outputc2[5];

	for (int fold = 0; fold < nfolds; fold++){
		for (int c = 0; c < nclass; c++){
			Mat img = imread("test" + to_string(fold) + "fold_class" + to_string(c) + "output.png", 0);

			//img.convertTo(img, CV_32F);// img /= 255.0;

			//normalize(img, img, 0, c + 1, NORM_MINMAX);
			outputc[c].push_back(img);

			Mat rimg = a(Rect(foldRect.at(fold).x + half_patch, foldRect.at(fold).y + half_patch, foldRect.at(fold).width - (pSize - 1), foldRect.at(fold).height - (pSize - 1)));
			//cout << img.size() << endl;
			//cout << rimg.size() << endl;
			Mat andMat;
			bitwise_and(rimg, img, andMat);
			normalize(andMat, andMat, 0, c + 1, NORM_MINMAX);
			outputc2[c].push_back(andMat);
		}
	}
	for (int c = 0; c < nclass; c++){
		Mat outclass;
		hconcat(outputc[c], outclass);
		imwrite(to_string(c) + "output.png", outclass);
	}

	//print one output depend on reference images
	Mat outclass0;
	hconcat(outputc2[0], outclass0);
	Mat ref = Mat::zeros(outclass0.size(),outclass0.type());
	ref += outclass0;
	for (int c = 1; c < nclass; c++){
		Mat outclass;
		hconcat(outputc2[c], outclass);
		ref += outclass;
	}
	ref.convertTo(ref, CV_8U);
	normalize(ref, ref, 0, 255, CV_MINMAX);
	applyColorMap(ref, ref, COLORMAP_RAINBOW);
	imwrite(to_string(this->K)+"_"+to_string(this->pSize)+"refalloutput.png", ref);

	////print one output
	//Mat oriOutput = Mat::zeros(outclass0.size(), outclass0.type());
	//for (int c = 0; c < nclass; c++){
	//	Mat outclass;
	//	hconcat(outputc3[c], outclass);
	//	oriOutput += outclass;
	//}
	//oriOutput.at<int>(0, 0) = 0;
	//oriOutput.convertTo(oriOutput, CV_8U);
	//normalize(oriOutput, oriOutput, 0, 255, CV_MINMAX);
	//applyColorMap(oriOutput, oriOutput, COLORMAP_RAINBOW);
	//imwrite("originalalloutput.png", oriOutput);
}
float PTexton::wishartDistance(float firstTerm, Mat invCenter, Mat comp){	//input ::region for training and testing

	//std::cout << "--wishart distance measure start--" << endl;
	//// calculate Wishart-distance ////
	Mat fcomp[2];
	Mat ccomp[2];

	split(comp, fcomp);
	split(invCenter, ccomp);

	float secondTerm = 0;
	secondTerm = trace(ccomp[0] * fcomp[0] - ccomp[1] * fcomp[1])[0];

	float wishart = firstTerm + secondTerm;

	return wishart;
}
float PTexton::wishartDistance(Mat center, Mat comp){	//input ::region for training and testing

	//std::cout << "--wishart distance measure start--" << endl;
	//// calculate Wishart-distance ////
	
	//first term calculation
	float firstTerm;
	float det = determinant_comp(center);
	
	
	if (det < 0){
		cerr << "Error:det is less than 0" << endl;
		cout << "determinant : " << det << endl;
		Mat sep[2];
		split(comp, sep);
		cout << sep[0] << endl;
		cout << sep[1] << endl;
		return -1;
	}

	firstTerm = log(det);
	
	//second term calculation
	Mat invCenter = invComplex(center);
	
	Mat fcomp[2];
	Mat ccomp[2];

	split(comp, fcomp);
	split(invCenter, ccomp);

	float secondTerm=0;
	secondTerm = trace(ccomp[0] * fcomp[0] - ccomp[1] * fcomp[1])[0];
	
	float wishart = firstTerm + secondTerm;
		
	return wishart;
}
// Function displays image (after proper normalization)
/*
img   :  Image that shall be displayed
cut   :  whether to cut or scale values outside of [0,255] range
*/
void PTexton::showImg(Mat img, string win, bool show, bool save){

	Mat aux = img.clone();

	// scale and convert
	if (img.channels() == 1){
		img.convertTo(aux, CV_8U);
		//normalize(aux, aux, 0, 255, CV_MINMAX);
		applyColorMap(aux, aux, COLORMAP_HSV);		
	}

	// show
	if (show){
		imshow(win, aux);
		waitKey(0);
	}

	// save
	if (save)
		imwrite(win+ string(".png"), aux);
}


std::vector<std::string> GetFileNamesInDirectory(std::string directory) 
{
	cout << "GetFileNamesInDirectory" << endl;
	
	char* buffer;

	// Get the current working directory: 
	if ((buffer = _getcwd(NULL, 0)) == NULL)
		perror("_getcwd error");
	else
	{
		//printf("%s \nLength: %d\n", buffer, strnlen(buffer,100));
		//free(buffer);
	}
	directory = string(buffer) + string("\\")+directory+string("\\*");
//	cout << directory << endl;

	std::vector<std::string> files;
	WIN32_FIND_DATA fileData;
	HANDLE hFind;
	
	if (!((hFind = FindFirstFile(directory.c_str(), &fileData)) == INVALID_HANDLE_VALUE)) {
		//cout << fileData.cFileName << endl;
		while (FindNextFile(hFind, &fileData)) {
			string fName = fileData.cFileName;
			size_t loc = fName.find(".");
			string ext = fName.substr(loc);
			//cout << ext << endl;
			if (ext == ".png"){
				//cout << fileData.cFileName << endl;
				files.push_back(fileData.cFileName);
			}
		}
	}
	else{
		cout << "Error:GetFileNamesInDirectory()" << endl;
	}

	FindClose(hFind);
	return files;
}


void PTexton::learningTexton(void){

	cout << "learningTexton start" << endl;

	for (int fold = 0; fold < this->nfolds; fold++){
		//// feature extraction ////

		for (int c = 0; c < nclass; c++){
			vector<vector<Mat>> fVectors;

			//feature extraction		
			fVectors = this->generateFVectors(this->foldRect.at(fold), c);

			//cluster textons
			textonT[c] = thread(&PTexton::clusterTextons, this, fVectors, fold);
			//this->clusterTextons(fVectors,fold);

			//fVectors.clear();

			//vector free
			//vector<vector<Mat>>().swap(fVectors);
		}
		for (int c = 0; c < nclass; c++){
			textonT[c].join();
		}
	}

	cout << "learningTexton end" << endl;
}


void PTexton::grayscaleTexton(void){
	this->imageData.project();

	this->imgType = GRAY;
	Mat currentImg = imageData.getData(GRAY, 0);
	//imwrite("grayscale.png", currentImg);
	//imshow("gray", currentImg/255.0);
	//waitKey(0);

	cout << "learningTexton start" << endl;
	vector<Mat> gtextons[5];

	for (int fold = 0; fold < this->nfolds; fold++){
				
		Rect region = this->foldRect.at(fold);
		/*cout << region << endl;
		imshow("fold", currentImg(region) / 255.0);
		waitKey(0);*/

		for (int nc = 0; nc< this->nclass; nc++){
			cout << "fold" << fold << "class" << nc << endl;

			//// feature extraction ////	
			vector<Mat> fVectors;// = Mat(region.height*region.width, pSize*pSize, CV_32S);

			//fold generation in reference data
			Mat refFold = referenceData.at(nc);

			for (int r = region.y; r < region.y + region.height; r++){
				for (int c = region.x; c < region.x + region.width; c++){
					//trace each pixel in a patch
					Mat fVec = Mat(1, pSize*pSize, CV_8U);
					int val = (int)refFold.at<uchar>(r, c);
					//if the pixel doesn't included the class, then discard it
					if (val == 0) {
						continue;
					}
					int n = 0;
					for (int j = -half_patch; j <= half_patch; j++){
						for (int i = -half_patch; i <= half_patch; i++, n++){
							cv::Point2i curPos(c + i, r + j);
							fVec.at<uchar>(0, n) = currentImg.at<uchar>(curPos);
						}
					}
					//cout << fVec << endl;
					fVectors.push_back(fVec);
				}
			}
			Mat fvecMat,bestlabel,centers;
			vconcat(fVectors, fvecMat);
			//cout << fvecMat << endl;
			fvecMat.convertTo(fvecMat, CV_32F);
			
			kmeans(fvecMat, K, bestlabel, TermCriteria(CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_RANDOM_CENTERS,centers);
			//////cluster textons////
			/*
			//int fsize = fVectors.size();
			//Mat label(fsize, 1, CV_32S);
			//cout << "original feature vectors size= " << fsize << endl;
			//ofile << "original feature vectors size= " << fsize << endl;

			//vector<Mat> centers;// = textons[fold];
			//int centerSize = this->K;

			//for (int i = 0; i < centerSize; i++){
			//	int random = rand() % fsize;
			//	centers.push_back(fVectors.at(random));
			//}

			//const int maxIter = 20;
			//int iter = 0;
			//// loop if stop criteria isn't satisfied
			//while (iter < maxIter){
			//	Mat nlabel = Mat::zeros(centerSize, 1, CV_32S);

			//	//assign labels to dataset
			//	for (int i = 0; i < fsize; i++){
			//		float minDist = 0;

			//		//find min_distance with assigning label
			//		for (int k = 0; k < centerSize; k++){	//cluster number
			//			float dist = norm(centers.at(k), fVectors.at(i));

			//			//cout << "distance=" << dist << ",";
			//			if (k == 0){
			//				label.at<int>(i) = 0;
			//				minDist = dist;
			//			}
			//			else{
			//				if (minDist>dist){
			//					label.at<int>(i) = k;
			//					minDist = dist;
			//				}
			//			}
			//		}
			//		nlabel.at<int>(label.at<int>(i)) += 1;
			//	}
			//	//cout << nlabel << endl;
			//	
			//	//improve intialization//
			//	bool one = false;
			//	for (int i = 0; i < centerSize; i++){
			//		if (nlabel.at <int>(i) < 3){
			//			one = true; break;
			//		}
			//	}
			//	if (one){
			//		for (int i = 0; i < centerSize; i++){
			//			int random = rand() % (fsize);
			//			centers.at(i) = fVectors.at(random);
			//		}

			//		continue;
			//	}
			//	
			//	////calculate new centers////
			//	//initialize centers
			//	vector<Mat> newcenters;
			//	for (int p = 0; p < this->K; p++){
			//		newcenters.push_back(Mat::zeros(1, pSize*pSize, CV_32S));
			//	}
			//	//integration of feature vector
			//	for (int i = 0; i < fsize; i++){
			//		newcenters.at(label.at<int>(i)) += fVectors.at(i).clone();

			//	}
			//	//cout << "add feature matrix-complete" << endl;

			//	//mean of feature vector (centers)
			//	for (int i = 0; i < centerSize; i++){
			//		if (nlabel.at<int>(i)>1){
			//			newcenters.at(i) /= (float)nlabel.at<int>(i);
			//		}
			//		//cout << newcenters.at(i) << endl;
			//	}
			//	//cout << "make centers - complete" << endl;

			//	centers = newcenters;
			//	newcenters.clear();

			//	iter++;

			//}
			*/
			//cout << centers << endl;
			centers.convertTo(centers, CV_32S);
			for (int i = 0; i < K; i++){
				//cout << centers.row(i) << endl;
				gtextons[fold].push_back(centers.row(i));
			}
		}
	}

	for (int fold = 0; fold < this->nfolds; fold++){
		// training //// 	
		////construct texton Database ////
		vector<Mat> textonDB;
		for (int j = 0; j < nfolds; j++){
			if (fold != j){
				textonDB.insert(textonDB.end(), gtextons[j].begin(), gtextons[j].end());
			}
		}

		cout << "texton size=" << textonDB.size() << endl;
		ofile << "texton size=" << textonDB.size() << endl;
		int centerSize = textonDB.size();

		//// texton mapping ////
		for (int trainfold = 0; trainfold < nfolds; trainfold++){
			//textonMappingG(textonDB, fold, trainfold);
			mapT[trainfold] = thread(&PTexton::textonMappingG, this, textonDB, fold, trainfold);
		}
		for (int trainfold = 0; trainfold < nfolds; trainfold++){
			mapT[trainfold].join();
		}
	}

	printTextonMap();
	cout << "learningTexton end" << endl;
}