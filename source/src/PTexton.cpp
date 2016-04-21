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
PTexton::PTexton(string fname, int patchSize, int K, string rp){
	
	this->fname = fname;
	this->pSize = patchSize;
	this->half_patch = this->pSize / 2;
	this->K = K;
	this->RP = rp;

	this->nfolds = 5;
	this->imgType = POLSAR;
//	this->vSize = 0;

	if (this->loadImageData()){
		cout << "loadImageData successfully" << endl;
	}
	else{
		cout << "loadImageData error" << endl;
	}

}

//! the destructor
PTexton::~PTexton(void){

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

void PTexton::generateFVectors(cv::Rect region, int num){

	cout << "generateFVectors()" << endl;

	int nVector = region.height * region.width;
	//cout << "expected fVectors size:" << nVector << endl;
	
	
	//! current polsar data
	vector<Mat> curPolSAR = imageData.getPolSARData();
	if (imgType == GRAY){
		currentImg = imageData.getData(GRAY, 0);
	}

	for (int r = region.y; r < region.y + region.height; r++){
		for (int c = region.x; c < region.x + region.width; c++){
			vector<Mat> fVec;
			cv::Point2i pos(c,r);

			//trace each pixel in a patch
			for (int i = -half_patch; i <= half_patch; i++){
				for (int j = -half_patch; j <= half_patch; j++){
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
						//float det = determinant_comp(aPixel);
						//cout << "determinant=" <<det<< endl;
						//float dst = wishartDistance(aPixel, aPixel.clone());
						//cout << "same matrix distance=" << dst << endl;
					}
					else if(this->imgType==GRAY){
						aPixel = Mat(1, 1, currentImg.type());
						aPixel.at<int>(0) = currentImg.at<int>(curPos);
					}
					fVec.push_back(aPixel.clone());
				}
			}
		this->fVectors[num].push_back(fVec);
		}
	}

	//size of feature vectors
	//cout << "current fVectors size:" << fVectors[num].size() << endl;

}

void PTexton::learningTexton(void){

	cout << "learningTexton start" << endl;

	this->selectCurrentImage();

	//variable for K-means
	//Mat featureVec = extractFVec("train",this->RP); // (train, RP)

	//clusterTextons(featureVec);
	
	cout << "learningTexton end" << endl;
}

Mat PTexton::textonMapping(vector<vector<Mat>> tfvectors, vector<vector<Mat>> tcenters){
	cout << "textonMapping" << endl;
	
	int vsSize = tfvectors.size();
	int vSize = tfvectors.at(0).size();
	int cSize = tcenters.size();

	cout << "center size=" << cSize << endl;
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
					//if (wishartDistance(centers.at(k).at(m).clone(), fVectors[num].at(i).at(m).clone()) == -1){
					//if (wDist == -1){
					//	cout << "wishart Distance calculation error:at=" << i << ",at(in vector)=" << m << endl;
					//}
					//efficient version
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
	return textonMap.clone();
}
//! training patch-based textons
void PTexton::train(){

	cout << "train" << endl;
	//// Image selection ////
	//selectCurrentImage();
	foldGeneration();
	loadReferenceData();

	//// feature extraction ////
	for (int j = 0; j < 5; j++){// this->nfolds; j++){
		//feature extraction				
		this->generateFVectors(this->foldRect.at(j), j);

		//Random Projection
		//this->RandomProjection(j);

		//cluster textons
		this->clusterTextons(j);
	}

	//// training and testing in each fold ////
	for (int i = 0; i < this->nfolds; i++){
		// training //// 	

		//texton dictionary construction
		vector<vector<Mat>> textonDic;
		for (int j = 0; j < this->nfolds; j++){
			if (j != i) //test data isn't included in training data
				textonDic.insert(textonDic.end(), textons[j].begin(), textons[j].end());
		}

		//printCenter(centers);
		cout << "K=" << textonDic.size() << endl;

		//texton mapping
		Mat textonMap[5];
		for (int j = 0; j < this->nfolds; j++){
			textonMap[j] = textonMapping(fVectors[j], textonDic);	//fVectors, textons Dictionary
			imwrite(to_string(j) + "textonMap.png", textonMap[j]);
			showImg(textonMap[j].clone(), (string("textonMap") + to_string(j)).c_str(), false, true);
		}
	}

	cout << "train-success" << endl;
}

//! testing patch-based textons
Mat PTexton::test(Mat textonMap, vector<Mat> histDB, int fold){
	cout << "test start" << endl;
	
	Mat trainData;
	//Random Projection
	if (this->RP == "yes"){
		Mat oneDB;
		vconcat(histDB, oneDB);

		vector<Mat> reduced;	//reduced
		for (int i = 0; i < oneDB.rows; i++){
			Mat r = oneDB.row(i);
			transpose(r, r);
			reduced.push_back(RandomProjection(r));
		}
		vconcat(reduced, trainData);
	}
	else{
		vconcat(histDB, trainData);
	}
	//K-nn training
	trainData.convertTo(trainData, CV_32F);
	Mat trainClass(trainData.rows,1,CV_32SC1);

	int n = 0;
	for (int i = 0; i < histDB.size(); i++){
		for (int j = 0; j < histDB.at(i).rows; j++){
			trainClass.at<int>(n) = i;
			n++;
		}
	}
	cout << "train success" << endl;

	cv::KNearest *knn = new KNearest(trainData, trainClass);
	Mat output = Mat::zeros(textonMap.rows - pSize, textonMap.cols - pSize,CV_32S);

	//trace test map
	for (int i = half_patch; i < textonMap.rows-2000 - half_patch; i++){
		for (int j = half_patch; j < textonMap.cols - half_patch; j++){

			//patch extraction in textonMap
			cv::Rect patchR(j-half_patch, i-half_patch, pSize, pSize);

			Mat patch = textonMap(patchR).clone();
			patch.convertTo(patch, CV_32F);

			//calculation of histogram in each patch
			int histSize = 4 * this->K;
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
				hist = RandomProjection(hist);
			}
			else{
				transpose(hist, hist);
				hist.convertTo(hist, CV_32F);
			}
			//cout << hist << endl;

			//find nearest neighbor
			//TODO:revise Knn
			int knn_k=3;
			Mat nearests(1, knn_k, CV_32FC1);
			float response = knn->find_nearest(hist, knn_k);// , 0, 0, nearests, 0);
			
			output.at<int>(i - half_patch,j - half_patch) = (int)response;
		}
		if (i%100==0)
			cout << "i=" << i << endl;
	}
	cout << "test end" << endl;
	showImg(output, "output"+to_string(fold), false, true);
	return output;
}

//! calculate histogram
vector<Mat> PTexton::trainHist(Mat textonMap,int fold){
	cout << "calculateHist start" << endl;

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
			int histSize = 4 * this->K;
			float range[] = { 0, histSize };
			const float* histRange = { range };
			bool uniform = true; bool accumulate = false;
			//cout << "patch size=" << patch.size() << endl;
			//cout << patch << endl;			

			Mat hist;
			calcHist(&patch, 1, 0, Mat(), hist, 1,&histSize, &histRange, uniform, accumulate);
			//cout << "hist size=" << hist.size() << endl;
			transpose(hist, hist);
			//normalize(hist, hist);

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
		if (dbSize == 0){
			cout << "There are no training data in class " << c << endl;
			continue;
		}
		Mat cHist(histDB[c].size(),histDB[c].at(0).cols,CV_32F);
		for (int i = 0; i < histDB[c].size(); i++){
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

	//read files in a directory
	string dir = "reference";
	std::vector<std::string> vFileNames = GetFileNamesInDirectory(dir);

	for (int i = 0; i < vFileNames.size(); i++){

		//define file name for saving result images
		string fname = string(dir) + string("\\") + vFileNames[i];

		cout << "load image:" << fname << endl;

		Mat img = imread(fname, 0);

		//adjust size of reference image for matching position of each pixel in input image
		//cout << "ref size:" << img.size() << endl;
		//cout << "image size:" << this->imageData.getSize() << endl;

		resize(img, img, this->imageData.getSize());
		//normalize(img, img, 0, 255, CV_MINMAX);
		cout << "resized ref size:" << img.size() << endl;
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
	std::cout << "There are less images than folds" << endl;
	
	cv::Size curSize = imageData.getSize();

	int height = curSize.height;
	int width = curSize.width;
	
	//use vertical folds
	for (int i = 0; i < this->nfolds; i++){
		int x = half_patch + ((width-pSize) / nfolds)*i;
		int y = half_patch+1000;

		//int w = (width - pSize) / nfolds;
	//	int h = (height - pSize)/50;
		int w = (width - pSize) / nfolds;
		int h = height - pSize-1000;

		cv::Rect tempRect(x, y, w, h);
		this->foldRect.push_back(tempRect);

		cout << i << "th fold=" << tempRect << endl;
	}

	// TODO:if training foldType = 0 else if testing foldType = 1
	int foldType = 0;

}
//! evaluate textons
void PTexton::evaluate(){

	std::cout << "Evaluation start" << endl;

	//// Image selection ////
	//selectCurrentImage();
	foldGeneration();
	loadReferenceData();

	////// feature extraction ////
	//for (int j = 0; j < 5;j++){// this->nfolds; j++){
	//	//feature extraction				
	//	this->generateFVectors(this->foldRect.at(j),j);

	//	//Random Projection
	//	//this->RandomProjection(j);

	//	//cluster textons
	//	this->clusterTextons(j);
	//}

	
	vector<Mat> output;
	//// training and testing in each fold ////
//	for (int i = 0; i < this->nfolds; i++){
		// training //// 	
		//
		////texton dictionary construction
		//vector<vector<Mat>> textonDic;
		//for (int j = 0; j < this->nfolds; j++){
		//	if (j!=i) //test data isn't included in training data
		//		textonDic.insert(textonDic.end(), textons[j].begin(), textons[j].end());
		//}

		////printCenter(centers);
		//cout << "K=" << textonDic.size() << endl;

		////texton mapping
		//Mat textonMap[5];
		//for (int j = 0; j < this->nfolds; j++){
		//	textonMap[j] = textonMapping(fVectors[j], textonDic);	//fVectors, textons Dictionary
		//	imwrite(to_string(j) + "textonMap.png", textonMap[j]);
		//	showImg(textonMap[j].clone(), (string("textonMap") + to_string(j)).c_str(), false, true);
		//}

	Mat textonMap[5];
	for (int i = 0; i < 5; i++){
		textonMap[i] = imread(to_string(i) + "textonMap.png", 0);
		//showImg(textonMap[i],"showtexton",true,false);
		//waitKey(0);
	}

	for (int i = 0; i < this->nfolds; i++){

		//histogram calculation
		//generate Database of histogram (size = # of classes)
		vector<Mat> histDB;
		for (int j = 0; j < this->nfolds; j++){
			if (j != i){ //test data isn't included in training data
				vector<Mat> tempHistDB = trainHist(textonMap[j].clone(),j);
				
				if (histDB.size() == 0){
					histDB = tempHistDB;
				}
				else{
					for (int k = 0; k < nclass; k++){
						Mat newHist;
						vconcat(histDB.at(k), tempHistDB.at(k), newHist);
						histDB.at(k) = newHist.clone();
					}					
				}
			}
		}
		
		//testing with i fold
		output.push_back(test(textonMap[i].clone(), histDB, i));

	}
	Mat classification_result;
	hconcat(output, classification_result);
	showImg(classification_result, "final output", false, true);
	//evaluation
//	ClassificationError error;


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
	int highD = fVectors[foldN].at(0).size();
	int lowD = highD / 3;
	Mat rMat;

	generateRandomMat(rMat, highD, lowD, "gaussian");//Rmode = "gaussian" or "achlioptas"
	//cout << rMat << endl;

	cout << "original vector size = " << fVectors[foldN].at(0).size() << endl;
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
		fVectors[foldN].at(j) = lowVector;
	}
	cout << "after RP, vector size = " << fVectors[foldN].at(0).size() << endl;
}
//! load image data
bool PTexton::loadImageData(void){
	ImgData img;

	cout << "file name:" << this->fname << endl;

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

//! clustering textons
void PTexton::clusterTextons(int num){

	cout << "kmeans clustering / K=" << this->K << endl;

	int foldnum = 0;

	vector<vector<Mat>> centers;
	vector<vector<Mat>> oldCenters;
	
	int fsize = fVectors[num].size();
	int vSize = fVectors[num].at(0).size();
	Mat label(fsize, 1, CV_32S);

	srand(time(NULL));
	// initialize centers
	for (int i = 0; i < this->K; i++){
		int random = rand() % fsize;
		cout << "random number = "<<random << endl;
		centers.push_back(fVectors[num].at(random));
		
		//dummy
		oldCenters.push_back(fVectors[num].at(i));
	}
		

	const int maxIter = 10;
	int iter = 0;
	// loop if stop criteria isn't satisfied
	while (iter < maxIter){
		Mat nlabel = Mat::zeros(K, 1,CV_32S);

		//save old center for stop criteria
		oldCenters = centers;
		cout << "feature vectors size = " << fVectors[num].size() << endl;
		

		//pre-calculation about center//
		vector<vector<Mat>> invCenters;
		vector<vector<float>> firstTerms;

		for (int k = 0; k < this->K; k++){	//cluster number
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

		//assign labels to dataset
		for (int i = 0; i < fsize; i++){
			float dist=0, minDist=0;

			//find min_distance with assigning label
			for (int k = 0; k < this->K; k++){	//cluster number
				dist = 0;
				for (int m = 0; m < vSize; m++){	//order in a patch
					if (imgType == POLSAR){
						//if (wishartDistance(centers.at(k).at(m).clone(), fVectors[num].at(i).at(m).clone()) == -1){
						//efficient version
						float wDist = wishartDistance(firstTerms.at(k).at(m), invCenters.at(k).at(m).clone(), fVectors[num].at(i).at(m).clone());
						if (wDist==-1){
							cout << "iteration="<<iter<<"fold=" << num << ", at=" << i <<",at(in vector)="<<m<< endl;
						}
							dist += wDist;
					}
					else if (imgType == GRAY){
						dist += abs(fVectors[num].at(i).at(m).at<int>(0) - centers.at(k).at(m).at<int>(0));
					}
				}
				dist /= (float)vSize;
				//cout << "distance=" << dist << ",";
				if (k == 0){
					label.at<int>(i)=0;
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
			//if (i % 10 == 0)
				//cout << i << "th vector label=" << label.at<int>(i) << endl;

		}
		//cout << "assign label -complete" << endl;

		cout << "nlabel=" << nlabel << endl;
				
		//calculate new centers
		//initialize centers
		vector<vector<Mat>> newcenters;
		for (int i = 0; i < K; i++){
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
				newcenters.at(label.at<int>(i)).at(j) += fVectors[num].at(i).at(j).clone();
			}
		}
		//cout << "add feature matrix-complete" << endl;

		//mean of feature vector (centers)
		for (int i = 0; i < K; i++){
			for (int j = 0; j < vSize; j++){	//in a patch vector
				if (nlabel.at<int>(i)>1){
					newcenters.at(i).at(j) /= (float)nlabel.at<int>(i);
				}
			}
			//cout << "K=" << i << "center matrix" << endl<<newcenters.at(i).at(0) << endl;
		}
		cout << "make centers - complete" << endl;
		
		////improve initial center setting////
		/*double min, max;
		Point mini, maxi;
		minMaxLoc(nlabel, &min, &max, &mini, &maxi);
*/
		for (int i = 0; i < K; i++){
			if (nlabel.at <int>(i) == 1){
				int random = rand() % fsize;
				newcenters.at(i) = fVectors[num].at(random);
			}
		}

		//stop criteria		
		//oldCenters != centers
		
		bool same=true;
		for (int i = 0; i < K; i++){
			for (int j = 0; j < vSize; j++){
				Mat diff,diff2;
				Mat old[2], newc[2];
				split(oldCenters.at(i).at(j), old);
				split(newcenters.at(i).at(j), newc);
				cv::compare(old[0], newc[0], diff, cv::CMP_NE);
				cv::compare(old[1], newc[1], diff2, cv::CMP_NE);
				int nz = countNonZero(diff);
				int nz2 = countNonZero(diff2);
				if (nz != 0){same = false;	break;}
				else{if (nz2 != 0){	same = false; break;}
				}
			}
			if (same == false){	break;}
		}
		if (same == true){ 
			cout << "iteration stop i = " << iter << endl;
			break;
		}

		centers = newcenters;
		newcenters.clear();
		//nlabel.release();// = Mat::zeros(nlabel.size(), nlabel.type());
		iter++;
	}

	cout << "cluster Textons success" << endl;
	this->textons[num] = centers;
		
	Mat map(foldRect[num].size(), CV_32S);
	
	int n = 0; 
	for (int i = 0; i < map.rows; i++){
		for (int j = 0; j < map.cols; j++){
			map.at<int>(i, j) = label.at<int>(n);
			n++;
		}
	}
	showImg(map, "textonMap from K-means_"+to_string(num), false, true);
	//waitKey(0);

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
		normalize(aux, aux, 0, 255, CV_MINMAX);
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