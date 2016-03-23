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

//! the constructor
PTexton::PTexton(string fname, int patchSize, int K, string rp){
	
	this->fname = fname;
	this->K = K;
	this->pSize = patchSize;
	this->half_patch = this->pSize / 2;
	this->imgType = COLOR;
	this->vSize = 0;
	this->RP = rp;

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
	cout << "generate Random Matrix" << endl;

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
			for (int j = 0; j < this->vSize; j++){
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
	cout << "generate Random Matrix - success" << endl;

}
void PTexton::calculateFVecRP(Mat tImg, Mat& featureVec){
	//ignore boundary pixel of image which can not make complete patch(size: pSize x pSize)
	int r = tImg.rows - this->pSize;
	int c = tImg.cols - this->pSize;
	int nVector = r*c;

	// vSize: size before RP (highD)
	int highD = this->pSize*this->pSize;

	// define lowD for dimension reduction
	int lowD = highD / 4;
	Mat randomMat;
	generateRandomMat(randomMat, highD, lowD, "gaussian");//Rmode = "gaussian" or "achlioptas"

	//this->vSize = highD*tImg.channels();
	this->vSize = lowD*tImg.channels();	//for RP

	//input matrix for k-means
	featureVec.create(nVector, 1, CV_32FC(vSize));	////mat type must same to vector type

	float* p;
	for (int i = half_patch, n = 0; i < r + half_patch; i++){
		for (int j = half_patch; j < c + half_patch; j++, n++){

			Mat highMat(highD, 1, tImg.type());

			//trace each pixel in a patch
			for (int k = -half_patch, n2 = 0; k <= half_patch; k++){
				for (int l = -half_patch; l <= half_patch; l++, n2++){
					if (i + k < tImg.rows && j + l < tImg.cols&&i + k >= 0 && j + l >= 0){
						switch (this->imgType){
						case GRAY:
							highMat.at<float>(n2) = tImg.at<float>(i + k, j + l);	break;
						case COLOR:
							highMat.at<Vec3f>(n2)[0] = tImg.at<Vec3f>(i + k, j + l)[0];
							highMat.at<Vec3f>(n2)[1] = tImg.at<Vec3f>(i + k, j + l)[1];
							highMat.at<Vec3f>(n2)[2] = tImg.at<Vec3f>(i + k, j + l)[2];	break;
						case POLSAR:
							break;
						}
					}
					else{
						cerr << "ERROR: boundary check(" << i + k << "," << j + l << ")" << endl;
						break;
					}					
				}
			}
			//for multiple channel,  split, mutply and then merge
			Mat lowMat;// = randomMat*highMat;
			switch (highMat.channels()){
			case 1:
				lowMat = randomMat*highMat;
				break;
			case 3:
				vector<Mat> eachHighCh;
				split(highMat, eachHighCh);
				for (int c = 0; c < 3; c++){
					eachHighCh[c] = randomMat*eachHighCh[c];
				}
				merge(eachHighCh, lowMat);
			}

			//vector normailzation is needed to increase the robustness
			normalize(lowMat, lowMat, 0, 255, CV_MINMAX);
			
			p = featureVec.ptr<float>(n);
			for (int a = 0,n2=0; a < lowD; a++,n2++){
				switch (this->imgType){
				case GRAY:
					p[n2] = lowMat.at<float>(a);	break;
				case COLOR:
					p[n2] = lowMat.at<Vec3f>(a)[0];	n2++;
					p[n2] = lowMat.at<Vec3f>(a)[1];	n2++;
					p[n2] = lowMat.at<Vec3f>(a)[2];	break;
				case POLSAR:

					break;
				}
			}

		}
	}
}

void PTexton::calculateFVecNRP(Mat tImg, Mat& featureVec){
	//ignore boundary pixel of image which can not make complete patch(size: pSize x pSize)
	int r = tImg.rows - this->pSize;
	int c = tImg.cols - this->pSize;
	int nVector = r*c;

	// vSize: size before RP (highD)
	int highD = this->pSize*this->pSize;
	this->vSize = highD*tImg.channels();	//no RP

	//input matrix for k-means
	featureVec.create(nVector, 1, CV_32FC(vSize));	////mat type must same to vector type

	float* p;
	for (int i = half_patch, n = 0; i < r + half_patch; i++){
		for (int j = half_patch; j < c + half_patch; j++, n++){
			p = featureVec.ptr<float>(n);

			//trace each pixel in a patch
			for (int k = -half_patch, n2 = 0; k <= half_patch; k++){
				for (int l = -half_patch; l <= half_patch; l++, n2++){
					if (i + k < tImg.rows && j + l < tImg.cols&&i + k >= 0 && j + l >= 0){
						switch (this->imgType){
						case GRAY:
							p[n2] = tImg.at<float>(i + k, j + l);	break;
						case COLOR:
							p[n2] = tImg.at<Vec3f>(i + k, j + l)[0];	n2++;
							p[n2] = tImg.at<Vec3f>(i + k, j + l)[1];	n2++;
							p[n2] = tImg.at<Vec3f>(i + k, j + l)[2];	break;
						case POLSAR:

							break;
						}
					}
					else{
						cerr << "ERROR: boundary check(" << i + k << "," << j + l << ")" << endl;
						break;
					}
				}
			}
		}
	}
}
Mat PTexton::extractFVec(string mode, string RP)
{
	cout << "extract feature vector" << endl;

	Mat img;
	if (mode == "train"){
		img = this->currentImg(this->trainRect);
	}
	else if (mode == "test"){
		img = this->currentImg(this->testRect);
	}

	Mat FVec;
	if (RP == "yes"){
		calculateFVecRP(img, FVec);
	}
	else if (RP == "no"){
		calculateFVecNRP(img, FVec);
	}
	cout << "extract feature vector-success" << endl;

	return FVec;
}

void PTexton::learningTexton(void){

	cout << "learningTexton start" << endl;

	int i = 0;

	for (vector<ImgData*>::iterator img = imageData.begin(); img != imageData.end(); img++, i++){
		cout << "image # = " << i << endl;

		// project it down (is the identiy function here, because multi-pol is the highest data-level, but the projections to lower data-levels will be performed here, too, so they are available later)
		(*img)->project();

		//get patch vector 
		int nRep = (*img)->getNumberOfRepresentations(POLSAR);

		for (int rep = 0; rep < nRep; rep++){

			Mat tImg;	//Learning image
			Mat grayData;
			Mat colData = (*img)->getData(COLOR, rep);
			vector<Mat> polData;

			switch (this->imgType){
			case GRAY:
				cvtColor(colData, grayData, CV_BGR2GRAY);
				tImg = grayData.clone(); break;
			case COLOR:
				tImg = colData.clone();// (*img)->getData(COLOR, rep);	
				break;
			case POLSAR:
				for (int c = 0; c < 3; c++){
					Mat channel = (*img)->getData(POLSAR, rep);
					if (channel.rows>0){
						polData.push_back(channel);
					}
				}

				//tImg;
				break;
			}
			this->currentImg = tImg.clone();

			int r = currentImg.rows;
			int c = currentImg.cols;

			//define ROI for training
			this->trainRect = cv::Rect(c * 4 / 5, 0, c - c * 4 / 5, r); 
			this->testRect = cv::Rect(0, 0, c * 3 / 5, r);

			//variable for K-means
			Mat featureVec = extractFVec("train",this->RP); // (train, RP)

			clusterTextons(featureVec);
		}
	}
	cout << "learningTexton end" << endl;
}
void PTexton::textonMapping(Mat featureVec, Mat& textonMap, string mode){
	cout << "textonMapping" << endl;

	Mat trainData = this->textons.clone();
	//showImg(trainData, "textons", true, false);

	Mat trainClasses(trainData.rows, 1, CV_32F);

	for (int i = 0; i < this->K; i++){
		trainClasses.at<float>(i) = i;
	}

	// learn classifier
	cv::KNearest *knn;
	knn = new KNearest(trainData, trainClasses);
	//Mat nearests(1, K, CV_32FC(vSize));

	Mat outputImg;	//output matrix from texton coding
	int r, c;
	if (mode == "train"){
		r = trainRect.height;
		c = trainRect.width;
	}
	else if (mode == "test"){
		r = testRect.height;
		c = testRect.width;
	}

	outputImg.create(r-pSize, c-pSize, CV_8U);

	cout << "r=" << r << ",c=" << c << endl;
	cout << "feature vector size =" << featureVec.rows << endl;
	cout << "r*c=" << (r-pSize)*(c-pSize) << endl;

	//code texton to train and test image
	float* p;
	int n = 0;
	for (int i = 0; i < r - pSize; i++){
		for (int j = 0; j < c - pSize; j++, n++){
			p = featureVec.ptr<float>(n);

			Mat sample(1, this->vSize, CV_32F);
		
			for (int m = 0; m < vSize; m++){
				sample.at<float>(0, m) = p[m];
			}

			// estimate the response and get the neighbors' labels
			int response = knn->find_nearest(sample, 1);
			//cout << response;
			outputImg.at<uchar>(i, j) = (uchar)response;
		}
		//cout << endl;
	}
	cout << "n=" << n << endl;
	textonMap = outputImg.clone();

	showImg(outputImg, "texton Mapping", false, true);
	cout << "Texton Mapping - success" << endl;

}
//! training patch-based textons
void PTexton::train(){

	cout << "train" << endl;

	//read reference image
	loadReferenceData();
	
	//variable for K-means
	Mat featureVec = extractFVec("train", this->RP); // (train, RP)

	//match texton for each pixel using feature vector of each patch
	Mat textonMap;
	textonMapping(featureVec, textonMap, "train");
	
	this->nPatches = trainHist(textonMap);
	
	cout << "train-success" << endl;
}
void PTexton::classification(Mat textonMap){

	Mat trainData = this->histDB.clone();
	////showImg(trainData, "textons", true, false);

	Mat trainClasses(trainData.rows, 1, CV_32F);
	int n = 0;
	for (int j = 0, i = 0; j < referenceData.size(); j++){
		//cout << n << endl;
		n += this->nPatches[j];

		for (; i<n; i++){
			trainClasses.at<float>(i) = j;
		}
	}
	//cout << n << endl;

	//// learn classifier
	cv::KNearest *knn;
	knn = new KNearest(trainData, trainClasses);
	//Mat nearests(1, K, CV_32FC(vSize));

	Mat outputImg;	//output matrix 
	int r = testRect.height;
	int c = testRect.width;

	outputImg.create(r - 2*pSize, c - 2*pSize, CV_8U);

	for (int i = this->pSize; i < r - pSize; i++){
		for (int j = this->pSize; j < c - pSize; j++){

			//calculate histogram in test image
			Mat sample = Mat::zeros(1, this->K, CV_32F);
			for (int k = -half_patch; k <= half_patch; k++){
				for (int l = -half_patch; l <= half_patch; l++){
					int textonVal = textonMap.at<uchar>(i + k-half_patch, j + l-half_patch);
					//cout << textonVal << " ";
					//cout << "(x,y)=(" << i + k - half_patch << "," << j + l - half_patch << "):";
					sample.at<float>(0, textonVal) += 1;
				}
			}

			// estimate the response and get the neighbors' labels
			int response = knn->find_nearest(sample, 1);
			outputImg.at<uchar>(i - pSize, j - pSize) = (uchar)response;
		}
	}

	showImg(outputImg, "classification result", true, true);
	this->resultImg = outputImg;

}

//! testing patch-based textons
void PTexton::test(void){
	cout << "test start" << endl;
	//classification


	//variable for K-means
	Mat featureVec = extractFVec("test", this->RP); // (train, RP)

	//match texton for each pixel using feature vector of each patch
	Mat textonMap;
	textonMapping(featureVec, textonMap, "test");

	classification(textonMap);

}

//! calculate histogram
vector<int> PTexton::trainHist(Mat textonMap){
	cout << "calculateHist start" << endl;

	//rows and columns
	int r, c;
	Mat histogram;
	
	r = trainRect.height;
	c = trainRect.width;
	
	histogram = Mat::zeros(referenceData.size()*r*c, this->K, CV_32F);
	nPatches.clear();

	//number of all histogram
	int n2 = 0;

	cout << "patch size=" << pSize << endl;
	cout << "half_patch size = " << half_patch << endl;

	for (int n = 0; n < referenceData.size(); n++){
		Mat refImg = referenceData.at(n).clone();
		Mat refTrain = refImg(trainRect);
		//imshow("ref", refTrain);

		//patch count for each class
		int patch_count = 0;
		cout << r << "," << c << endl;
		cout << textonMap.size() << endl;

		//calculate histogram
		for (int i = this->pSize; i < r-this->pSize; i++){
			for (int j = this->pSize; j < c-this->pSize; j++){
				int val= (int)refTrain.at<uchar>(i, j);

				//the position (i,j) is included in nth class
				if (val){
					patch_count++;

					//trace each pixel in a patch
					for (int k = -half_patch; k <= half_patch; k++){
						for (int l = -half_patch; l <= half_patch; l++){
							cv::Point p(j + l - half_patch, i + k - half_patch);
							cv::Rect rect(cv::Point(), textonMap.size());

							if (rect.contains(p)){
								int textonVal = textonMap.at<uchar>(p);
								//cout << textonVal << " ";
								histogram.at<float>(n2, textonVal) += 1;
							}
							else{
								cerr << "check boundary:" << p << endl;
								break;
							}
						}
					}
					n2++; j += this->pSize;
					//cout << n2<<endl;
				}
			}
			i += this->pSize;
			//cout << endl;
		}
		//store number of patches in each class
		cout << "patch_count(" << n << ")=" << patch_count << endl;
		nPatches.push_back(patch_count);
	}
	cout << "n2=" << n2 << endl;

	//resize histogram 
	this->histDB = histogram(cv::Range(0,n2), cv::Range(0,this->K));
	cout << "calculate hist-end" << endl;

	//normalize(histDB,histDB, 0, 255, CV_MINMAX);

	return nPatches;
}

void PTexton::loadReferenceData(){
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
		//cout << "current size (" << this->currentImg.rows << "," << this->currentImg.cols << endl;
		
		resize(img, img, this->currentImg.size());
		normalize(img, img, 0, 255, CV_MINMAX);
		//cout << "resized ref size (" << img.rows << "," << img.cols << endl;
		this->referenceData.push_back(img);

		if (!img.data){
			cout << "ERROR: original image not specified" << endl;
			cout << "Press enter to exit" << endl;
			cin.get();
			return;
		}
	}
	cout << "success" << endl;
}
//! evaluate textons
void PTexton::evaluate(){



}

//! load image data
bool PTexton::loadImageData(void){
	ImgData* img;

	cout << "file name:" << this->fname << endl;

	if (this->fname.size() > 0){
		img = new ImgData(this->fname, POLSAR);
		img->load();
		this->imageData.push_back(img);
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
void PTexton::clusterTextons(Mat featureVec){

	cout << "kmeans clustering";

	Mat labels, centers;
	Mat outputImg;	//output matrix from k-means	

	centers.create(this->K, 1, featureVec.type());
	outputImg.create(this->trainRect.height-pSize,this->trainRect.width-pSize, CV_32F);
	//outputImg.create(this->train_rows, this->train_cols, CV_32F);

//	Mat trainFeature = featureVec(cv::Range(new_rows*(new_cols -train_cols)+1, new_rows*new_cols),cv::Range(0,1));

	int attempts = 3, flags = cv::KMEANS_RANDOM_CENTERS;
	TermCriteria tc(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1.0);
	kmeans(featureVec, this->K, labels, tc, attempts, flags, centers);
	cout << "-success" << endl;
	
	this->textons = centers.clone();
	//printCenter(centers);

	//compute clustered image 			
	for (int i = 0, n = 0; i < outputImg.rows; i++){
		for (int j = 0; j < outputImg.cols; j++, n++){
			int cIndex = labels.at<int>(n);

			outputImg.at<float>(i, j) = (float)cIndex; // / (float)this->K;
		}
	}
	showImg(outputImg, "cluster",false, true);
	cout << "cluster Textons success" << endl;
}

// Function displays image (after proper normalization)
/*
img   :  Image that shall be displayed
cut   :  whether to cut or scale values outside of [0,255] range
*/
void PTexton::showImg(Mat img, const char* win, bool show, bool save){

	Mat aux;// = img.clone();

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
		imwrite((string(win) + string(".png")).c_str(), aux);
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