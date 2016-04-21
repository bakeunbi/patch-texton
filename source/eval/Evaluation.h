/*
* author : Haensch Ronny
*/

#ifndef EVALUATION_H
#define EVALUATION_H

#include <vector>

#include "ClassificationError.h"
#include "PRFDataPoint.h"
#include "ImgData.h"

using namespace std;

//! this class handles the (generic) evaluation (ie. cross validation) of classification systems
template <typename T>
class Evaluation{

	public:
		//! constructor
		Evaluation(void){};
		//! constructor for PRF-classifier
		Evaluation(vector<PRFDataPoint*>& data, int numberOfFolds, int type, int answerLevel=-1, bool horizontalFolds=true, vector<ImgData*>* imageData=0);
		//! generic constructor
		Evaluation(vector<T>& data, int numberOfFolds, int answerLevel=-1);
		~Evaluation(void){};
				
		//! divides data into folds
		void generateFolds(vector<T>& data);
		//! divides data into folds
		void generateFolds(vector<PRFDataPoint*>& data, int type, bool horizontalFolds=true, vector<ImgData*>* imageData=0);
		//! uses parts of an image as folds
		void imagePartsAsFold(vector<PRFDataPoint*>& data, bool horizontalFolds, vector<ImgData*>* imageData);
		//! uses whole images as folds
		void imagesAsFold(vector<PRFDataPoint*>& data, vector<ImgData*>* imageData);

		//! increases the fold id
		void incFoldID(void){currentFold++;};
		//! returns the current training data
		void getTrainData(vector<T>& data);
		//! returns the current training data
		void getTrainData(vector<PRFDataPoint*>& data, Mat prior, double trainSetSize);
		//! forces the training dataset to follow a certain class distribution
		void forceTrainingPrior(vector<PRFDataPoint*>& trainSet, Mat prior, double trainSetSize);
		//! returns the current test data
		void getTestData(vector<T>& data);
		//! returns the size of the current fold
		int getFoldSize(int f){return foldData.at(f).size();};

		//! adds an errror statistic based on current estimate
		void addErrorStatistics(Mat& estimate, Mat& reference);
		//! adds an errror statistic based on current estimate
		void addErrorStatistics(vector< vector<Mat> >& estimate, vector< vector<Mat> >& reference);
		//! prints errror statistic
		void printCurrentError(void){error.at(this->currentFold)->print();};
		
		//! starts evaluation
		void evaluate(void);
		//! prints overall error (averaged over all folds)
		void printFinalError(void){error.at(this->numberOfFolds)->print();};
		
		//! which type has the system answer
		int getAnswerLevel(void){return this->answerLevel;};
		//! returns number (id) of current fold
		int getCurrentFold(void){return this->currentFold;};
		//! returns number of fold
		int getNumberOfFolds(void){return this->numberOfFolds;};
				
		//! defines type of system answer
		void setAnswerLevel(int level){this->answerLevel = level;};
		//! defines current fold
		void setCurrentFold(int fold){this->currentFold = fold;};
		//! defines number of folds
		void setNumberOfFolds(int num){this->numberOfFolds = num;};
		//! sets prior
		void setPrior(Mat& p){ this->prior = p.clone();};

	private:
		//! the type of the system answer
		int answerLevel;

		//! current fold
		int currentFold;
		//! number of folds
		int numberOfFolds;
		
		//! the data of the individual folds
		vector< vector< T > > foldData;
		//! to which fold each pixel belongs
		vector< Mat > foldID;
		
		//! this vector will contain the error-objects for each cv-iteration
		vector< ClassificationError*> error;
		//! this vector will contain the error-objects of the reference-classificators for each cv-iteration
		vector< vector< ClassificationError*> > refError;

};

template <typename T> Evaluation<T>::Evaluation(vector<T>& data, int numberOfFolds, int answerLevel){

  this->answerLevel = answerLevel;

  this->currentFold = 0;
  this->numberOfFolds = numberOfFolds;
		
  this->error.resize(numberOfFolds);
  this->refError.resize(2);
  for(vector< vector< ClassificationError*> >::iterator ref = this->refError.begin(); ref != this->refError.end(); ref++){
	  ref->resize(numberOfFolds);
  }
  foldData.resize(numberOfFolds);

  generateFolds(data);

}

template <typename T> void Evaluation<T>::evaluate(void){
	this->error.push_back(new ClassificationError(this->error));
}

template <typename T> void Evaluation<T>::generateFolds(vector<T>& data){
	
	int n=0;
	for(typename vector<T>::iterator it = data.begin(); it != data.end(); it++, n++){
		foldData.at( n % foldData.size() ).push_back(*it);
		Mat id(1,1, CV_8UC1, (n % foldData.size()));
		foldID.push_back(id);
	}
	
}

template <typename T> void Evaluation<T>::getTestData(vector<T>& data){

	for(typename vector<T>::iterator dp = foldData.at(this->currentFold).begin(); dp != foldData.at(this->currentFold).end(); dp++){
		data.push_back(*dp);
	}	
}

template <typename T> void Evaluation<T>::getTrainData(vector<T>& data){

	for(int f = 0; f < foldData.size(); f++){
		if (f != this->currentFold){
			for(typename vector<T>::iterator dp = foldData.at(f).begin(); dp != foldData.at(f).end(); dp++){
				data.push_back(*dp);
			}
		}
	}	
}

template <typename T> void Evaluation<T>::addErrorStatistics(Mat& estimate, Mat& reference){
/*
      // init
      curError = new ClassificationError();
      // estimate
      curError->estimate(refData, estData_ll, fold, numberOfFolds, horizontalFolds);*/

}

template <typename T> void Evaluation<T>::addErrorStatistics(vector< vector<Mat> >& estimate, vector< vector<Mat> >& reference){

      // init
      error.at(this->currentFold) = new ClassificationError(reference, estimate, this->currentFold, this->foldID);

}

#endif
