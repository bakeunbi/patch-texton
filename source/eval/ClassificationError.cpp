#include "ClassificationError.h"

// "fold" is the number of the current fold. "foldID" stores for each pixel in each image to which fold it belongs, 
// ie. whether it has to be regarded as training or test data within the current fold.
// default constructor
ClassificationError::ClassificationError(vector< vector<Mat> >& reference, vector< vector<Mat> >& estimate, int fold, vector<Mat>& foldID){

  // mean
  this->errAbs[0] = this->errAbs[1] = this->errLoss[0] = this->errLoss[1] = this->margin[0] = this->margin[1] = 0;
  // standard deviation
  this->errAbsStd[0] = this->errAbsStd[1] = this->errLossStd[0] = this->errLossStd[1] = this->marginStd[0] = this->marginStd[1] = 0;
  // allocate and init confusion matrices
  double numberOfClasses = reference.front().size();
  this->confusionAbs[0] = Mat::zeros(numberOfClasses, numberOfClasses+1, CV_32FC1);
  this->confusionAbs[1] = Mat::zeros(numberOfClasses, numberOfClasses+1, CV_32FC1);
  this->confusionLoss[0] = Mat::zeros(numberOfClasses, numberOfClasses+1, CV_32FC1);
  this->confusionLoss[1] = Mat::zeros(numberOfClasses, numberOfClasses+1, CV_32FC1);
  
  // some temporary variables
  Mat tmp;
  double count[2] = {0, 0};
  double e, curProb_ref, curProb_est, maxProb_ref, maxProb_est, sndMaxProb_ref, sndMaxProb_est;
  int bestClass_ref, bestClass_est;
  
  for(int c=0; c<reference.begin()->size(); c++){
	  this->stats[0].push_back(Mat::zeros(11,4,CV_32FC1));
	  this->stats[1].push_back(Mat::zeros(11,4,CV_32FC1));
  }
  
  // loop through images
  int i = 0, isTestData;
  for(vector< vector<Mat> >::iterator ref=reference.begin(), est=estimate.begin(); ref != reference.end(); ref++, est++, i++){
      // loop through all pixel
      for(int y=0; y<(*ref).at(0).rows; y++){
		  for(int x=0; x<(*ref).at(0).cols; x++){

			  // if no class was defined in reference data, this pixel wont count in accuracy assessment
			  double sum_est, sum_ref;
			  sum_est = sum_ref = 0;
			  for(int c=0; c<(*ref).size(); c++){
				  sum_est += (*est).at(c).at<float>(y, x);
				  sum_ref += (*ref).at(c).at<float>(y, x);
			  }
			  if ( (sum_est <= 0) || (sum_ref <= 0) ) continue;

			  e = 0;
			  maxProb_ref = maxProb_est = sndMaxProb_ref = sndMaxProb_est = -1;
			  bestClass_ref = bestClass_est = -1;
			  // loop through classes
			  // seek maximum class-probability in reference and estimation data
			  for(int c=0; c<(*ref).size(); c++){
				  // get probability of this class within the reference data
				  curProb_ref = (*ref).at(c).at<float>(y, x);
				  // get probability of this class within the estimation
				  curProb_est = (*est).at(c).at<float>(y, x);
				  // seek probability of best class in reference data
				  if (maxProb_ref < curProb_ref){
					  sndMaxProb_ref = maxProb_ref;
					  maxProb_ref = curProb_ref;
					  bestClass_ref = c;
				  }
				  // seek probability of snd best class in reference data
				  if ( (sndMaxProb_ref < curProb_ref) && (maxProb_ref > curProb_ref) ){
					  sndMaxProb_ref = curProb_ref;
				  }
				  // seek probability of best class in estimation data
				  if (maxProb_est < curProb_est){
					  sndMaxProb_est = maxProb_est;
					  maxProb_est = curProb_est;
					  bestClass_est = c;
				  }
				  // seek probability of snd best class in estimation data
				  if ( (sndMaxProb_est < curProb_est) && (maxProb_est > curProb_est) ){
					  sndMaxProb_est = curProb_est;
				  }
				  // store absolute difference
				  e += abs(curProb_ref - curProb_est);
			  }
			  
			  bool isTestData = (foldID.at(i).at<uchar>(min(y, foldID.at(i).rows-1),min(x, foldID.at(i).cols-1)) == fold);
			  
			  for(int c=0; c<(*ref).size(); c++){
				  for(int t=0; t<=10; t++){
					  double thresh = t / 10.;
					  double curProb_est_t, maxProb_est_t=-1;
					  int bestClass_est_t = -1;
					  for(int cc=0; cc<(*ref).size(); cc++){
						  if (cc == c)
							curProb_est_t = (1-thresh) * (*est).at(cc).at<float>(y, x);
						  else
						    curProb_est_t = thresh * (*est).at(cc).at<float>(y, x);
						  // seek probability of best class in estimation data
						  if (maxProb_est_t < curProb_est_t){
							maxProb_est_t = curProb_est_t;
							bestClass_est_t = cc;
						  }
					  }
					  if ( (bestClass_ref == c) && (bestClass_est_t == c) ) stats[isTestData].at(c).at<float>(t,0)++; // TP
					  if ( (bestClass_ref == c) && (bestClass_est_t != c) ) stats[isTestData].at(c).at<float>(t,1)++; // FN
					  if ( (bestClass_ref != c) && (bestClass_est_t == c) ) stats[isTestData].at(c).at<float>(t,2)++; // FP
					  if ( (bestClass_ref != c) && (bestClass_est_t != c) ) stats[isTestData].at(c).at<float>(t,3)++; // TN
				  }
			  }
			  // add absolute difference of current pixel to overall sum
			  this->errAbs[isTestData] += e/numberOfClasses;
			  // check if another class as the true class was estimated, and if so, denote it as error
			  if (bestClass_ref != bestClass_est){
				this->errLoss[isTestData]++;
			  }
			  // add margin
			  this->margin[isTestData] += (maxProb_est - sndMaxProb_est);

			  // update 0-1-loss confusion matrix
			  this->confusionLoss[isTestData].at<float>(bestClass_ref, bestClass_est)++;
			  // save number class-samples in last column
			  this->confusionLoss[isTestData].at<float>(bestClass_ref, this->confusionLoss[isTestData].cols-1)++;

			  // update absolute difference confusion matrix
			  for(int c=0; c<(*ref).size(); c++){
				  // add certainty to about class c to confusion matrix
				  this->confusionAbs[isTestData].at<float>(bestClass_ref, c) += (*est).at(c).at<float>(y, x);
				  // save "number" class-samples in last column (its "number" because it will be 1 when summed over all classes)
				  this->confusionAbs[isTestData].at<float>(bestClass_ref, this->confusionAbs[isTestData].cols-1) += (*est).at(c).at<float>(y, x);
			  }
			  // denote number of samples in test and train set
			  count[isTestData]++;
		  }
      }   
  }
    
  // normalize errors
  this->errAbs[0] /= count[0];
  this->errLoss[0] /= count[0];
  this->margin[0] /= count[0];
  this->errAbs[1] /= count[1];
  this->errLoss[1] /= count[1];
  this->margin[1] /= count[1];

}

// average over individual error-estimates of different folds
ClassificationError::ClassificationError(vector< ClassificationError*>& cvError){
  
    // temporary variables that will contain sum and sum of squares of the errors
    double sumErrAbs[2] = {0,0};
    double sumErrLoss[2] = {0,0};
    double sumMargin[2] = {0,0};
    double sum2ErrAbs[2] = {0,0};
    double sum2ErrLoss[2] = {0,0};
    double sum2Margin[2] = {0,0};

    // allocate and init confusion matrices
    this->confusionAbs[0]  = Mat::zeros(cvError.front()->getConfusionAbs(0).rows,  cvError.front()->getConfusionAbs(0).cols,  CV_32FC2);
    this->confusionAbs[1]  = Mat::zeros(cvError.front()->getConfusionAbs(1).rows,  cvError.front()->getConfusionAbs(1).cols,  CV_32FC2);
    this->confusionLoss[0] = Mat::zeros(cvError.front()->getConfusionLoss(0).rows, cvError.front()->getConfusionLoss(0).cols, CV_32FC2);
    this->confusionLoss[1] = Mat::zeros(cvError.front()->getConfusionLoss(1).rows, cvError.front()->getConfusionLoss(1).cols, CV_32FC2);
    
    for(int c=0; c<cvError.front()->getConfusionAbs(0).rows; c++){
		this->stats[0].push_back( Mat::zeros(11,4, CV_32FC1) );
		this->stats[1].push_back( Mat::zeros(11,4, CV_32FC1) );
	}

    // sum over all individual estimates
    for(vector< ClassificationError*>::iterator err = cvError.begin(); err != cvError.end(); err++){
		sumErrAbs[0] += (*err)->getErrAbs(0);
		sumErrAbs[1] += (*err)->getErrAbs(1);
		sumErrLoss[0] += (*err)->getErrLoss(0);
		sumErrLoss[1] += (*err)->getErrLoss(1);
		sumMargin[0] += (*err)->getMargin(0);
		sumMargin[1] += (*err)->getMargin(1);
		
		sum2ErrAbs[0] += pow((*err)->getErrAbs(0),2);
		sum2ErrAbs[1] += pow((*err)->getErrAbs(1),2);
		sum2ErrLoss[0] += pow((*err)->getErrLoss(0),2);
		sum2ErrLoss[1] += pow((*err)->getErrLoss(1),2);
		sum2Margin[0] += pow((*err)->getMargin(0),2);
		sum2Margin[1] += pow((*err)->getMargin(1),2);
		
		// get individual mean and std conf-mat entry
		double n;
		for(int c1=0; c1<(*err)->getConfusionAbs(0).rows; c1++){
			n = (*err)->getConfusionAbs(0).at<float>(c1, (*err)->getConfusionAbs(0).cols-1);
			for(int c2=0; c2<(*err)->getConfusionAbs(0).cols-1; c2++){
				this->confusionAbs[0].at<Vec2f>(c1, c2) += Vec2f( (*err)->getConfusionAbs(0).at<float>(c1, c2)/n, pow((*err)->getConfusionAbs(0).at<float>(c1, c2)/n,2) );
			}
			this->confusionAbs[0].at<Vec2f>(c1, this->confusionAbs[0].cols-1) += Vec2f(1,1);
		}
		for(int c1=0; c1<(*err)->getConfusionAbs(1).rows; c1++){
			n = (*err)->getConfusionAbs(1).at<float>(c1, (*err)->getConfusionAbs(1).cols-1);
			for(int c2=0; c2<(*err)->getConfusionAbs(1).cols-1; c2++){
				this->confusionAbs[1].at<Vec2f>(c1, c2) += Vec2f( (*err)->getConfusionAbs(1).at<float>(c1, c2)/n, pow((*err)->getConfusionAbs(1).at<float>(c1, c2)/n,2) );
			}
			this->confusionAbs[1].at<Vec2f>(c1, this->confusionAbs[1].cols-1) += Vec2f(1,1);
		}
		for(int c1=0; c1<(*err)->getConfusionLoss(0).rows; c1++){
			n = (*err)->getConfusionLoss(0).at<float>(c1, (*err)->getConfusionLoss(0).cols-1);
			for(int c2=0; c2<(*err)->getConfusionLoss(0).cols-1; c2++){
				this->confusionLoss[0].at<Vec2f>(c1, c2) += Vec2f( (*err)->getConfusionLoss(0).at<float>(c1, c2)/n, pow( (*err)->getConfusionLoss(0).at<float>(c1, c2)/n,2) );
			}
			this->confusionLoss[0].at<Vec2f>(c1, this->confusionLoss[0].cols-1) += Vec2f(1,1);
		}
		for(int c1=0; c1<(*err)->getConfusionLoss(1).rows; c1++){
			n = (*err)->getConfusionLoss(1).at<float>(c1, (*err)->getConfusionLoss(1).cols-1);
			for(int c2=0; c2<(*err)->getConfusionLoss(1).cols-1; c2++){
				this->confusionLoss[1].at<Vec2f>(c1, c2) += Vec2f( (*err)->getConfusionLoss(1).at<float>(c1, c2)/n, pow((*err)->getConfusionLoss(1).at<float>(c1, c2)/n,2) );
			}
			this->confusionLoss[1].at<Vec2f>(c1, this->confusionLoss[1].cols-1) += Vec2f(1,1);
		}
		
		double m;
		int numOfClasses = (*err)->getConfusionLoss(1).rows;
		for(int c=0; c<numOfClasses; c++){
			this->stats[0].at(c) += (*err)->getStats(0).at(c) * 1./numOfClasses;
			this->stats[1].at(c) += (*err)->getStats(1).at(c) * 1./numOfClasses;
		}
    }

    // calculate mean-error
    this->errAbs[0] = 1.0/cvError.size() * sumErrAbs[0];
    this->errAbs[1] = 1.0/cvError.size() * sumErrAbs[1];
    this->errLoss[0] = 1.0/cvError.size() * sumErrLoss[0];
    this->errLoss[1] = 1.0/cvError.size() * sumErrLoss[1];
    this->margin[0] = 1.0/cvError.size() * sumMargin[0];
    this->margin[1] = 1.0/cvError.size() * sumMargin[1];

    // calculate standard-deviation
    this->errAbsStd[0] = sqrt(abs(1.0/(cvError.size()-1)*(sum2ErrAbs[0] - 1.0/cvError.size() * pow(sumErrAbs[0],2))));
    this->errAbsStd[1] = sqrt(abs(1.0/(cvError.size()-1)*(sum2ErrAbs[1] - 1.0/cvError.size() * pow(sumErrAbs[1],2))));
    this->errLossStd[0] = sqrt(abs(1.0/(cvError.size()-1)*(sum2ErrLoss[0] - 1.0/cvError.size() * pow(sumErrLoss[0],2))));
    this->errLossStd[1] = sqrt(abs(1.0/(cvError.size()-1)*(sum2ErrLoss[1] - 1.0/cvError.size() * pow(sumErrLoss[1],2))));
    this->marginStd[0] = sqrt(abs(1.0/(cvError.size()-1)*(sum2Margin[0] - 1.0/cvError.size() * pow(sumMargin[0],2))));
    this->marginStd[1] = sqrt(abs(1.0/(cvError.size()-1)*(sum2Margin[1] - 1.0/cvError.size() * pow(sumMargin[1],2))));

}

// destructor
ClassificationError::~ClassificationError(void){
  
  // nothing to do

}

/*
// estimate individual error of one fold using trival uniform-classificator
void ClassificationError::estimate(list< vector<Mat> >& refData, int fold, int numberOfFolds, bool horizontalFolds){

  // allocate and init confusion matrices
  double numberOfClasses = refData.front().size();
  this->confusionAbs[0] = Mat::zeros(numberOfClasses, numberOfClasses+1, CV_32FC1);
  this->confusionAbs[1] = Mat::zeros(numberOfClasses, numberOfClasses+1, CV_32FC1);
  this->confusionLoss[0] = Mat::zeros(numberOfClasses, numberOfClasses+1, CV_32FC1);
  this->confusionLoss[1] = Mat::zeros(numberOfClasses, numberOfClasses+1, CV_32FC1);

  // some temporary variables
  Mat tmp;
  double count[2] = {0, 0};
  double e, curProb_ref, curProb_est, maxProb_ref, maxProb_est, sndMaxProb_ref, sndMaxProb_est;
  int bestClass_ref, bestClass_est;
  // loop through images
  int i = 0, isTestData;
  for(list< vector<Mat> >::iterator ref=refData.begin(); ref != refData.end(); ref++, i++){

      vector<Mat> est;
      Mat sum = Mat::zeros( (*ref).front().rows, (*ref).front().cols, CV_32FC1 );

      for(int c=0; c<(*ref).size(); c++){
		  Mat tmp = Mat( (*ref).front().rows, (*ref).front().cols, CV_32FC1 );

		  randu( tmp, 1, 100 );
		  sum += tmp;

		  est.push_back(tmp);
      }

      for(int c=0; c<(*ref).size(); c++){
		divide(est.at(c), sum, est.at(c));
      }

      // if there are less folds than images; each fold consists of a subset of images (if not it consists of image parts)
      if (numberOfFolds <= refData.size()){
		  // check if current image is part of test- or training set
		  if ((i % numberOfFolds) == fold)
			  isTestData = 1;
		  else
			  isTestData = 0;
      }
      // loop through all pixel
      for(int y=0; y<(*ref).at(0).rows; y++){
		  for(int x=0; x<(*ref).at(0).cols; x++){

			  // if there are more folds than images, each fold consists of image parts
			  if (numberOfFolds > refData.size()){
				  if (horizontalFolds){
					  // check if current pixel belongs to test- or trainings data
					  int width = (*ref).at(0).cols/numberOfFolds;
					  if ((x >= fold*width) and (x < (fold+1)*width))
						isTestData = 1;
					  else
						isTestData = 0;
				  }else{
					  // check if current pixel belongs to test- or trainings data
					  int height = (*ref).at(0).rows/numberOfFolds;
					  if ((y >= fold*height) and (y < (fold+1)*height))
						isTestData = 1;
					  else
						isTestData = 0;
				  }
			  }

			  e = 0;
			  maxProb_ref = maxProb_est = sndMaxProb_ref = sndMaxProb_est = -1;
			  bestClass_ref = bestClass_est = -1;
			  // loop through classes
			  // seek maximum class-probability in reference and estimation data
			  for(int c=0; c<(*ref).size(); c++){
				  // get probability of this class within the reference data
				  curProb_ref = (*ref).at(c).at<float>(y, x);
				  // get probability of this class within the estimation
				  curProb_est = (est).at(c).at<float>(y, x);
				  // seek probability of best class in reference data
				  if (maxProb_ref < curProb_ref){
					  sndMaxProb_ref = maxProb_ref;
					  maxProb_ref = curProb_ref;
					  bestClass_ref = c;
				  }
				  // seek probability of snd best class in reference data
				  if ( (sndMaxProb_ref < curProb_ref) and (maxProb_ref > curProb_ref) ){
					  sndMaxProb_ref = curProb_ref;
				  }
				  // seek probability of best class in estimation data
				  if (maxProb_est < curProb_est){
					  sndMaxProb_est = maxProb_est;
					  maxProb_est = curProb_est;
					  bestClass_est = c;
				  }
				  // seek probability of snd best class in estimation data
				  if ( (sndMaxProb_est < curProb_est) and (maxProb_est > curProb_est) ){
					  sndMaxProb_est = curProb_est;
				  }
				  // store absolute difference
				  e += abs(curProb_ref - curProb_est);
			  }
			  
			  // if no class was defined in reference data, this pixel wont count in accuracy assessment
			  if (bestClass_ref<0)
				continue;
			  
			  // add absolute difference of current pixel to overall sum
			  this->errAbs[isTestData] += e/numberOfClasses;
			  // check if another class as the true class was estimated, and if so, denote it as error
			  if (bestClass_ref != bestClass_est){
			  this->errLoss[isTestData]++;
			  }
			  // add margin
			  this->margin[isTestData] += (maxProb_est - sndMaxProb_est);

			  // update 0-1-loss confusion matrix
			  this->confusionLoss[isTestData].at<float>(bestClass_ref, bestClass_est)++;
			  // save number class-samples in last column
			  this->confusionLoss[isTestData].at<float>(bestClass_ref, this->confusionLoss[isTestData].cols-1)++;

			  // update absolute difference confusion matrix
			  for(int c=0; c<(*ref).size(); c++){
				  // add certainty to about class c to confusion matrix
				  this->confusionAbs[isTestData].at<float>(bestClass_ref, c) += (est).at(c).at<float>(y, x);
				  // save "number" class-samples in last column (its "number" because it will be 1 when summed over all classes)
				  this->confusionAbs[isTestData].at<float>(bestClass_ref, this->confusionAbs[isTestData].cols-1) += (est).at(c).at<float>(y, x);
			  }
			  // denote number of samples in test and train set
			  count[isTestData]++;
		  }
      }

  }
  // normalize errors
  this->errAbs[0] /= count[0];
  this->errLoss[0] /= count[0];
  this->margin[0] /= count[0];
  this->errAbs[1] /= count[1];
  this->errLoss[1] /= count[1];
  this->margin[1] /= count[1];

}

// estimate individual error of one fold using trival maxPrior-classificator
void ClassificationError::estimate(list< vector<Mat> >& refData, Mat prior, int fold, int numberOfFolds, bool horizontalFolds){

  // allocate and init confusion matrices
  double numberOfClasses = refData.front().size();
  this->confusionAbs[0] = Mat::zeros(numberOfClasses, numberOfClasses+1, CV_32FC1);
  this->confusionAbs[1] = Mat::zeros(numberOfClasses, numberOfClasses+1, CV_32FC1);
  this->confusionLoss[0] = Mat::zeros(numberOfClasses, numberOfClasses+1, CV_32FC1);
  this->confusionLoss[1] = Mat::zeros(numberOfClasses, numberOfClasses+1, CV_32FC1);
  
  // some temporary variables
  Mat tmp;
  double count[2] = {0, 0};
  double e, curProb_ref, curProb_est, maxProb_ref, maxProb_est, sndMaxProb_ref, sndMaxProb_est;
  int bestClass_ref, bestClass_est;
  
  // loop through images
  int i = 0, isTestData;
  for(list< vector<Mat> >::iterator ref=refData.begin(); ref != refData.end(); ref++, i++){
   
      // if there are less folds than images; each fold consists of a subset of images (if not it consists of image parts)
      if (numberOfFolds <= refData.size()){
		  // check if current image is part of test- or training set
		  if ((i % numberOfFolds) == fold)
			  isTestData = 1;
		  else
			  isTestData = 0;
      }
      // loop through all pixel
      for(int y=0; y<(*ref).at(0).rows; y++){
		  for(int x=0; x<(*ref).at(0).cols; x++){

			  // if there are more folds than images, each fold consists of image parts
			  if (numberOfFolds > refData.size()){
				  if (horizontalFolds){
					  // check if current pixel belongs to test- or trainings data
					  int width = (*ref).at(0).cols/numberOfFolds;
					  if ((x >= fold*width) and (x < (fold+1)*width))
						isTestData = 1;
					  else
						isTestData = 0;
				  }else{
					  // check if current pixel belongs to test- or trainings data
					  int height = (*ref).at(0).rows/numberOfFolds;
					  if ((y >= fold*height) and (y < (fold+1)*height))
						isTestData = 1;
					  else
						isTestData = 0;
				  }
			  }

			  e = 0;
			  maxProb_ref = maxProb_est = sndMaxProb_ref = sndMaxProb_est = -1;
			  bestClass_ref = bestClass_est = -1;
			  // loop through classes
			  // seek maximum class-probability in reference and estimation data
			  for(int c=0; c<(*ref).size(); c++){
				  // get probability of this class within the reference data
				  curProb_ref = (*ref).at(c).at<float>(y, x);
				  // get probability of this class within the estimation
				  curProb_est = prior.at<float>(c);
				  // seek probability of best class in reference data
				  if (maxProb_ref < curProb_ref){
					  sndMaxProb_ref = maxProb_ref;
					  maxProb_ref = curProb_ref;
					  bestClass_ref = c;
				  }
				  // seek probability of snd best class in reference data
				  if ( (sndMaxProb_ref < curProb_ref) and (maxProb_ref > curProb_ref) ){
					  sndMaxProb_ref = curProb_ref;
				  }
				  // seek probability of best class in estimation data
				  if (maxProb_est < curProb_est){
					  sndMaxProb_est = maxProb_est;
					  maxProb_est = curProb_est;
					  bestClass_est = c;
				  }
				  // seek probability of snd best class in estimation data
				  if ( (sndMaxProb_est < curProb_est) and (maxProb_est > curProb_est) ){
					  sndMaxProb_est = curProb_est;
				  }
				  // store absolute difference
				  e += abs(curProb_ref - curProb_est);
			  }
			  
			  // if no class was defined in reference data, this pixel wont count in accuracy assessment
			  if (bestClass_ref<0)
				continue;
			  
			  // add absolute difference of current pixel to overall sum
			  this->errAbs[isTestData] += e/numberOfClasses;
			  // check if another class as the true class was estimated, and if so, denote it as error
			  if (bestClass_ref != bestClass_est){
				this->errLoss[isTestData]++;
			  }
			  // add margin
			  this->margin[isTestData] += (maxProb_est - sndMaxProb_est);

			  // update 0-1-loss confusion matrix
			  this->confusionLoss[isTestData].at<float>(bestClass_ref, bestClass_est)++;
			  // save number class-samples in last column
			  this->confusionLoss[isTestData].at<float>(bestClass_ref, this->confusionLoss[isTestData].cols-1)++;

			  // update absolute difference confusion matrix
			  for(int c=0; c<(*ref).size(); c++){
				  // add certainty to about class c to confusion matrix
				  this->confusionAbs[isTestData].at<float>(bestClass_ref, c) += prior.at<float>(c);
				  // save "number" class-samples in last column (its "number" because it will be 1 when summed over all classes)
				  this->confusionAbs[isTestData].at<float>(bestClass_ref, this->confusionAbs[isTestData].cols-1) += prior.at<float>(c);
			  }
			  // denote number of samples in test and train set
			  count[isTestData]++;
		  }
      }   
  }
  // normalize errors
  this->errAbs[0] /= count[0];
  this->errLoss[0] /= count[0];
  this->margin[0] /= count[0];
  this->errAbs[1] /= count[1];
  this->errLoss[1] /= count[1];
  this->margin[1] /= count[1];
  
}*/

// print error estimate
void ClassificationError::print(void){

  double n, v;
  Vec2f oa;
  
  cout << endl << "*** Accuracy Assessment ***" << endl << endl;
  cout << "*** on training data ***" << endl;
  cout << " >> Loss:\t" << this->errLoss[0] << "\t+-\t" << this->errLossStd[0] << endl;
  cout << " >> Error:\t" << this->errAbs[0] << "\t+-\t" << this->errAbsStd[0] << endl;
  cout << " >> Margin:\t" << this->margin[0] << "\t+-\t" << this->marginStd[0] << endl;
  if (CV_MAT_CN(this->confusionLoss[0].type()) == 1){
      oa.val[0] = oa.val[1] = 0;
      cout << " >> Confusion (Loss):" << endl;
      for(int c_ref=0; c_ref<this->confusionLoss[0].rows; c_ref++){
	  n = this->confusionLoss[0].at<float>(c_ref, this->confusionLoss[0].cols-1);
	  cout << "\t";
	  for(int c_est=0; c_est<this->confusionLoss[0].cols-1; c_est++){
	      if (n>0){
			cout << setw(10) << left << (this->confusionLoss[0].at<float>(c_ref, c_est)/n) << "\t";
		  if (c_ref == c_est){
		      oa.val[0] += this->confusionLoss[0].at<float>(c_ref, c_est)/n;
		      oa.val[1] += pow( this->confusionLoss[0].at<float>(c_ref, c_est)/n,2);
		  }
	      }else
		  cout << setw(10) << left << 0.0 << "\t";
	  }
	  cout << "( " << n << " )" << endl;
      }
      cout << " >> OA (Loss Training): " << oa.val[0]/this->confusionLoss[0].rows << "\t+-\t" << sqrt((oa.val[1] - pow(oa.val[0],2)/this->confusionLoss[0].rows)/(this->confusionLoss[0].rows-1)) << endl;
  }else{
      n = this->confusionLoss[0].at<Vec2f>(0, this->confusionLoss[0].cols-1).val[0];
      oa.val[0] = oa.val[1] = 0;
      cout << " >> Confusion (Loss) Mean:" << endl;
      for(int c_ref=0; c_ref<this->confusionLoss[0].rows; c_ref++){
		  cout << "\t";
		  for(int c_est=0; c_est<this->confusionLoss[0].cols-1; c_est++){
			  Vec2f cur = this->confusionLoss[0].at<Vec2f>(c_ref, c_est);
			  cout << setw(10) << left << 1./n * cur.val[0] << "\t";
			  if (c_ref == c_est){
				  oa.val[0] += 1./n * cur.val[0];
				  oa.val[1] += pow(1./n * cur.val[0],2);
			  }
		  }
		  cout << endl;
      }
      cout << " >> Confusion (Loss) StdDev:" << endl;
      for(int c_ref=0; c_ref<this->confusionLoss[0].rows; c_ref++){
	  cout << "\t";
	  for(int c_est=0; c_est<this->confusionLoss[0].cols-1; c_est++){
		  Vec2f cur = this->confusionLoss[0].at<Vec2f>(c_ref, c_est);
		  v = 1./(n-1)*(cur.val[1] - 1./n*pow(cur.val[0],2));
		  if (v>0)
		      cout << setw(10) << left << sqrt(v) << "\t";
		  else
		      cout << setw(10) << left << 0.0 << "\t";
	  }
	  cout << endl;
      }
      cout << " >> OA (Loss Training): " << oa.val[0]/this->confusionLoss[0].rows << "\t+-\t" << sqrt((oa.val[1] - pow(oa.val[0],2)/this->confusionLoss[0].rows)/(this->confusionLoss[0].rows-1)) << endl;
  }
  if (CV_MAT_CN(this->confusionAbs[0].type()) == 1){
      oa.val[0] = oa.val[1] = 0;
      cout << " >> Confusion (Error):" << endl;
      for(int c_ref=0; c_ref<this->confusionAbs[0].rows; c_ref++){
		  n = this->confusionAbs[0].at<float>(c_ref, this->confusionAbs[0].cols-1);
		  cout << "\t";
		  for(int c_est=0; c_est<this->confusionAbs[0].cols-1; c_est++){
			  if (n>0){
				  cout << setw(10) << left << this->confusionAbs[0].at<float>(c_ref, c_est)/n << "\t";
				  if (c_ref == c_est){
					  oa.val[0] += this->confusionAbs[0].at<float>(c_ref, c_est)/n;
					  oa.val[1] += pow(this->confusionAbs[0].at<float>(c_ref, c_est)/n,2);
				  }
			  }else
					cout << setw(10) << left << 0.0 << "\t";
		  }
		  cout << "( " << n << " )" << endl;
      }
      cout << " >> OA (Error Training): " << oa.val[0]/this->confusionAbs[0].rows << "\t+-\t" << sqrt((oa.val[1] - pow(oa.val[0],2)/this->confusionAbs[0].rows)/(this->confusionAbs[0].rows-1)) << endl;
  }else{
      n = this->confusionAbs[0].at<Vec2f>(0, this->confusionAbs[0].cols-1).val[0];
      oa.val[0] = oa.val[1] = 0;
      cout << " >> Confusion (Error) Mean:" << endl;
      for(int c_ref=0; c_ref<this->confusionAbs[0].rows; c_ref++){
		cout << "\t";
		for(int c_est=0; c_est<this->confusionAbs[0].cols-1; c_est++){
		  Vec2f cur = this->confusionAbs[0].at<Vec2f>(c_ref, c_est);
		  cout << setw(10) << left << 1./n * cur.val[0] << "\t";
		  if (c_ref == c_est){
		      oa.val[0] += 1./n * cur.val[0];
		      oa.val[1] += pow(1./n * cur.val[0],2);
		  }
		}
	  cout << endl;
      }
      cout << " >> Confusion (Error) StdDev:" << endl;
      for(int c_ref=0; c_ref<this->confusionAbs[0].rows; c_ref++){
		  cout << "\t";
		  for(int c_est=0; c_est<this->confusionAbs[0].cols-1; c_est++){
			  Vec2f cur = this->confusionAbs[0].at<Vec2f>(c_ref, c_est);
			  v = 1./(n-1)*(cur.val[1] - 1./n*pow(cur.val[0],2));
			  if (v>0)
				  cout << setw(10) << left << sqrt(v) << "\t";
			  else
				  cout << setw(10) << left << 0.0 << "\t";
		  }
		  cout << endl;
      }
      cout << " >> OA (Error Training): " << oa.val[0]/this->confusionAbs[0].rows << "\t+-\t" << sqrt((oa.val[1] - pow(oa.val[0],2)/this->confusionAbs[0].rows)/(this->confusionAbs[0].rows-1)) << endl;
  }
  cout << "True Positives TP" << endl;
  cout << "False Negatives FN" << endl;
  cout << "False Positives FP" << endl;
  cout << "True Negatives TN" << endl;
  cout << "Test outcome Positive TPos" << endl;
  cout << "Test outcome Negative TNeg" << endl;
  cout << "Test outcome Correct TCor" << endl;
  cout << "Test outcome Wrong TWro" << endl;
  cout << "Positive Predictive Value, precision PPV" << endl;
  cout << "False Discovery Rate FDR" << endl;
  cout << "False Ommision Rate FOR" << endl;
  cout << "Negative Predictive Value NPV" << endl;
  cout << "True Positive Rate, sensitivity, recall TPR" << endl;
  cout << "False Positive Rate, fall-out FPR" << endl;
  cout << "False Negative Rate FNR" << endl;
  cout << "True Negative Rate, specificity TNR" << endl;
  cout << "Positive Likelihood Ratio LR+" << endl;
  cout << "Negative Likelihood Ratio LR-" << endl;
  cout << "F-measure (beta=0.5) F0.5" << endl;
  cout << "F-measure (beta=1) F1" << endl;
  cout << "F-measure (beta=2) F2" << endl;
  cout << "G-measure G" << endl;
  cout << "Information Content IC" << endl;
  cout << "(Overall) Accuracy AC" << endl;
  cout << "Balanced Accuracy" << endl;
  cout << "Expected Accuracy" << endl;
  cout << "Matthews correlations coefficient MCC" << endl;
  cout << "Informedness I" << endl;
  cout << "Markedness M" << endl;
  cout << "kappa k" << endl;
  cout << "Diagnostics Odds Ratio DOR" << endl;
  cout << endl << endl;
  
  // TP FN FP TN
  for(int c=0; c < this->confusionAbs[1].rows; c++){
	  double Pos = this->stats[0].at(c).at<float>(0,0) + this->stats[0].at(c).at<float>(0,1);
	  double Neg = this->stats[0].at(c).at<float>(0,2) + this->stats[0].at(c).at<float>(0,3);
	  double N = Pos + Neg;
	  double Prev = Pos/N;
	  cout << endl << "Statistics for class " << c << endl;
	  cout << "Total Population N:\t" << N << endl;
	  cout << "Condition Positive Pos:\t" << Pos << endl;
	  cout << "Condition Negative Neg:\t" << Neg << endl;
	  cout << "Prevalence Prev:\t" << Prev << endl;
	  
	  Mat statistics(31, 11, CV_32FC1);
	  double AUC = 0;
	  for(int t=0; t<=10; t++){
		  double TP = this->stats[0].at(c).at<float>(t,0);
		  double FN = this->stats[0].at(c).at<float>(t,1);
		  double FP = this->stats[0].at(c).at<float>(t,2);
		  double TN = this->stats[0].at(c).at<float>(t,3);
		  statistics.at<float>(0, t) = TP;
		  statistics.at<float>(1, t) = FN;
		  statistics.at<float>(2, t) = FP;
		  statistics.at<float>(3, t) = TN;
		  // Level 0
		  double TPos = TP + FP;
		  double TNeg = TN + FN;
		  double TCor = TP + TN;
		  double TWro = FP + FN;
		  statistics.at<float>(4, t) = TPos;
		  statistics.at<float>(5, t) = TNeg;
		  statistics.at<float>(6, t) = TCor;
		  statistics.at<float>(7, t) = TWro;
		  // Level 1
		  double PPV = TP / TPos; if (TPos == 0) PPV = 0;
		  double FDR = FP / TPos; if (TPos == 0) FDR = 0;
		  double FOR = FN / TNeg; if (TNeg == 0) FOR = 0;
		  double NPV = TN / TNeg; if (TNeg == 0) NPV = 0;
		  double TPR = TP / Pos;
		  double FPR = FP / Neg;
		  double FNR = FN / Pos;
		  double TNR = TN / Neg;
		  statistics.at<float>(8, t) = PPV;
		  statistics.at<float>(9, t) = FDR;
		  statistics.at<float>(10, t) = FOR;
		  statistics.at<float>(11, t) = NPV;
		  statistics.at<float>(12, t) = TPR;
		  statistics.at<float>(13, t) = FPR;
		  statistics.at<float>(14, t) = FNR;
		  statistics.at<float>(15, t) = TNR;
		  // Level 2
		  double LRp = TPR / FPR; if (FPR == 0) LRp = DBL_MAX;
		  double LRm = FNR / TNR; if (TNR == 0) LRm = DBL_MAX;
		  double Fhalf = 1.25*PPV*TPR / (0.25*PPV + TPR);  if (TP == 0) Fhalf = 0;
		  double Fone = 2*PPV*TPR / (PPV + TPR);  if (TP == 0) Fone = 0;
		  double Ftwo = 5*PPV*TPR / (4*PPV + TPR);  if (TP == 0) Ftwo = 0;
		  double G = sqrt(PPV*TPR);
		  double IC = (PPV+TPR) / 2.;
		  double AC = TCor / N;
		  double BA = (TPR + TNR) / 2.;
		  double EA = ( Pos*TPos/N + Neg*TNeg/N) / N;
		  double MCC = (TP*TN - FP*FN) / sqrt(TPos*Pos*TNeg*Neg); if (TPos*TNeg == 0) MCC = 0;
		  double I = TPR + TNR - 1;
		  double M = PPV + NPV - 1;
		  statistics.at<float>(16, t) = LRp;
		  statistics.at<float>(17, t) = LRm;
		  statistics.at<float>(18, t) = Fhalf;
		  statistics.at<float>(19, t) = Fone;
		  statistics.at<float>(20, t) = Ftwo;
		  statistics.at<float>(21, t) = G;
		  statistics.at<float>(22, t) = IC;
		  statistics.at<float>(23, t) = AC;
		  statistics.at<float>(24, t) = BA;
		  statistics.at<float>(25, t) = EA;
		  statistics.at<float>(26, t) = MCC;
		  statistics.at<float>(27, t) = I;
		  statistics.at<float>(28, t) = M;
		  // Level 3
		  double k = (AC - EA) / (1 - EA);
		  double DOR = LRp / LRm; if (LRm == 0) DOR = DBL_MAX;
		  statistics.at<float>(29, t) = k;
		  statistics.at<float>(30, t) = DOR;
		  AUC += TPR*0.1;
	  }
	  cout << statistics << endl;
	  cout << "Area Under the Curve AUC:\t" << AUC << endl;
	  cout << "Discrimination D:\t" << (AUC-0.5) << endl;
  }
  
  cout << endl << "*** on test data ***" << endl;
  cout << " >> Loss:\t" << this->errLoss[1] << "\t+-\t" << this->errLossStd[1] << endl;
  cout << " >> Error:\t" << this->errAbs[1] << "\t+-\t" << this->errAbsStd[1] << endl;
  cout << " >> Margin:\t" << this->margin[1] << "\t+-\t" << this->marginStd[1] << endl;
  if (CV_MAT_CN(this->confusionLoss[1].type()) == 1){
      oa.val[0] = oa.val[1] = 0;
      cout << " >> Confusion (Loss):" << endl;
      for(int c_ref=0; c_ref<this->confusionLoss[1].rows; c_ref++){
		  n = this->confusionLoss[1].at<float>(c_ref, this->confusionLoss[1].cols-1);
		  cout << "\t";
		  for(int c_est=0; c_est<this->confusionLoss[1].cols-1; c_est++){
			  if (n>0){
				  cout << setw(10) << left << this->confusionLoss[1].at<float>(c_ref, c_est)/n << "\t";
				  if (c_ref == c_est){
					  oa.val[0] += this->confusionLoss[1].at<float>(c_ref, c_est)/n;
					  oa.val[1] += pow(this->confusionLoss[1].at<float>(c_ref, c_est)/n,2);
				  }
			  }else	
				  cout << setw(10) << left << 0.0 << "\t";
		  }
		  cout << "( " << n << " )" << endl;
      }
      cout << " >> OA (Loss Test): " << oa.val[0]/this->confusionLoss[1].rows << "\t+-\t" << sqrt((oa.val[1] - pow(oa.val[0],2)/this->confusionLoss[1].rows)/(this->confusionLoss[1].rows-1)) << endl;
  }else{
      n = this->confusionLoss[1].at<Vec2f>(0, this->confusionLoss[1].cols-1).val[0];
      oa.val[0] = oa.val[1] = 0;
      cout << " >> Confusion (Loss) Mean:" << endl;
      for(int c_ref=0; c_ref<this->confusionLoss[1].rows; c_ref++){
		  cout << "\t";
		  for(int c_est=0; c_est<this->confusionLoss[1].cols-1; c_est++){
			  Vec2f cur = this->confusionLoss[1].at<Vec2f>(c_ref, c_est);
			  cout << setw(10) << left << 1./n * cur.val[0] << "\t";
			  if (c_ref == c_est){
				  oa.val[0] += 1./n * cur.val[0];
				  oa.val[1] += pow(1./n * cur.val[0],2);
			  }
		  }
		  cout << endl;
      }
      cout << " >> Confusion (Loss) StdDev:" << endl;
      for(int c_ref=0; c_ref<this->confusionLoss[1].rows; c_ref++){
		  cout << "\t";
		  for(int c_est=0; c_est<this->confusionLoss[1].cols-1; c_est++){
			  Vec2f cur = this->confusionLoss[1].at<Vec2f>(c_ref, c_est);
			  v = 1./(n-1)*(cur.val[1] - 1./n*pow(cur.val[0],2));
			  if (v>0)
				  cout << setw(10) << left << sqrt(v) << "\t";
			  else
				  cout << setw(10) << left << 0.0 << "\t";
		  }
		  cout << endl;
      }  
      cout << " >> OA (Loss Test): " << oa.val[0]/this->confusionLoss[1].rows << "\t+-\t" << sqrt((oa.val[1] - pow(oa.val[0],2)/this->confusionLoss[1].rows)/(this->confusionLoss[1].rows-1)) << endl;
  }
  if (CV_MAT_CN(this->confusionAbs[1].type()) == 1){
      oa.val[0] = oa.val[1] = 0;
      cout << " >> Confusion (Error):" << endl;
      for(int c_ref=0; c_ref<this->confusionAbs[1].rows; c_ref++){
		  n = this->confusionAbs[1].at<float>(c_ref, this->confusionAbs[1].cols-1);
		  cout << "\t";
		  for(int c_est=0; c_est<this->confusionAbs[1].cols-1; c_est++){
			  if (n>0){
				cout << setw(10) << left << this->confusionAbs[1].at<float>(c_ref, c_est)/n << "\t";
			  if (c_ref == c_est){
				  oa.val[0] += this->confusionAbs[1].at<float>(c_ref, c_est)/n;
				  oa.val[1] += pow(this->confusionAbs[1].at<float>(c_ref, c_est)/n,2);
			  }
			  }else
				  cout << setw(10) << left << 0.0 << "\t";
		  }
		  cout << "( " << n << " )" << endl;
      }
      cout << " >> OA (Error Test): " << oa.val[0]/this->confusionAbs[1].rows << "\t+-\t" << sqrt((oa.val[1] - pow(oa.val[0],2)/this->confusionAbs[1].rows)/(this->confusionAbs[1].rows-1)) << endl;
  }else{
      n = this->confusionAbs[1].at<Vec2f>(0, this->confusionAbs[1].cols-1).val[0];
      oa.val[0] = oa.val[1] = 0;
      cout << " >> Confusion (Error) Mean:" << endl;
      for(int c_ref=0; c_ref<this->confusionAbs[1].rows; c_ref++){
		  cout << "\t";
		  for(int c_est=0; c_est<this->confusionAbs[1].cols-1; c_est++){
			  Vec2f cur = this->confusionAbs[1].at<Vec2f>(c_ref, c_est);
			  cout << setw(10) << left << 1./n * cur.val[0] << "\t";
			  if (c_ref == c_est){
				  oa.val[0] += 1./n * cur.val[0];
				  oa.val[1] += pow(1./n * cur.val[0],2);
			  }
		  }
		  cout << endl;
      }
      cout << " >> Confusion (Error) StdDev:" << endl;
      for(int c_ref=0; c_ref<this->confusionAbs[1].rows; c_ref++){
		  cout << "\t";
		  for(int c_est=0; c_est<this->confusionAbs[1].cols-1; c_est++){
			  Vec2f cur = this->confusionAbs[1].at<Vec2f>(c_ref, c_est);
			  v = 1./(n-1)*(cur.val[1] - 1./n*pow(cur.val[0],2));
			  if (v>0)
				  cout << setw(10) << left << sqrt(v) << "\t";
			  else
				  cout << setw(10) << left << 0.0 << "\t";
		  }
		  cout << endl;
      }
      cout << " >> OA (Error Test): " << oa.val[0]/this->confusionAbs[1].rows << "\t+-\t" << sqrt((oa.val[1] - pow(oa.val[0],2)/this->confusionAbs[1].rows)/(this->confusionAbs[1].rows-1)) << endl;
  }
  cout << "True Positives TP" << endl;
  cout << "False Negatives FN" << endl;
  cout << "False Positives FP" << endl;
  cout << "True Negatives TN" << endl;
  cout << "Test outcome Positive TPos" << endl;
  cout << "Test outcome Negative TNeg" << endl;
  cout << "Test outcome Correct TCor" << endl;
  cout << "Test outcome Wrong TWro" << endl;
  cout << "Positive Predictive Value, precision PPV" << endl;
  cout << "False Discovery Rate FDR" << endl;
  cout << "False Ommision Rate FOR" << endl;
  cout << "Negative Predictive Value NPV" << endl;
  cout << "True Positive Rate, sensitivity, recall TPR" << endl;
  cout << "False Positive Rate, fall-out FPR" << endl;
  cout << "False Negative Rate FNR" << endl;
  cout << "True Negative Rate, specificity TNR" << endl;
  cout << "Positive Likelihood Ratio LR+" << endl;
  cout << "Negative Likelihood Ratio LR-" << endl;
  cout << "F-measure (beta=0.5) F0.5" << endl;
  cout << "F-measure (beta=1) F1" << endl;
  cout << "F-measure (beta=2) F2" << endl;
  cout << "G-measure G" << endl;
  cout << "Information Content IC" << endl;
  cout << "(Overall) Accuracy AC" << endl;
  cout << "Balanced Accuracy" << endl;
  cout << "Expected Accuracy" << endl;
  cout << "Matthews correlations coefficient MCC" << endl;
  cout << "Informedness I" << endl;
  cout << "Markedness M" << endl;
  cout << "kappa k" << endl;
  cout << "Diagnostics Odds Ratio DOR" << endl;
  cout << endl << endl;
  
  // TP FN FP TN
  for(int c=0; c < this->confusionAbs[1].rows; c++){
	  double Pos = this->stats[1].at(c).at<float>(0,0) + this->stats[1].at(c).at<float>(0,1);
	  double Neg = this->stats[1].at(c).at<float>(0,2) + this->stats[1].at(c).at<float>(0,3);
	  double N = Pos + Neg;
	  double Prev = Pos/N;
	  cout << endl << "Statistics for class " << c << endl;
	  cout << "Total Population N:\t" << N << endl;
	  cout << "Condition Positive Pos:\t" << Pos << endl;
	  cout << "Condition Negative Neg:\t" << Neg << endl;
	  cout << "Prevalence Prev:\t" << Prev << endl;
	  
	  Mat statistics(31, 11, CV_32FC1);
	  double AUC = 0;
	  for(int t=0; t<=10; t++){
		  double TP = this->stats[1].at(c).at<float>(t,0);
		  double FN = this->stats[1].at(c).at<float>(t,1);
		  double FP = this->stats[1].at(c).at<float>(t,2);
		  double TN = this->stats[1].at(c).at<float>(t,3);
		  statistics.at<float>(0, t) = TP;
		  statistics.at<float>(1, t) = FN;
		  statistics.at<float>(2, t) = FP;
		  statistics.at<float>(3, t) = TN;
		  // Level 0
		  double TPos = TP + FP;
		  double TNeg = TN + FN;
		  double TCor = TP + TN;
		  double TWro = FP + FN;
		  statistics.at<float>(4, t) = TPos;
		  statistics.at<float>(5, t) = TNeg;
		  statistics.at<float>(6, t) = TCor;
		  statistics.at<float>(7, t) = TWro;
		  // Level 1
		  double PPV = TP / TPos; if (TPos == 0) PPV = 1;
		  double FDR = FP / TPos; if (TPos == 0) FDR = 1;
		  double FOR = FN / TNeg; if (TNeg == 0) FOR = 1;
		  double NPV = TN / TNeg; if (TNeg == 0) NPV = 1;
		  double TPR = TP / Pos;
		  double FPR = FP / Neg;
		  double FNR = FN / Pos;
		  double TNR = TN / Neg;
		  statistics.at<float>(8, t) = PPV;
		  statistics.at<float>(9, t) = FDR;
		  statistics.at<float>(10, t) = FOR;
		  statistics.at<float>(11, t) = NPV;
		  statistics.at<float>(12, t) = TPR;
		  statistics.at<float>(13, t) = FPR;
		  statistics.at<float>(14, t) = FNR;
		  statistics.at<float>(15, t) = TNR;
		  // Level 2
		  double LRp = TPR / FPR; if (FPR == 0) LRp = DBL_MAX;
		  double LRm = FNR / TNR; if (TNR == 0) LRm = DBL_MAX;
		  double Fhalf = 1.25*PPV*TPR / (0.25*PPV + TPR);  if (TP == 0) Fhalf = 1;
		  double Fone = 2*PPV*TPR / (PPV + TPR);  if (TP == 0) Fone = 1;
		  double Ftwo = 5*PPV*TPR / (4*PPV + TPR);  if (TP == 0) Ftwo = 1;
		  double G = sqrt(PPV*TPR);
		  double IC = (PPV+TPR) / 2.;
		  double AC = TCor / N;
		  double BA = (TPR + TNR) / 2.;
		  double EA = ( Pos*TPos/N + Neg*TNeg/N) / N;
		  double MCC = (TP*TN - FP*FN) / sqrt(TPos*Pos*TNeg*Neg); if (TPos*TNeg == 0) MCC = DBL_MAX;
		  double I = TPR + TNR - 1;
		  double M = PPV + NPV - 1;
		  statistics.at<float>(16, t) = LRp;
		  statistics.at<float>(17, t) = LRm;
		  statistics.at<float>(18, t) = Fhalf;
		  statistics.at<float>(19, t) = Fone;
		  statistics.at<float>(20, t) = Ftwo;
		  statistics.at<float>(21, t) = G;
		  statistics.at<float>(22, t) = IC;
		  statistics.at<float>(23, t) = AC;
		  statistics.at<float>(24, t) = BA;
		  statistics.at<float>(25, t) = EA;
		  statistics.at<float>(26, t) = MCC;
		  statistics.at<float>(27, t) = I;
		  statistics.at<float>(28, t) = M;
		  // Level 3
		  double k = (AC - EA) / (1 - EA);
		  double DOR = LRp / LRm; if (LRm == 0) DOR = DBL_MAX;
		  statistics.at<float>(29, t) = k;
		  statistics.at<float>(30, t) = DOR;
		  AUC += TPR*0.1;
	  }
	  cout << statistics << endl;
	  cout << "Area Under the Curve AUC:\t" << AUC << endl;
	  cout << "Discrimination D:\t" << (AUC-0.5) << endl;
  }
}

double ClassificationError::getErrAbs(int d){

  return this->errAbs[d];
  
}

double ClassificationError::getErrLoss(int d){

  return this->errLoss[d];
  
}

double ClassificationError::getMargin(int d){

  return this->margin[d];
  
}

Mat ClassificationError::getConfusionAbs(int d){

  return this->confusionAbs[d];

}

Mat ClassificationError::getConfusionLoss(int d){

  return this->confusionLoss[d];

}

vector<Mat> ClassificationError::getStats(int d){

  return this->stats[d];

}
