#include "Evaluation.h"

template <> void Evaluation<PRFDataPoint*>::imagePartsAsFold(vector<PRFDataPoint*>& data, bool horizontalFolds, vector<ImgData*>* imageData){

    Mat id;
            
    // create images, where in each position the corresponding fold-id is noted
    for(vector<ImgData*>::iterator img = imageData->begin(); img != imageData->end(); img++){

		int height = (*img)->getSize().height;
		int width = (*img)->getSize().width;

		// create ID-image
		id = Mat::zeros(height, width, CV_8UC1) + foldData.size()-1;
		if (horizontalFolds){
			// calculate fold-width for current data-image
			int fwidth= width/foldData.size();
			// set fold id
			for(int f=0; f<foldData.size()-1; f++){
				Mat roi(id, Rect(f*fwidth, 0, fwidth, height));
				roi.setTo(f);
			}
			this->foldID.push_back(id);
		}else{
			// calculate fold-height for current data-image
			int fheight = height/foldData.size();
			// set fold id
			for(int f=0; f<foldData.size()-1; f++){
				Mat roi(id, Rect(0, f*fheight, width, fheight));
				roi.setTo(f);
			}
			this->foldID.push_back(id);
		}
    }
    // check for each data-point to which fold it belongs
    for(vector< PRFDataPoint* >::iterator dp = data.begin(); dp != data.end(); dp++){

		foldData.at((int)(foldID.at((*dp)->getPosition(2)).at<uchar>((*dp)->getPosition(1), (*dp)->getPosition(0)))).push_back(*dp);
	}
}
  
template <> void Evaluation<PRFDataPoint*>::imagesAsFold(vector<PRFDataPoint*>& data, vector<ImgData*>* imageData){

    // if there are enough images to fill all folds with at least one image, then:
    // use modulo-operation to decide which image belongs to which fold
    for(vector< PRFDataPoint* >::iterator dp = data.begin(); dp != data.end(); dp++){
		foldData.at( (*dp)->getPosition(2) % foldData.size() ).push_back(*dp);
    }
    for(int img = 0; img < imageData->size(); img++){
		Mat id(1,1, CV_8UC1, (img % foldData.size()));
		foldID.push_back(id);
	}
}

// generate folds: divide data-points into a given number of folds
template <> void Evaluation<PRFDataPoint*>::generateFolds(vector<PRFDataPoint*>& data, int type, bool horizontalFolds, vector<ImgData*>* imageData){

  for(vector< vector< PRFDataPoint* > >::iterator fold = foldData.begin(); fold != foldData.end(); fold++){
	  fold->clear();
  }

  int numberOfImages = imageData->size();
  // check if a fold consists of several images, or of image parts
  // if there are more folds than images ==> divide images into parts
  if (foldData.size() > numberOfImages) type = 0;

  switch(type){
	case 1: imagesAsFold(data, imageData); break;
	default: imagePartsAsFold(data, horizontalFolds, imageData); break;
  }
}

template <> void Evaluation<PRFDataPoint*>::forceTrainingPrior(vector<PRFDataPoint*>& trainSet, Mat prior, double trainSetSize){

    int numberOfClasses = max(prior.cols, prior.rows);

    Mat trueFreq = Mat::zeros(1,numberOfClasses,CV_32FC1);
    for(vector<PRFDataPoint*>::iterator dp = trainSet.begin(); dp != trainSet.end(); dp++){
		for (int c=0; c<numberOfClasses; c++){
			trueFreq.at<float>(c) += (*dp)->getLabel(c);
		}
    }
    
    Mat trainPrior = Mat(1, numberOfClasses, CV_32FC1);
	for(int c=0; c<numberOfClasses; c++){
		if (prior.at<float>(c)>0)
			trainPrior.at<float>(c) = prior.at<float>(c);
		else
			trainPrior.at<float>(c) = trueFreq.at<float>(c) / sum(trueFreq).val[0];
	}
	if (trainSetSize == 0)
		trainSetSize = trainSet.size();
    
    // calculate number of class instances to get training set size with this prior
    double trainClassSize[numberOfClasses];
    double s = 1;
    for(int c=0; c<numberOfClasses; c++){
		double num = round(trainSetSize*trainPrior.at<float>(c));
		if (num > trueFreq.at<float>(c)){
			cout << "WARNING: too many instances of class " << c << endl;
			cout << "		--> Changing size of training set" << endl;
			if (s < trueFreq.at<float>(c)/num)
			  s = trueFreq.at<float>(c)/num;
		}
		trainClassSize[c] = num;
    }
    
    for(int c=0; c<numberOfClasses; c++){
		trainClassSize[c] = round(s*trainClassSize[c]);
    }
    
    // shuffle training data
    random_shuffle( trainSet.begin(), trainSet.end() );
    double curFreq[numberOfClasses];
    for(int c=0; c<numberOfClasses; c++)
		curFreq[c] = 0;
    
    vector<PRFDataPoint*> curTrainSet;
    double diff = trainSetSize;
    double eps = trainSetSize*0.001;
    // loop through the data and select data points
    int d=0;
    for(vector<PRFDataPoint*>::iterator dp = trainSet.begin(); dp != trainSet.end(); dp++, d++){
		// check if there are already enough instanced of this class within the new training set
		bool doNotAddIt = false;
		for(int c=0; c<numberOfClasses; c++){
			if (curFreq[c] + (*dp)->getLabel(c) > trainClassSize[c])
			doNotAddIt = true;
		}
		// if so, do not add it and continue
		if (doNotAddIt)
		  continue;
		// else: add it
		curTrainSet.push_back(*dp);
		// get size of current subset
		for (int c=0; c<numberOfClasses; c++){
			curFreq[c] += (*dp)->getLabel(c);
		}
		// check if size of current subset is close to final size
		diff = abs(curTrainSet.size() - trainSetSize);
		if ( diff < eps )
			break;
    }
    trainSet.swap(curTrainSet);
    
    for(int c=0; c<numberOfClasses; c++){
		if (abs(trainPrior.at<float>(c) - curFreq[c]/trainSet.size()) > 0.001 ){
			cout << "WARNING: Not possible to force specified prior! Continue?" << endl;
			cout << "\t\tClass " << c << ":\tDesired: " << trainPrior.at<float>(c) << "\t <--> \t" << (curFreq[c]/trainSet.size()) << endl;
	// 	    cin.get();
			break;
		}
    }
    
    cout << "Desired size: " << trainSetSize << endl;
    cout << "Obtained size:" << trainSet.size() << endl;
    cout << "Desired frequency: ";
    for(int c=0; c<numberOfClasses; c++)
      cout << trainClassSize[c] << "\t";
    cout << endl;
    cout << "Obtained frequency: ";
    for(int c=0; c<numberOfClasses; c++)
      cout << curFreq[c] << "\t";
    cout << endl;
    cout << "Desired prior: ";
    for(int c=0; c<numberOfClasses; c++)
      cout << trainPrior.at<float>(c) << "\t";
    cout << endl;
    cout << "Obtained prior: ";
    for(int c=0; c<numberOfClasses; c++)
      cout << curFreq[c]/trainSet.size() << "\t";
    cout << endl;
//     cin.get();
    
}

template <> void Evaluation<PRFDataPoint*>::getTrainData(vector<PRFDataPoint*>& data, Mat prior, double trainSetSize){

	for(int f = 0; f < foldData.size(); f++){
		if (f != this->currentFold){
			for(typename vector<PRFDataPoint*>::iterator dp = foldData.at(f).begin(); dp != foldData.at(f).end(); dp++){
				data.push_back(*dp);
			}
		}
	}
	
	forceTrainingPrior(data, prior, trainSetSize);
	
}

template <> Evaluation<PRFDataPoint*>::Evaluation(vector<PRFDataPoint*>& data, int numberOfFolds, int type, int answerLevel, bool horizontalFolds, vector<ImgData*>* imageData){

  this->answerLevel = answerLevel;

  this->currentFold = 0;
  this->numberOfFolds = numberOfFolds;
  this->error.resize(numberOfFolds);
  this->refError.resize(2);
  for(vector< vector< ClassificationError*> >::iterator ref = this->refError.begin(); ref != this->refError.end(); ref++){
	  ref->resize(numberOfFolds);
  }
  foldData.resize(numberOfFolds);

  if (data.size()>0){
	generateFolds(data, type, horizontalFolds, imageData);
   }
}
