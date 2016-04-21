#ifndef CLASSIFICATIONERROR_H
#define CLASSIFICATIONERROR_H

#include <iostream>
#include <iomanip>
#include <list>
#include <vector>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

class ClassificationError{
  
  public:
    //! constructor
    ClassificationError(void);
    //! constructor
    ClassificationError(vector< vector<Mat> >& reference, vector< vector<Mat> >& estimate, int fold, vector<Mat>& foldID);
    //! constructor
    ClassificationError(vector< ClassificationError*>& cvError);
    //! destructor
    ~ClassificationError(void);

    //! print results
    void print(void);
    
    //! get absulte difference between in- and output (@param d specifying training or test-set)
    double getErrAbs(int d);
    //! get zero-one-loss (@param d specifying training or test-set)
    double getErrLoss(int d);
    //! get difference between best- and second best (@param d specifying training or test-set)
    double getMargin(int d);
    //! get confusion matrix based on absolute differences (@param d specifying training or test-set)
    Mat getConfusionAbs(int d);
    //! get confusion matrix based on 0-1-loss (@param d specifying training or test-set)
    Mat getConfusionLoss(int d);
    //! get error statistics
    vector<Mat> getStats(int d);
    
  private:
    //! average absolute difference on training and test-set
    double errAbs[2];
    //! average zero-one-loss on training and test-set
    double errLoss[2];
    //! average difference between best and second best on training and test-set
    double margin[2];
    
    //! std absolute difference on training and test-set
    double errAbsStd[2];
    //! std zero-one-loss on training and test-set
    double errLossStd[2];
    //! std difference between best and second best on training and test-set
    double marginStd[2];

    //! confusion matrix based on absolute differences
    Mat confusionAbs[2];
    //! confusion matrix based on 0-1-loss
    Mat confusionLoss[2];
    //! error statistics
    vector<Mat> stats[2];

};

#endif
