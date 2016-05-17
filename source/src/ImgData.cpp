/**
* @file ImgData.cpp
* @brief This class is container of image data
* @author Ronny Haensch
*/

#include "ImgData.h"

// the default constructor
ImgData::ImgData(void){

  this->level = -1;
  this->path = "";
  this->size = Size(0,0);
  
}

// constructor
ImgData::ImgData(string path, int level){
  
  this->level = level;
  this->path = path;
  this->size = Size(0,0);
  
}

// destructor
ImgData::~ImgData(void){
	// nothing to do
}

// load data from disk
void ImgData::load(string path, int level){
  
  // define data path and level
  this->level = level;
  this->path = path;
  // and call actual load function
  this->load();

}

// load data from disk
void ImgData::load(void){
  
  Mat img;
  vector<Mat> data;  
  // chose correct loading function depending on the data level
  switch(this->level){
    // PolSAR images
    case 4:
      // this function is defined in Tools.hpp and might be buggy (in particular for untested number formats in .rat)
      loadRAT(this->path, data);
      this->polSAR.push_back(data);
      this->size = this->polSAR.at(0).at(0).size();
      break;
    // color images
    case 2: 
      // load image
      img = imread(this->path.c_str());
      if (!img.ptr()){
			cerr << "ERROR: cant load file: " << this->path << endl;
			exit(-1);
      }
      // transform to 32bit float
      img.convertTo(img, CV_32FC3);
      this->color.push_back(img);
      this->size = img.size();
      break;
    case 1:
      cerr << "\nDas hier noch implementieren! Hatte Ronny nicht, der Fall tritt aber doch auf.";
      exit(-1);
  }  
}

// projects data to all lower data-levels
void ImgData::project(void){
  // check if data was already projected, and if so: do nothin
  if (this->color.size() > 0)
    return;
  // chose projection function depending on data level
  switch(level){
    case 4: this->polSARProjection(); break;
  }
}

// projection of PolSAR data
void ImgData::polSARProjection(void){

    // to multi-pol: nothin to do
    
    bool dual = false;
    // to single-pol: use individual channels
    double minVal, maxVal;
    int i=0;
    Mat tmp;
    
    // to color: use channels from speckle reduced image
    // load speckle-reduced image; produced with RAT until speckle-reduction C/C++-Code is available
	vector< Mat > cov = polSAR.at(0);
    //loadRAT(this->path.substr(0,this->path.rfind('_')) + "_cov_reflee.rat", cov);
	//loadRAT(this->path, cov);

	// check if dual-pol or full-pol
    if (cov.size() == 9)
		dual = false;
    else
		dual = true;

    Mat blue, red, green, span, colImg_small, colImg;
    vector<Mat> channels;

    // get channel info
    if (!dual){
	  split(cov.at(0), channels);
	  red = channels.at(0).clone();
	  split(cov.at(4), channels);
	  green = channels.at(0).clone();
	  split(cov.at(8), channels);
	  blue = channels.at(0).clone();
	  //cout << "the data isn't dual pol" << endl;
    }else{
      // in the case of dual pol, use average of the first two channels as the third channel
      split(cov.at(0), channels);
	  red = channels.at(0).clone();
	  split(cov.at(3), channels);
	  green = channels.at(0).clone();
	  blue = (red + green) * 0.5;
    }

    // perform log-transform
	cout << this->path << endl;
	if (this->path.find("oph") != string::npos){
		//cout << "use log" << endl;
		red = red + 1;
		green = green + 1;
		blue = blue + 1;
		log(red, red);
		log(green, green);
		log(blue, blue);
	}else{
		cout << "use pow" << endl;
		(red, 0.5, red);
		pow(green, 0.5, green);
		pow(blue, 0.5, blue);
	}
	//cin.get();
    // thats new and wasnt used before!!
    // START
    threshold(red, red, 2.5*mean(red).val[0], 0, THRESH_TRUNC);
    threshold(green, green, 2.5*mean(green).val[0], 0, THRESH_TRUNC);
    threshold(blue, blue, 2.5*mean(blue).val[0], 0, THRESH_TRUNC);
    // END
    // get max to scale to [min,255]
    max(red, green, tmp);
    max(blue, tmp, tmp);
    //minMaxLoc(tmp, &minVal, &maxVal);
	minMaxLoc(red, &minVal, &maxVal);
    red = red * 255./maxVal;
	minMaxLoc(green, &minVal, &maxVal);
    green = green * 255./maxVal;
	minMaxLoc(blue, &minVal, &maxVal);
    blue = blue * 255./maxVal;
	channels.clear();
	channels.push_back(blue);
	channels.push_back(green);
	channels.push_back(red);
    // merge color planes to image
    merge(channels, colImg_small);
    // NOTE: SPECKLE-REDUCED IMAGE MIGHT BE SMALLER THAN ORIGINAL IMAGE ==> re-size!!
    resize(colImg_small, colImg, Size(this->polSAR.at(0).front().cols, this->polSAR.at(0).front().rows));
	
    // add
    this->color.push_back(colImg.clone());
	
	Mat gray_image;
	cvtColor(colImg, gray_image, CV_BGR2GRAY);
	this->grayscale.push_back(gray_image.clone());

	//imwrite("color.png", colImg);
	//imwrite("gray.png", gray_image);
	//waitKey(0);
}

// return image size
Size ImgData::getSize(void){

  return this->size;

}

// return data level
int ImgData::getLevel(void){

  return this->level;

}

// return image path
string ImgData::getPath(void){

  return this->path;

}

// return number of representations at given level
int ImgData::getNumberOfRepresentations(int level){

  switch(level){
    case 4: return polSAR.size();
    case 2: return color.size();
    case 1: return grayscale.size();
  }

}
    
// return data of specified representation at specified level
Mat ImgData::getData(int level, int rep){

  Mat out;
  // check if there is such a representation at specified level
  switch(level){
    case 4: 
		if (rep < polSAR.size()) {
			out = polSAR.at(rep).at(0);  break;
		}
    case 2: if (rep < color.size()) out = color.at(rep);break;
    case 1: if (rep < grayscale.size()) out = grayscale.at(rep);break;
  }
  if (out.rows == 0){
	  cerr << "Error: getData() error - there is no data" << endl;
  }
  return out;  
}

/*
* author: Eunbi Park
* date	: 12.04.2016
* input	: void
* output: vector<Mat> PolSAR
* contents
*		implement function for getting PolSAR data
*/
vector<Mat> ImgData::getPolSARData(){
	if (polSAR.at(0).at(0).rows == 0){
		cerr << "Error: getPolSARData() error - there is no data" << endl;
	}
	return polSAR.at(0);
}
