#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <fstream>
#include <iostream>
#include <vector>
#include <ciso646>

#include <boost/math/special_functions/gamma.hpp>
#include <boost/random.hpp>
#include <boost/random/gamma_distribution.hpp>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

// check if buffer is a valid command or a command or a blank line
static bool isCommentline(string* buffer){

    // if line is empty or starts with # (which means its a comments)
    if ((buffer->length()==0) or (buffer->at(0) == '#'))
      return true;
  
    // delete leading blanks
    if (buffer->at(0)== ' '){
      buffer->erase(0, buffer->find_first_not_of(" "));
    }

    // delete blanks at end of string
    int diff = (buffer->length() - buffer->find_last_not_of(" ") - 1);
    if (diff !=0)
      buffer->erase(buffer->find_last_not_of(" ") + 1, diff);
  
    // if line is empty or starts with # (which means its a comments)
    if ((buffer->length()==0) or (buffer->at(0) == '#'))
      return true;

    return false;
}

static double gamrnd( const double shape, double scale, boost::mt19937& rng ) {

  boost::gamma_distribution<> gd( shape );
  boost::variate_generator<boost::mt19937&,boost::gamma_distribution<> > var_gamma( rng, gd );

  return scale*var_gamma();
}

static Mat sampleFromDirichlet( Mat alpha, int n ){
 
    boost::mt19937 rng;
 
	int k = alpha.cols;
	Mat theta = Mat::zeros(n, k, CV_32FC1);
	
	double scale = sum(alpha).val[0]/k;

	for(int j=0; j<n; j++){
		if (scale > 0){
			double S = 0;
			for(int i=0; i<k; i++){

				double shape = alpha.at<float>(i);
				if (shape == 0) continue;
				
				theta.at<float>(j,i) = gamrnd(shape, scale, rng);
				S += theta.at<float>(j,i);
			}
			for(int i=0; i<k; i++){
				theta.at<float>(j,i) /= S;
			}
		}else{
			int i = ((double)rand())/RAND_MAX*k;
			theta.at<float>(j,i) = 1;
		}
	}
	
	return theta;

}

static Mat sampleFromDirichlet_old( Mat alpha, int n ){
 
  double B = 0, sum = 0, v=0;
    
  // B serves just as normalization, which is constant for constant alpha
//   for(int i=0; i<alpha->cols; i++){
//       sum += cvGetReal1D(alpha, i);
//       B -= boost::math::lgamma( cvGetReal1D(alpha, i) );
//   }
//   B += boost::math::lgamma(sum);

  int num = n*alpha.cols*100;
  num = 1000;
  Mat x = Mat(num, alpha.cols, CV_32FC1);
  double* P = new double[num];
  double max;
  for(int i=0; i<num; i++){
      sum=0;
      for(int j=0; j<alpha.cols; j++){
		  v = rand();
		  x.at<float>(i,j) = v;
		  sum += v;
      }
      P[i] = 0;
      for(int j=0; j<alpha.cols; j++){
		x.at<float>(i, j) /= sum;
	  
	  if ( x.at<float>(i, j) > 0 )
	    P[i] += (alpha.at<float>(j)-1) * log(x.at<float>(i, j));
      }
//       P[i] += B;
      if ((max < P[i]) or (i==0))
		max = P[i];
  }
  for(int i=0; i<num; i++){
      P[i] = exp(P[i]-max+10);
      if (i>0)
		P[i] += P[i-1];
  }
  
  v = ((double)rand())/RAND_MAX*P[num-1];
  
  int i=0;
  while(P[i]<=v){
      i++;
  }
  Mat out = Mat(1, alpha.cols, CV_32FC1);
  for(int j=0; j<alpha.cols; j++){
      out.at<float>(j) = x.at<float>(i, j);
  }

  delete[](P);
  
  return out;

}

static void loadRAT(string fname, vector<Mat>& data){

  bool verbose = true;
  
  // header info
  unsigned int dim;
  unsigned int* size;
  unsigned int var;
  unsigned int type;
  unsigned int dummy;
  char info[80];
  
  // open file
  fstream file(fname.c_str(), ios::in | ios::binary);
  if (!file){
	  cerr << "ERROR: Cannot open file: " << fname << endl;
	  return;
  }
  // read header
  file.read((char*)(&dim), sizeof(dim));
  dim = (dim>>24) | ((dim<<8) & 0x00FF0000) | ((dim>>8) & 0x0000FF00) | (dim<<24);
  size = new unsigned int[dim];
  for(int i=0; i<dim; i++){
      file.read((char*)(size+i), sizeof(size[i]));
      size[i] = (size[i]>>24) | ((size[i]<<8) & 0x00FF0000) | ((size[i]>>8) & 0x0000FF00) | (size[i]<<24);
  }
  file.read((char*)(&var), sizeof(var));
  var = (var>>24) | ((var<<8) & 0x00FF0000) | ((var>>8) & 0x0000FF00) | (var<<24);
  file.read((char*)(&type), sizeof(type));
  type = (type>>24) | ((type<<8) & 0x00FF0000) | ((type>>8) & 0x0000FF00) | (type<<24);
  file.read((char*)(&dummy), sizeof(dummy));
  file.read((char*)(&dummy), sizeof(dummy));
  file.read((char*)(&dummy), sizeof(dummy));
  file.read((char*)(&dummy), sizeof(dummy));
  file.read(info, sizeof(info));
  
  if (verbose){
      cout << "Number of image dimensions:\t" << dim << endl;
      cout << "Image dimensions:\t";
      for(int i=0; i<dim-1; i++)
	  cout << size[i] << " x ";
      cout << size[dim-1] << endl;
      cout << "Data type:\t" << var << endl;
      cout << "Type:\t" << type << endl;
      cout << "Info:\t" << info << endl;
  }

  int nChannels=0, dsize=0;
  switch (var){
    case 1: nChannels=1;dsize=1; break;
    case 2: nChannels=1;dsize=4; break;
    case 3: nChannels=1;dsize=4; break;
    case 4: nChannels=1;dsize=4; break;
    case 5: nChannels=1;dsize=8; break;
    case 12: nChannels=1;dsize=4; break;
    case 13: nChannels=1;dsize=4; break;
    case 14: nChannels=1;dsize=8; break;
    case 15: nChannels=1;dsize=8; break;
    case 6: nChannels=2;dsize=4; break;
    case 9: nChannels=2;dsize=8; break;
    default: cerr << "ERROR: arraytyp not recognized (wrong format?)" << endl;
  }  
 
  //char buf[dsize];
  //char swap[dsize];
  char *buf = new char[dsize];
  char *swap = new char[dsize];
  int i,j,x,y;
  Mat img, real, imag;
  switch(dim){
      case 2:
		  real = Mat::zeros(size[1], size[0], CV_32FC1);
		  imag = Mat::zeros(size[1], size[0], CV_32FC1);
	      for(y=0; y<size[1]; y++){
			  for(x=0; x<size[0]; x++){
				  double realVal, imagVal;
				  file.read((buf), dsize);
				  for(int d=0; d<dsize; d++) swap[d] = buf[dsize-d-1];
				  switch (var){
					  case 1: dsize=1;realVal = *((char*)swap);break;	// byte
					  case 2: dsize=4;realVal = *((int*)swap);break;	// int
					  case 3: dsize=4;realVal = *((long*)swap);break;	// long
					  case 4: dsize=4;realVal = *((float*)swap);break;	// float
					  case 5: dsize=8;realVal = *((double*)swap);break;	// double
					  case 6: dsize=4;					// complex
						  realVal = *((float*)swap);
						  file.read((buf), dsize);
						  for(int d=0; d<dsize; d++) swap[d] = buf[dsize-d-1];
						  imagVal = *((float*)swap);break;
					  case 9: dsize=8;					// dcomplex
						  realVal = *((double*)swap);
						  file.read((buf), dsize);
						  for(int d=0; d<dsize; d++) swap[d] = buf[dsize-d-1];
						  imagVal = *((double*)swap);break;
					  case 12: dsize=4;realVal = *((unsigned int*)swap);break;	// uint
					  case 13: dsize=4;realVal = *((unsigned long*)swap);break;	// ulong
					  case 14: dsize=4;realVal = *((double*)swap);break;	// l64
					  case 15: dsize=4;realVal = *((double*)swap);break;	// ul64
				  }
				  real.at<float>(size[1]-y-1, x) = realVal;
				  if (nChannels == 2)
					imag.at<float>(size[1]-y-1, x) = imagVal;
				  //cvSet2D(data->at(0), size[1]-y-1, x, cur);
			  }
	      }
	      if (nChannels == 2){
			  vector<Mat> channels;
			  channels.push_back(real);
			  channels.push_back(imag);
			  merge(channels, img);
		  }else
			  img = real.clone();
	      data.push_back(img);
	      break;
      case 3: 
		  for(i=0; i<size[0]; i++){
		  real = Mat::zeros(size[2], size[1], CV_32FC1);
		  
		  if (nChannels == 2){
			  vector<Mat> channels;
			  imag = Mat::zeros(size[2], size[1], CV_32FC1);
			  channels.push_back(real);
			  channels.push_back(imag);
			  merge(channels, img);
		  }else
			  img = real.clone();
			
		  data.push_back(img.clone());
		  }
			  
	      for(y=0; y<size[2]; y++){
			  for(x=0; x<size[1]; x++){
				  for(i=0; i<size[0]; i++){
				  double realVal, imagVal;
				  //file.read((char*)&buf, dsize);
				  file.read(buf, dsize);
				  for(int d=0; d<dsize; d++) swap[d] = buf[dsize-d-1];
				  switch (var){
					  case 1: dsize=1;realVal = *((char*)swap);break;	// byte
					  case 2: dsize=4;realVal = *((int*)swap);break;	// int
					  case 3: dsize=4;realVal = *((long*)swap);break;	// long
					  case 4: dsize=4;realVal = *((float*)swap);break;	// float
					  case 5: dsize=8;realVal = *((double*)swap);break;	// double
					  case 6: dsize=4;					// complex
						  realVal = *((float*)swap);
						  //file.read((char*)&buf, dsize);
						  file.read(buf, dsize);
						  for(int d=0; d<dsize; d++) swap[d] = buf[dsize-d-1];
						  imagVal = *((float*)swap);break;
					  case 9: dsize=8;					// dcomplex
						  realVal = *((double*)swap);
						  //file.read((char*)&buf, dsize);
						  file.read(buf, dsize);
						  for(int d=0; d<dsize; d++) swap[d] = buf[dsize-d-1];
						  imagVal = *((double*)swap);break;
					  case 12: dsize=4;realVal = *((unsigned int*)swap);break;	// uint
					  case 13: dsize=4;realVal = *((unsigned long*)swap);break;	// ulong
					  case 14: dsize=4;realVal = *((double*)swap);break;	// l64
					  case 15: dsize=4;realVal = *((double*)swap);break;	// ul64
				  }
				  if (nChannels != 2)
					data.at(i).at<float>(size[2]-y-1, x) = realVal;
				  else
					data.at(i).at<Vec2f>(size[2]-y-1, x) = Vec2f(realVal, imagVal);
				  //cvSet2D(data->at(0), size[1]-y-1, x, cur);
				}
			  }
	      }
		  /*
		  img = cvCreateImage(cvSize(size[1], size[2]), IPL_DEPTH_32F, nChannels);
		  cvZero(img);
		  data->push_back(img);
	      }
	      for(y=0; y<size[2]; y++){
		  for(x=0; x<size[1]; x++){
		      for(i=0; i<size[0]; i++){
			  CvScalar cur;
			  file.read((char*)(&buf), dsize);
			  for(int d=0; d<dsize; d++) swap[d] = buf[dsize-d-1];
			  switch (var){
			      case 1: dsize=1;cur.val[0] = *((char*)swap);break;	// byte
			      case 2: dsize=4;cur.val[0] = *((int*)swap);break;	// int
			      case 3: dsize=4;cur.val[0] = *((long*)swap);break;	// long
			      case 4: dsize=4;cur.val[0] = *((float*)swap);break;	// float
			      case 5: dsize=8;cur.val[0] = *((double*)swap);break;	// double
			      case 6: dsize=4;					// complex
				      cur.val[0] = *((float*)swap);
				      file.read((char*)(&buf), dsize);
				      for(int d=0; d<dsize; d++) swap[d] = buf[dsize-d-1];
				      cur.val[1] = *((float*)swap);break;
			      case 9: dsize=8;					// dcomplex
				      cur.val[0] = *((double*)swap);
				      file.read((char*)(&buf), dsize);
				      for(int d=0; d<dsize; d++) swap[d] = buf[dsize-d-1];
				      cur.val[1] = *((double*)swap);break;
			      case 12: dsize=4;cur.val[0] = *((unsigned int*)swap);break;	// uint
			      case 13: dsize=4;cur.val[0] = *((unsigned long*)swap);break;	// ulong
			      case 14: dsize=4;cur.val[0] = *((double*)swap);break;	// l64
			      case 15: dsize=4;cur.val[0] = *((double*)swap);break;	// ul64
			  }
			  cvSet2D(data->at(i), size[2]-y-1, x, cur);
		      }
		  }
	      }
		  }*/
	      break;
      case 4: for(i=0; i<size[0]; i++){
				  for(j=0; j<size[1]; j++){
					  real = Mat::zeros(size[3], size[2], CV_32FC1);
					  if (nChannels == 2){
						  imag = Mat::zeros(size[3], size[2], CV_32FC1);
						  vector<Mat> channels;
						  channels.push_back(real);
						  channels.push_back(imag);
						  merge(channels, img);
					  }else
						  img = real.clone();
					  data.push_back(img.clone());
				  }
				}
			  
	      for(y=0; y<size[3]; y++){
			  for(x=0; x<size[2]; x++){
				  for(j=0; j<size[0]; j++){
			  for(i=0; i<size[1]; i++){
				  double realVal, imagVal;
				  file.read((buf), dsize);
				  for(int d=0; d<dsize; d++) swap[d] = buf[dsize-d-1];
				  switch (var){
					  case 1: dsize=1;realVal = *((char*)swap);break;	// byte
					  case 2: dsize=4;realVal = *((int*)swap);break;	// int
					  case 3: dsize=4;realVal = *((long*)swap);break;	// long
					  case 4: dsize=4;realVal = *((float*)swap);break;	// float
					  case 5: dsize=8;realVal = *((double*)swap);break;	// double
					  case 6: dsize=4;					// complex
						  realVal = *((float*)swap);
						  file.read((buf), dsize);
						  for(int d=0; d<dsize; d++) swap[d] = buf[dsize-d-1];
						  imagVal = *((float*)swap);break;
					  case 9: dsize=8;					// dcomplex
						  realVal = *((double*)swap);
						  file.read((buf), dsize);
						  for(int d=0; d<dsize; d++) swap[d] = buf[dsize-d-1];
						  imagVal = *((double*)swap);break;
					  case 12: dsize=4;realVal = *((unsigned int*)swap);break;	// uint
					  case 13: dsize=4;realVal = *((unsigned long*)swap);break;	// ulong
					  case 14: dsize=4;realVal = *((double*)swap);break;	// l64
					  case 15: dsize=4;realVal = *((double*)swap);break;	// ul64
				  }
				  if (nChannels != 2)
					data.at(j*size[1] + i).at<float>(size[3]-y-1, x) = realVal;
				  else
					data.at(j*size[1] + i).at<Vec2f>(size[3]-y-1, x) = Vec2f(realVal,imagVal);
				}
			}
				  //real.at<float>(size[3]-y-1, x) = realVal;
				  //if (nChannels == 2)
					//imag.at<float>(size[3]-y-1, x) = imagVal;
				  //cvSet2D(data->at(0), size[1]-y-1, x, cur);
			  }
	      }
	      
	  /*
		      img = cvCreateImage(cvSize(size[2], size[3]), IPL_DEPTH_32F, nChannels);
		      cvZero(img);
		      data->push_back(img);
		  }
	      }	 
	      for(y=0; y<size[3]; y++){
		  for(x=0; x<size[2]; x++){
		      for(j=0; j<size[0]; j++){
			  for(i=0; i<size[1]; i++){
			      CvScalar cur;
			      file.read((char*)(&buf), dsize);
			      for(int d=0; d<dsize; d++) swap[d] = buf[dsize-d-1];
			      switch (var){
				  case 1: dsize=1;cur.val[0] = *((char*)swap);break;	// byte
				  case 2: dsize=4;cur.val[0] = *((int*)swap);break;	// int
				  case 3: dsize=4;cur.val[0] = *((long*)swap);break;	// long
				  case 4: dsize=4;cur.val[0] = *((float*)swap);break;	// float
				  case 5: dsize=8;cur.val[0] = *((double*)swap);break;	// double
				  case 6: dsize=4;					// complex
					  cur.val[0] = *((float*)swap);
					  file.read((char*)(&buf), dsize);
					  for(int d=0; d<dsize; d++) swap[d] = buf[dsize-d-1];
					  cur.val[1] = *((float*)swap);break;
				  case 9: dsize=8;					// dcomplex
					  cur.val[0] = *((double*)swap);
					  file.read((char*)(&buf), dsize);
					  for(int d=0; d<dsize; d++) swap[d] = buf[dsize-d-1];
					  cur.val[1] = *((double*)swap);break;
				  case 12: dsize=4;cur.val[0] = *((unsigned int*)swap);break;	// uint
				  case 13: dsize=4;cur.val[0] = *((unsigned long*)swap);break;	// ulong
				  case 14: dsize=4;cur.val[0] = *((double*)swap);break;	// l64
				  case 15: dsize=4;cur.val[0] = *((double*)swap);break;	// ul64
			      }
			      cvSet2D(data->at(j*size[1] + i), size[3]-y-1, x, cur);
			  }
		      }
		  }
	      }*/
	      break;
  }
  delete[](size);
  delete[](swap);
  delete[](buf);
}

static bool compPos(CvScalar first, CvScalar second){

  return first.val[3] > second.val[3];

}

#endif
