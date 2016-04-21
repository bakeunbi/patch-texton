/**
* @file ImgData.h
* @brief This class is container of image data
* @author Ronny Haensch
*/

#ifndef IMGDATA_H
#define IMGDATA_H

#include "opencv2/opencv.hpp"

#include "Tools.hpp"

//! This class handels the image data
/*!
  The basis of this class is opencv::Mat
  It is extended by more sophisticated load functions that can handle SAR-data, and projection-functions, that are able to project data of a higher level (eg. PolSAR) to all lower levels (down to binary)
*/
class ImgData{

  public:
    //! the default constructor
    ImgData(void);
    //! constructor
    /*! 
      initilizes object with path to data and its level
      does not actualle load the data
    */
    ImgData(string, int);
    //! deconstructor
    /*!
      releases the original data as well as projections
    */
    ~ImgData(void);

    //! loads the image data
    /*!
      depending on the defined data-level a proper method to load the data is chosen
    */
    void load(void);
    //! loads the image data
    /*!
      sets data path and data level, then calls @sa load(void)
    */
    void load(string, int);
    //! projection of data down to other data levels
    /*!
      projects the given data of data level L_0 down to all other data levels L_i, that hold: L_0 > L_i
      there is no projection to higher data levels
    */
    void project(void);
    
    //! returns the data path
    string getPath(void);
    //! returns the number of representations at level @param level
    int getNumberOfRepresentations(int level);
    //! returns the size of the image
    Size getSize(void);
    //! returns the original data level
    int getLevel(void);
    //! returns the data at level @param level and representation @param rep
    /*!
      returns NULL if there is no data at this level for this representation
    */
    Mat getData(int level, int rep);
	vector<Mat> getPolSARData();


  private:
    //! projection of PolSAR-data down to all other data levels; called by project(void)
    /*!
      --> SAR: take the individual PolSAR channels
      --> Color: load a speckle-reduced version defined by path-till-the"_" + _cov_reflee.rat
		 use transformed PolSAR-channels (entries at main-diagonal of covariance matrix) as color channels (those are already real-valued)
		 Perform log-transformation at each channel: new = log(old + 1)
		 if PolSAR is dual-pol: red-channel = 0.5*(blue + green) (before log-transform)
		 scale to [min, 255]
		 since speckle-reduced image is probably smaller than original, it color image is resized using linear interpolation
		 => at the end: *one* PolSAR-image ==> *one* color-image
    */
    void polSARProjection(void);
    
    //! the data level
    int level;
    //! the data path
    string path;
    //! the size of the image-data
    Size size;
    
    //! the data of level 4: PolSAR-data
    /*!
      each element of the vector corresponds to a channel of the PolSAR data in this order: [0] = _hh, [1] = _vv 
	  (, [2] = _hv <-- in the case of fully-polarimetric data)
    */
    vector< vector<Mat> > polSAR;
    
    //! the data of level 2: color-image
    /*!
      each element of the vector corresponds to different representations of higher-level data
      in the current version this vector should always have only one component; its defined as vector for consistency
    */
    vector<Mat> color;

    //! the data of level 1: gray-image
    /*!
    */
    vector<Mat> grayscale;

};

#endif
