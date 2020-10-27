/********************************************************************
 *  File: linear_classifier.hh
 *  Copyright (C) 2012, 2013 Tong Zhang
 *
 *  Description: Data structure for linear classifiers
 *
 ********************************************************************/

#ifndef _LINEAR_CLASSIFIER_HH

#define _LINEAR_CLASSIFIER_HH

#include <cctype>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <cstring>
#include <cstdio>

#include <memory>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

/*! \mainpage Package: Acclerated Proximal Stochastic Dual Coordinate Ascent for Solving L1-L2 Regularized Binary Linear Classification & Regression
 *
 * \section intro_sec Introduction
 *
 * This code employs acclerated prox-SDCA to train binary linear optimization problems based on the paper 
 *       Accelerated Proximal Stochastic Dual Coordinate Ascent for Regularized Loss Minimization.
 *      
 * It may be used to do experiments similar to what's reported in the paper (with different default parameter configurations)   
 *
 * \section install_sec Installation
 *
 * go to src and type make
 *
 * \section exp_sec Experiments
 *
 * use the "exp/run" shell script to do experiments
 *  
 */

/** 
 * use double for weights
 */
typedef double linfloat_t;

/**
 * @brief basic structure for data
 */
class LinearDataPoint {
public:
  /** number of nonzero float features in sparse format */
  int numFloatFeatures;

  /** sparse float array */
  struct FltArray {
    int index;
    float value;
  } * floatArray;

  /** number of nonzero binary features in sparse format */
  int numBinaryFeatures;

  /** binary index array */
  int *binaryIndexArray;

  /** normalization factor */
  double norm_factor;

  /**
   * default constructor 
   */
  LinearDataPoint() : numFloatFeatures(0), floatArray(0),
		      numBinaryFeatures(0), binaryIndexArray(0) , norm_factor(1){}
  /**
   * constructor 
   */
  LinearDataPoint(int _numFloatFeatures, int *_flt_ind, float * _flt_val,
		  int _numBinaryFeatures, int *_bin_ind, float _norm_fact);

  /** compute normalization factor
   *  @param norm normalize to normalization-factor of norm
   */
  void normalize(double norm);
  
  /** maximum float feature index */
  int max_float_index(); 

  /** maximum binary feature index */
  int max_binary_index(); 

  /** destroy the data */
  void destroy();
};

/**
 * @breaf Linear Dataset structure
 */
class LinearDataSet {
public:
  /** array of linear data points */
  LinearDataPoint * ldps;
  
  /** size of the data set */
  int num;

  LinearDataSet() : ldps(0), num(0) {}

  /** return the size of the data set */
  int size() {
    return num;
  }

  /** destroy the data set */
  void destroy();
  
  /** clear the data set */
  void clear() {
    ldps=0;
    num=0;
  }
    
};



#include <vector>

/**
 * @brief structure to read linear dataset
 */
class LinearDataReader {
protected:
  /** tempory binary labels */
  bool * binaryLabels;
  /** temporary linear data set */
  LinearDataSet lds;

  /**
   * push one label
   */
  void push_label(double);

  /**
   * read labels from input stream
   */
  void read_labels(istream & is);

  /**
   * read features from input stream
   */
  void read_features(istream & is, bool read_label, float norm_fact);

  /** linear data points */
  vector<LinearDataPoint> ldp_vec;
  /** labels */
  vector<int> label_vec;

  /** targets */
  vector<float> target_vec;

public:

  LinearDataReader() : binaryLabels(0) {}

  /** size of the data set */
  int size() {
    return label_vec.size();
  }

  /** get binary labels for class k */
  bool * get_binary_labels(int k) {
    for (int i=0; i< size(); i++) binaryLabels[i]=(label_vec[i]==k);
    return binaryLabels;
  }

  /**
   * get regression targets
   */
  float * get_targets() {
    return &target_vec[0];
  }


  /** the i-th label */
  int get_label(int i) {
    return label_vec[i];
  }

  /** get data set */
  LinearDataSet & get_dataset() {
    lds.ldps= & ldp_vec.front();
    lds.num = size();
    return lds;
  }

  /** get the maximum label number */
  int max_label();


  /** 
   * read labels from file label_fn (if label_fn is not null), and features from file feat_fn
   * with normlalization factor norm_fact
   * return true if labels are read from file, false if filled with zero
   */
  bool append_data(char * label_fn, char * feat_fn, float norm_fact);

  ~LinearDataReader();

};


/**
 * @brief Paramer Vector
 */
class ParamVector {
public:
  /** number of sparse float features */
  int numFloatFeatures;
  /** float feature weights */
  linfloat_t * floatVec;

  /** number of binary features */
  int numBinaryFeatures;
  /** binary feature weights */
  linfloat_t * binaryVec;

  /** bias */
  linfloat_t b;

  /**
   * default constructor
   */
  ParamVector() {
    clear();
  }

  /** allocate memory 
   *  @param nf  number of float features
   *  @param nb  number of binary features
   */
  void allocate(int nf, int nb);

  /**
   *  clear the vector
   */
  void clear();

  /** 
   * init to zero 
   */
  void setZero();


  /** 
   * implement saxpy:
   * self = self + a * src 
   *
   * @param src src vector
   * @param a   scaling factor
   */
  void saxpy(ParamVector & src, double a);


  /** 
   * implement saxpy with sparse vector
   * self = self + del * ldp 
   *
   * @param ldp  sparse data vector
   * @param del scaling factor
   * @param update_b whether should update the bias
   *
   */
  void saxpy(LinearDataPoint & ldp, double del, bool update_b);

  /**
   *  implement scaled addition of two vectors
   *  self= alpha*x+beta*y 
   *
   *  @param alpha scaling factor for vector x 
   *  @param x vector
   *  @param beta scaling factor for vector y
   *  @param y vector
   *
   **/
  void scaled_add(double alpha, ParamVector &x, double beta, ParamVector &y);

  /**
   * copy content from source (assume ememory already allocated)
   * 
   */
  void copyFrom(ParamVector & src);

  /**
   * compute the linear weight dot the datum
   *
   * @param ldp  data to compute dot product
   * @return dot-product
   */
  double dot(LinearDataPoint & ldp);

  /**
   * compute the linear weight dot the datum with boundary check
   *
   * @param ldp  data to compute dot product
   * @return dot-product
   */
  double dot_with_boundary_check(LinearDataPoint & ldp);

  /**
   * compute the truncated linear weight (soft-shrinkage to zero by eps) dot the datum with truncation
   *
   * @param ldp  data to compute dot product
   * @param eps truncation amount
   * @return dot-product
   */
  double trunc_dot(LinearDataPoint & ldp, double eps);

  /**
   * truncate the weights (soft-shrinkage to zero by eps)
   * @param eps truncation amount
   */
  void trunc(double eps);

  /** 
   *de-allocate memory 
   */
  void destroy();

  /** 
   * write to os 
   */
  void write(ostream & os);

  /** 
   * read from is 
   */
  void read(istream & is);
};


/**
 * @brief Binary Linear classifier
 */
class BinaryLinearApplier {
protected:
  /** tempory linear data-point */
  LinearDataPoint ldp;
public:
  /** linear weights */
  ParamVector weightVec;

  /** classification threhold */
  double threshold;
  
  /** probability model for linear applier */
  class ProbabilityModel {
  public:
    /** the type of probability model 
     *  determined by traning loss function
     */
    enum prob_t {
      LINEAR=0,
      EXP=1
    } type;

    /** return probability given score */
    double get_probability(double score) {
      if (type==LINEAR) {
 	return (score<-1)?0:(score>1?1:(score+1.0)/2.0);
      }
      return 1.0/(1+exp(-score));
    }

    ProbabilityModel() : type(LINEAR) {}
  } prob;

  BinaryLinearApplier() : threshold(0) {}

  /**
   * get score from LinearDataPoint.
   * @param ldp the LinearDataPoint for which to obtain score
   * @return the score
   */
  double apply(LinearDataPoint &ldp); 

  /**
   * @param score the score from apply
   * @return the associated probability
   */
  double get_probability(double score) {
    return prob.get_probability(score);
  }

  /**
   * compute the class label for a LinearDataPoint
   * @param ldp input LinearDataPoint to be classified
   * @return the class label
   */
  bool classify(LinearDataPoint & ldp) {
    return apply(ldp)>threshold;
  }

  /**
   * allocate array for this applier based on the max number of features in training data
   * @param trn training data
   */
  void allocate(LinearDataSet &trn);


  /**
   * write to stream
   * @param use_binary wehther to use binary format
   * @param os output-stream
   */
  void write(ostream & os);

  /**
   * read from stream
   * @param is input-stream
   */
  void read(istream & is);

  /**
   * destroy the applier
   */
  void destroy();

  /**
   * clear the applier
   */
  void clear() {
    weightVec.clear();
  }

  ~BinaryLinearApplier() {}

};


/** 
 * @brief Structure for multi-category classifier
 */
class MultiLinearClassifier {
public:
  vector<BinaryLinearApplier *> appl_vec;

  /**
   * classify a data point and return scr
   */
  int classify(LinearDataPoint & ldp, double &scr);

  /** 
   * write to file
   */  
  void write(ostream & os);

  /**
   * read from file
   */
  void read(istream & is);

  /**
   *  destroy the data
   */
  void destroy();
};


class BinaryTestStat {
public:
  /**
   * true positive
   */
  int tp;
  /**
   * true negative
   */
  int tn;
  /**
   * false positive
   */
  int fp;
  /**
   * flase negative
   */
  int fn; 

  BinaryTestStat(): tp(0), tn(0), fp(0), fn(0) {}

  /**
   * update statistics
   * @param bts add bts stats to current stats
   */
  void update(BinaryTestStat & bts)
  {
    tp += bts.tp;
    tn += bts.tn;
    fp += bts.fp;
    fn += bts.fn;
  }
  
  /**
   * update statistics
   * @param truth true label
   * @param pred predicted label
   */
  void update(bool truth, bool pred) {
    if (truth) {
      if (pred) tp++;
      else fn++;
    }
    else {
      if (pred) fp++;
      else tn++;
    }
  }

  /**
   * validate a binary classifier on the data set and update statitics
   * @param appl binary classifier
   * @param ds test data
   * @param labels true labels for the test data
   */
  void update(BinaryLinearApplier & appl, LinearDataSet & ds, bool * labels) 
  {
    for (int i=0; i<ds.size(); i++) 
      update(labels[i],appl.classify(ds.ldps[i]));
  }

  /**
   * validate a binary classifier on the data set using threshold theta and update statitics
   * @param appl binary classifier
   * @param ds test data
   * @param labels true labels for the test data
   * @param theta decision threshold
   */
  void update(BinaryLinearApplier & appl, LinearDataSet & ds, bool * labels, double theta) {
    for (int i=0; i<ds.size(); i++) 
      update(labels[i],(appl.apply(ds.ldps[i])>theta));
  }
    
    /**
     * accuracy
     */
    double accuracy() {
      return (tp+tn)/(tp+tn+fp+fn+1e-10);
    }
    /**
     * precision
     */
    double precision() {
      return tp/(tp+fp+1e-10);
    }
    /**
     * recall
     */
    double recall() {
      return tp/(tp+fn+1e-10);
    }
   /**
     * f-measure
     */
    double fb1() {
      return 2.0/(1.0/precision()+1.0/recall());
    }
  /**
   * print statistics
   */
  void print(ostream & os, const char *indent="") {
    os << indent <<
      "tp=" << tp << " fp=" << fp << " tn="<< tn << " fn=" << fn <<endl;
    os << indent <<
      "precision="<<precision() << " recall=" << recall()
       << " Fb1=" << fb1() 
       << " accuracy=" << accuracy() <<endl;
  }
  /** clear the statistics */
  void clear() {
    tp=fp=tn=fn=0;
  }


  /**
   * validate a linear classifier on the data set and update statitics
   * @param appl binary linear classifier
   * @param lds linear test data set
   * @param labels true labels for the test data
   */
  void multi_update(BinaryLinearApplier & appl, LinearDataSet lds, bool *labels)
  {
    for (int i=0; i<lds.size(); i++) {
      update(labels[i],appl.classify(lds.ldps[i]));
    }
  }
};


  
#endif

