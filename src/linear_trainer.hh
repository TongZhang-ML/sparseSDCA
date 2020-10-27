/********************************************************************
 *  File: linear_trainer.hh
 *  Copyright (C) 2012, 2013 Tong Zhang
 *
 *  Description: Data structure for linear classifier trainers
 *
 ********************************************************************/

#ifndef _LINEAR_TRAIN_HH

#define _LINEAR_TRAIN_HH

#include "linear_classifier.hh"

/**
 * @brief linear classifer trainer
 */
class LinearTrainer {

public:
  /** the loss function type */
  enum loss_t {
    SVM=0,
    LS=1,
    MOD_LS=2,
    SM_HINGE=3,
    LOGISTIC=4
  } loss_type;

  /** how often to check duality gap */
  static const int chk_interval0=3;
  int chk_interval;

  /** gamma for smoothed hinge loss */
  double loss_gamma;

  /** L1 regularization for sparsity */
  double lambda1;
  /** L2 regularization parameter */
  double lambda2;
  /** number of iterations */
  int iters;


  /** first epoch use modified sgd (<0: dynamically determine) */
  int use_sgd0, use_sgd;
  /** whether to use averaging (<0: dynamically determine) */
  int use_avg0, use_avg;

  /** learning rate for fista (>0: use fista with learning rate eta)*/
  double fista_eta;

  /** accleration parameters */
  int accl_iters;
  double accl_kappa, accl_beta;

  /** minimum duality gap */
  float min_dgap;

  /** decision threshold */
  double theta;

  /**
   * eps = lambda1/lambda2
   */
  double eps;

  /** whetehr to update bias */
  bool update_b;

  /** constructor */
  LinearTrainer() : loss_type(SVM), chk_interval(chk_interval0), loss_gamma(1),lambda1(0),lambda2(1e-4), iters(50), 
		    use_sgd0(-1), use_sgd(0), use_avg0(-1), use_avg(0),
		    fista_eta(-1.0), accl_iters(-1), accl_kappa(0.0001), accl_beta(0.9),
 		    min_dgap(1e-3), theta(0), eps(0), update_b(false)
  {
  }

  /**
   * training binary linear classifier using prox-sdca or fista or accelerated proximal sdca
   * @param trn training data
   * @param labels label for the training points  (for classification: equals 0 for regression)
   * @param targets real-valued targets for the training points (for regression: equals 0 for classification)
   * @param tst test data
   * @param tst_labels test label for binary classification
   * @param tst_labels real-valued test targets
   * @param verbose_level determine level of details to print to stderr
   * @return a binary linear applier
   */
  BinaryLinearApplier * train(LinearDataSet & trn, bool * labels, float *targets, 
			      LinearDataSet * tst, bool * tst_labels, float * tst_targets, int verbose_level);

  /**
   * training binary linear classifier using prox-sdca
   * @param trn training data
   * @param labels label for the training points  (for classification: equals 0 for regression)
   * @param targets real-valued targets for the training points (for regression: equals 0 for classification)
   * @param tst test data
   * @param tst_labels test label for binary classification
   * @param tst_labels real-valued test targets
   * @param verbose_level determine level of details to print to stderr
   * @return a binary linear applier
   */
  BinaryLinearApplier * train_sdca(LinearDataSet & trn, bool * labels, float *targets, 
			      LinearDataSet * tst, bool * tst_labels, float * tst_targets, int verbose_level);

  /**
   * training binary linear classifier using fista
   * @param trn training data
   * @param labels label for the training points  (for classification: equals 0 for regression)
   * @param targets real-valued targets for the training points (for regression: equals 0 for classification)
   * @param tst test data
   * @param tst_labels test label for binary classification
   * @param tst_labels real-valued test targets
   * @param verbose_level determine level of details to print to stderr
   * @return a binary linear applier
   */
  BinaryLinearApplier * train_fista(LinearDataSet & trn, bool * labels, float *targets, 
				    LinearDataSet * tst, bool * tst_labels, float * tst_targets, int verbose_level);


  /**
   * training linear binary classifier dusing accelerated prox-sdca
   * @param trn training data
   * @param labels label for the training points  (for classification: equals 0 for regression)
   * @param targets real-valued targets for the training points (for regression: equals 0 for classification)
   * @param tst test data
   * @param tst_labels test label for binary classification
   * @param tst_labels real-valued test targets
   * @param verbose_level determine level of details to print to stderr
   * @return a binary linear applier
   */
  BinaryLinearApplier * train_accl(LinearDataSet & trn, bool * labels, float *targets, 
				  LinearDataSet * tst, bool * tst_labels, float * tst_targets, int verbose_level);

  /**
   * parse str to get parameters
   * @param str string to be parsed (possibly contains parameter info)
   * @return whether the string contains parameter info
   */
  bool commandLineParse(char * str);

  /**
   *  print possible parsing options to os
   */
  static void printCommandLineOptions(ostream & os);
  
  /**
   *  print current training parameters options to os
   */
  void printTrainingParameters(ostream & os);

};


/**
 * basic structure for dual coordinate ascent
 */
class AlphCalcBasic {
public:
  /** training parameters */
  LinearTrainer trn;
  /** training data */
  LinearDataPoint *ldps;
  /** binary label for classification */
  bool *labels;
  /** real valued target for regression */
  float *targets;
  /** training data size */
  int num;

  /** dual variable */
  double *alpha;

  /** training data's lambda2 times training-data-size */
  double lambda;

  /** temporary variable inverse of data norm square */
  double *ix2;
  /** upper bound for dual */
  double pC2;

  /** whether to update b */
  bool update_b;

  /** eps is lambda1/lambda2 */
  double eps;
  /** trained weight vector */
  ParamVector *weightVec;

  /** constructor */
  AlphCalcBasic(LinearTrainer _trn, BinaryLinearApplier *_appl,
	   LinearDataSet & _lds,  bool *_labels, float *_targets,
	   bool _update_b) : 
    trn(_trn), ldps(_lds.ldps),  labels(_labels), targets(_targets), num(_lds.num), update_b(_update_b) 
  {
    weightVec=&_appl->weightVec;

    if (targets) {
      labels=0;
    }

    // initialize dual
    alpha = new double [num];
    bzero(alpha,sizeof(double)*num);

    ix2 = new double [num];
  }

  /**
   * compute delta of the k-th dual variable based on primal prediction sum
   * @param sum  primal prediction value
   * @param k the k-th dual variable
   * @return delta of the k-th dual variable to maximize dual 
   */
  double deltaAlpha(double sum, int k);

  /** create ix2 for modified sgd */
  void initIx2 ();

  /** after modified sgd before dual updates */
  void updateIx2();

  /**
   * compute the linear weight dot the k-th datum
   */
  double computeSum(int k) {
    double sum = weightVec->trunc_dot(ldps[k],eps);
    return labels?((labels[k])?sum:-sum):sum;
  }

  /**
   * update weight vector for the k-th data point using formula:
   *   weight += del * data[k] * label[k] 
   * @param k  the k-th data point
   * @param del scalar rate of update
   */
  void updateWeight(int k, double del) {
    if (del<1e-10 && del>-1e-10) return;

    alpha[k] +=del;

    if (labels) { // classification
      if (!labels[k]) del = -del;
    }
    weightVec->saxpy(ldps[k],del,update_b);
  }


  
  /**
   * update dual alpha for the k-th point
   * @param k k-th data point
   */
  void updateAlpha(int k) {
    double sum=computeSum(k);
    double dalph=deltaAlpha(sum,k);
    updateWeight(k,dalph);
  }

  /** destructor */
  void destroy() {
    delete [] alpha;
    delete [] ix2;
  }

};

/** compute permutation */
int *new_order(int num);
void permutation(int * order, int num);

  
#endif

