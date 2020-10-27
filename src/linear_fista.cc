/********************************************************************
 *  File: linear_fasta.cc
 *  Copyright (C) 2012, 2013 Tong Zhang
 *
 *  Description: FISTA algorithm for sparse binary linear classifiers with L1-L2 regularization
 *
 ********************************************************************/

#include "linear_trainer.hh"



class GradCalc {
public:
  /** training parameters */
  LinearTrainer trn;
  /** trained classifier */
  BinaryLinearApplier *appl;
  /** training data */
  LinearDataPoint *ldps;
  /** binary label for classification */
  bool *labels;
  /** real valued target for regression */
  float *targets;
  /** training data size */
  int num;


  /** whether to update b */
  bool update_b;

  /** x, y, gradient */
  ParamVector x0,x,y,g;

  GradCalc(LinearTrainer _trn, BinaryLinearApplier *_appl,
	   LinearDataSet & _lds,  bool *_labels, float *_targets,
	   bool _update_b) : 
    trn(_trn), appl(_appl), ldps(_lds.ldps),  labels(_labels), targets(_targets), num(_lds.num), update_b(_update_b) 
  {
    
    if (targets) {
      labels=0;
    }
    ParamVector ww=appl->weightVec;
    x=ww;
    x.setZero();
    x0.allocate(ww.numFloatFeatures,ww.numBinaryFeatures);
    y.allocate(ww.numFloatFeatures,ww.numBinaryFeatures);
    g.allocate(ww.numFloatFeatures,ww.numBinaryFeatures);
  }


  /**
   * compute gradient for the k-th point
   */
  double div(ParamVector & w, int k) {
    double sum=w.dot(ldps[k]);
    sum = labels?((labels[k])?sum:-sum):sum;

    double v;
    
    if (labels) { // classification
      switch(trn.loss_type) {
      case LinearTrainer::SVM:
	v= (sum<1)?-1.0:0.0;
	break;
      case LinearTrainer::SM_HINGE:
	v=((sum>=1.0)?0:((sum<=1-trn.loss_gamma)?-1.0:((sum-1.0)/trn.loss_gamma)));
	break;
      case LinearTrainer::MOD_LS:
	v=((sum>=1)?0:((sum<=0)?-1.0:(sum-1.0)));
	break;
      case LinearTrainer::LS:
	v= (sum-1.0);
	break;
      default: // LinearDualTrainer::LOGISTIC
	v= - 1.0/(1.0+exp(sum));
      }
    }
    else { // regression
      v= sum-targets[k];
    }
    return v;
  }

  /**
   * compute Gradient using y
   */ 
  void grad(ParamVector & w, ParamVector &g) {
    g.setZero();
    for (int k=0; k<num; k++) {
      LinearDataPoint ldp=ldps[k];
      double divv=div(w,k)/num;
      if (labels) { // classification
	if (!labels[k]) divv=-divv;
      }
      g.saxpy(ldp,divv,update_b);
    }
    g.saxpy(w,trn.lambda2);
  }

  

  /** compute primal objective value
   *  primal w
   *  pobj: primal objective value
   *  return: primal objective
   */
  double pobj() 
  {
    // primal value
    double p=0;

    int i,k;
    double sum;

    double pobj=0;
    // go through data
    for (k=0; k<num; k++) {
      sum=x.dot(ldps[k]);
      sum = labels?((labels[k])?sum:-sum):sum;

      switch(trn.loss_type) {
      case LinearTrainer::SVM:
	p = ((sum>1)?0:(1-sum));
	break;
      case LinearTrainer::SM_HINGE:
	p = ((sum>=1)?0:((sum<=1-trn.loss_gamma)?(1-sum-0.5*trn.loss_gamma)
			 :.5*(1-sum)*(1-sum)/trn.loss_gamma));
	break;
      case LinearTrainer::MOD_LS:
	p = ((sum>1)?0:.5*(1-sum)*(1-sum));
	break;
      case LinearTrainer::LS:
	if (labels) { // classification
	  p = (.5*(1-sum)*(1-sum));
	}
	else {  // regression
	  p = (.5*(targets[k]-sum)*(targets[k]-sum));
	}
	break;
      default: // LinearDualTrainer::LOGISTIC
	p = log(1.0+exp(-sum));
      }
      pobj+=p;
    }

    // regularization
    double r=x.b*x.b;
    {
      double eps=trn.eps;
      for (i=0; i<x.numFloatFeatures; i++) {
	register double ww=x.floatVec[i];
	r += ww*ww+2*eps*fabs(ww);
      }
      for (i=0; i<x.numBinaryFeatures; i++) {
	register double ww=x.binaryVec[i];
	r += ww*ww+2*eps*fabs(ww);
      }
    }
    pobj=pobj/num + 0.5*r* trn.lambda2;
    return pobj;
  }

  void proximal(double eta) {
    grad(y,g);
    y.saxpy(g,-eta);
    x.copyFrom(y);
    x.trunc(trn.lambda1*eta);
  }

  void fistaIter(double eta, double beta) {
    x0.copyFrom(x);
    proximal(eta);
    y.scaled_add((1+beta),x,-beta,x0);
    appl->weightVec.b=x.b;
  }

  ~GradCalc() {
    x0.destroy();
    y.destroy();
    g.destroy();
  }
    
};



BinaryLinearApplier * LinearTrainer::train_fista
(LinearDataSet & trn, bool * labels, float * targets,
 LinearDataSet * tst, bool * tst_labels, float * tst_targets, int verbose_level)
{

  if (eps<0) eps=0;
  eps = (lambda1<1e-10)?0:(lambda1/(lambda2+1e-20));

  BinaryLinearApplier *appl = new BinaryLinearApplier;
  appl->allocate(trn);
  appl->threshold=theta;


  if (verbose_level>=1) {
    printTrainingParameters(cerr);
    if (fista_eta==0) {
      fista_eta=1.0;
      switch (loss_type) {
      case LinearTrainer::SVM:
	cerr << "!!! warning: SVM-loss is nonsmooth !!!" <<endl;
	fista_eta=0.025;
	break;
      case LinearTrainer::SM_HINGE:
	fista_eta= loss_gamma;
	break;
      case LinearTrainer::LOGISTIC:
	fista_eta=4.0;
      default:
	break;
      }
      cerr << "automatically set eta to 1/loss-smoothness" <<endl;
    }
    cerr << "  using FISTA with learning rate eta=" << fista_eta <<endl;
    cerr <<endl;
  }


  GradCalc grad(*this,appl,trn,labels,targets,update_b);

  double startTime=clock();
  double cpuTime=0;
  
  
  double t0, t=1.0;
  double beta;
  // doing fista iterations
  for (int it=0; it<iters; it++) {
    t0=t;
    t=0.5*(1+sqrt(1+4*t*t));
    beta=(t0-1)/t;
    startTime=clock();
    grad.fistaIter(fista_eta,beta);
    cpuTime +=clock()-startTime;

    if (it%chk_interval==0 && verbose_level>=2) {
      double pobj=grad.pobj();
      cerr << "-- epoch " << it+1 << ": (p="<<pobj <<",d=na"<<")"<< endl;
    }

    if (verbose_level>=3 && tst && it%chk_interval==0) {
      if (labels) {// classification
	BinaryTestStat bts;
	for (int i=0; i<tst->size(); i++) {
	  float v=appl->apply(tst->ldps[i]);
	  bts.update(tst_labels[i], v>appl->threshold);
	}
	cerr << "test set performance: " <<endl;
	bts.print(cerr, " ");
      }
      else { // regression
	float * targets= tst_targets;
	double se=0;
	for (int i=0; i<tst->size(); i++) {
	  double err=targets[i]-appl->apply(tst->ldps[i]);
	  se += err*err;
	}
	cerr << " root mean squared error = " << sqrt(se/(float)tst->size()) 
	     << endl;
      }
    }
  }

  if (verbose_level>=1) {
    cpuTime /= (CLOCKS_PER_SEC);
    cerr << " cputime = " << cpuTime << endl;
  }

  if (loss_type==LOGISTIC) {
    appl->prob.type=BinaryLinearApplier::ProbabilityModel::EXP;
  }
  return appl;
}


