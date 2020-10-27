/********************************************************************
 *  File: linear_accl.cc
 *  Copyright (C) 2012, 2013 Tong Zhang
 *
 *  Description: Accelerated prox-SDCA algorithm for sparse binary linear classifiers with L1-L2 regularization
 *
 ********************************************************************/

#include "linear_trainer.hh"


class AlphCalc2 : public AlphCalcBasic {
public:
  /** adding additional regularizer 0.5 kappa || w-y ||_2^2 */
  double kappa;


  AlphCalc2(LinearTrainer _trn, BinaryLinearApplier *_appl, double _kappa, 
	   LinearDataSet & _lds,  bool *_labels, float *_targets,
	   bool _update_b) 
    : AlphCalcBasic(_trn,_appl,_lds,_labels,_targets,_update_b) , kappa(_kappa)
  {   
    eps=trn.lambda1/(trn.lambda2+kappa);
    lambda= num*(trn.lambda2+kappa);
    pC2=1.0/lambda;

    initIx2();
    updateIx2();
  }


  /** compute primal objective value
   *  primal w
   *  pobj: primal objective value
   *  return: primal objective
   */
  double pobj(ParamVector & x) 
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
      double eps=trn.lambda1/(1e-20+trn.lambda2);
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


  /**
   * solving the proximal problem using prox-SDCA
   *   min_x [ loss(x) + reg(x) + 0.5 kappa ||x-y||_2^2 ]
   *        where reg(x) = lambda1 ||x||_1 + 0.5 lambda2 ||x||_2^2
   *
   * @param iters number of prox-SDCA iterations
   * @param x  solution vector
   * @param y  previous solution
   * @param order for random permutation
   * @param init_with_previous_alpha whether to initialize with previous alpha
   * @return primal objective value
   *
   */
  double proximal(int iters, ParamVector & x, ParamVector &y, int * order, bool init_with_previous_alpha) {

    weightVec=&x;
    weightVec->setZero();
    weightVec->saxpy(y,kappa/(kappa+trn.lambda2));

    if (init_with_previous_alpha) {
      for (int j=0; j< num; j++) {
	double del=alpha[j];
	alpha[j]=0.0;
	updateWeight(j,del);
      }
    }
    else { // initialize with zero
      bzero(alpha,sizeof(double)*num);
    }

    for (int it=0; it<iters; it++) {
      permutation(order,num);
      for (int j=0; j<num; j++) {
	int k=order[j];
	updateAlpha(k);
      }
    }
    weightVec->trunc(eps);
    return pobj(x);
  }

    
};




BinaryLinearApplier * LinearTrainer::train_accl
(LinearDataSet & trn, bool * labels, float * targets,
 LinearDataSet * tst, bool * tst_labels, float * tst_targets, int verbose_level)
{

  // checking max feature size and allocate array for appl
  BinaryLinearApplier *appl = new BinaryLinearApplier;
  appl->allocate(trn);
  appl->threshold=theta;

  int num=trn.size();

  if (verbose_level>=1) {
    printTrainingParameters(cerr);
    if (accl_iters==0) {
      accl_iters=5;
      cerr << "    automatically set inner iterations" <<endl;
    }
    if (accl_kappa<=0) {
      accl_kappa=10.0/num;
      cerr << "    automatically set kappa to 10/data-size=" << accl_kappa << endl;
    }
    if (accl_beta<=0) {
      double accl_alpha= sqrt(lambda2/(lambda2+2*accl_kappa));
      accl_beta= (1.0-accl_alpha)/(1.0+accl_alpha);
      cerr << "    automatically set momentum cofficient beta to (1-a)/(1+a) with a=sqrt(lambda2/(lambda2+2*kappa))=" << accl_beta << endl;
    }
    cerr << "                       additional reg= 0.5 * " 
       << accl_kappa << " * ||w-y||_2^2" <<endl;
    cerr << "                       acceleration momentum cofficient beta= " << accl_beta
       <<endl;
    cerr << "                       number of prox-SDCA inner iterations= " << accl_iters <<endl;
    cerr <<endl;
  }


  // generate random ordering
  srand48(27432042);
  int * order = new_order(num);
  permutation(order,num);
  

  ParamVector x0, x, y;
  
  x=appl->weightVec;
  x.setZero();
  x0.allocate(x.numFloatFeatures,x.numBinaryFeatures);
  y.allocate(x.numFloatFeatures,x.numBinaryFeatures);

  AlphCalc2 alph(*this,appl,accl_kappa, trn,labels,targets,update_b);
  for (int it=0; it<iters; it++) {
    x0.copyFrom(x);
    double pobj = alph.proximal(accl_iters,x,y,order,(it>0));
    y.scaled_add((1+accl_beta),x,-accl_beta,x0);
    cerr << "-- epoch " << it+1 << ": (p="<<pobj <<",d=na)"<< endl;
  }
  delete [] order;
  
  alph.destroy();

  appl->weightVec=x;
  x0.destroy();
  y.destroy();

  if (loss_type==LOGISTIC) {
    appl->prob.type=BinaryLinearApplier::ProbabilityModel::EXP;
  }
  return appl;
}


