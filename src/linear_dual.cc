/********************************************************************
 *  File: linear_dual.cc
 *  Copyright (C) 2012, 2013 Tong Zhang 
 *
 *  Description: Proximal SDCA algorithm for sparse binary linear classifiers with L1-L2 regularization
 *
 ********************************************************************/

#include "linear_trainer.hh"


class AlphCalc : public AlphCalcBasic {
public:

  /** whether should shrink the weight (used when modified sgd is applied) */
  double shr_fact;
  /** use for modified sgd --- use flat lambda/learning rate for the first burnIn steps */
  int burnIn;

  AlphCalc(LinearTrainer _trn, BinaryLinearApplier * _appl,
	   LinearDataSet & _lds,  bool *_labels, float *_targets,
	   bool _update_b)
    : AlphCalcBasic(_trn,_appl,_lds, _labels,_targets,_update_b) , shr_fact(1.0)
  {
    eps=trn.eps;
    lambda= num*trn.lambda2;
    pC2=1.0/lambda;
    initIx2();
  }

  
  /**
   * assign weights to sum new_alpha*x
   */ 
  void assignWeights(double * new_alpha) {
    weightVec->setZero();
    for (int k=0; k<num; k++) {
      LinearDataPoint ldp=ldps[k];
      double del=new_alpha[k];
      if (labels) { // classification
	if (!labels[k]) del = -del;
      }
      weightVec->saxpy(ldp,del,update_b);
    }
  }

  /**
   * compute the linear weight dot the k-th datum
   */
  double computeSum_modSGD(int k, int n0) {
    register double eps0=eps*max(n0,burnIn)/(double)num;
    double sum=weightVec->trunc_dot(ldps[k],eps0)*shr_fact;
    return labels?((labels[k])?sum:-sum):sum;
  }

  /**
   * update weight for the modified sgd iterations with changing lambda handled by shr_fact:
   *    first burnIn iterations use trn.lambda2 /burnIn
   *    followed by trn.lambda2 / n0 -- where n0 is the iteration no.
   */
  void updateWeight_modSGD(int k, int n0, double del) {
    if (del<1e-10 && del>-1e-10) del=0;

    alpha[k] =del;

    if (labels) { // classification
      if (!labels[k]) del=-del;
    }
    del = del *num/max(n0,burnIn);
    if (n0>burnIn) {
      double shr=(n0-1.0)/n0;
      shr_fact *=shr;
      del /=shr_fact;
    }

    weightVec->saxpy(ldps[k],del,update_b);
  }

  /**
   * update weight for the modified sgd iterations with changing lambda handled by shr_fact:
   *    first burnIn iterations use trn.lambda2 /burnIn
   *    followed by trn.lambda2 / n0 -- where n0 is the iteration no.
   */
  void updateAlpha_modSGD(int k, int n0) {
    double sum=computeSum_modSGD(k,n0);
    double dalph;
    double rr=lambda*max(n0,burnIn)/num;
    if (labels) { // classification
      switch(trn.loss_type) {
      case LinearTrainer::SVM:
	dalph=(1.0-sum)*ix2[k];
	if (dalph<0) dalph=0;
	if (dalph>pC2) dalph=pC2;
	break;
      case LinearTrainer::SM_HINGE:
	dalph= (1.0-sum)/(rr*trn.loss_gamma+ix2[k]);
	if (dalph<0) dalph=0;
	if (dalph>pC2) dalph=pC2;
	break;
      case LinearTrainer::MOD_LS:
	dalph= (1.0-sum)/(rr+ix2[k]);
	if (dalph<0) dalph=0;
	break;
      case LinearTrainer::LS:
	dalph= (1.0-sum)*ix2[k]/(rr+ix2[k]);
	break;
      default: // LinearDualTrainer::LOGISTIC
	double rr=lambda*max(n0,burnIn);
	dalph= 1.0/(1.0+exp(sum))/(rr+0.25*ix2[k]);
	dalph += 1.0/(1.0+exp(sum+dalph*ix2[k])-rr*dalph)/(rr+0.25*ix2[k]);
      }
    }
    else { // regression
      dalph= (targets[k]-sum)/(rr+ix2[k]);
    }

    updateWeight_modSGD(k,n0,dalph);
  }

  /** compute duality gap with the current loss function, and
   *  primal w, and dual alpha 
   *  pobj: primal objective value
   *  dobj: dual objective value
   *  return: duality gap
   */
  double duality_gap(double &pobj, double &dobj)
  {
    // primal value
    double p=0;
    // negative dual value
    double d=0;
    double gap=0;

    int i,k;
    double sum, xi;

    pobj=0;
    dobj=0;
    // go through data
    for (k=0; k<num; k++) {
      sum=computeSum(k);
      xi=alpha[k]*lambda;

      switch(trn.loss_type) {
      case LinearTrainer::SVM:
	p = ((sum>1)?0:(1-sum));
	d = (-xi);
	break;
      case LinearTrainer::SM_HINGE:
	p = ((sum>=1)?0:((sum<=1-trn.loss_gamma)?(1-sum-0.5*trn.loss_gamma):.5*(1-sum)*(1-sum)/trn.loss_gamma));
	d = (0.5*trn.loss_gamma*xi*xi-xi);
	break;
      case LinearTrainer::MOD_LS:
	p = ((sum>1)?0:.5*(1-sum)*(1-sum));
	d = (0.5*xi*xi-xi);
	break;
      case LinearTrainer::LS:
	if (labels) { // classification
	  p = (.5*(1-sum)*(1-sum));
	  d = (0.5*xi*xi-xi);
	}
	else {  // regression
	  p = (.5*(targets[k]-sum)*(targets[k]-sum));
	  d = (0.5*xi*xi-targets[k]*xi);
	}
	break;
      default: // LinearDualTrainer::LOGISTIC
	p = log(1.0+exp(-sum));
	d = ((xi)*log(xi+1e-12)+(1.0-xi)*log(1.0-xi+(1e-12)));
      }
      pobj+=p;
      dobj-=d;
    }

    // regularization
    register ParamVector w=*weightVec;
    double r;
    double rp;
    if (eps<1e-10) {
      r=w.b*w.b;
      for (i=0; i<w.numFloatFeatures; i++) {
	register double ww=weightVec->floatVec[i];
	r +=ww*ww;
      }
      for (i=0; i<w.numBinaryFeatures; i++) {
	register double ww=weightVec->binaryVec[i];
	r +=ww*ww;	
      }
      rp=r;
    }
    else {
      rp=w.b*w.b;
      r=0;
      for (i=0; i<w.numFloatFeatures; i++) {
	register double ww=w.floatVec[i];
	ww=ww>eps?ww-eps:(ww<-eps?-(ww+eps):0);
	r += ww;
	rp +=ww*ww;
      }
      for (i=0; i<w.numBinaryFeatures; i++) {
	register double ww=w.binaryVec[i];
	ww=ww>eps?ww-eps:(ww<-eps?-(ww+eps):0);
	r += ww;
	rp +=ww*ww;
      }
      r = 2*eps*r+rp;
    }
    pobj=(pobj/lambda+0.5*r)*lambda/num;
    dobj=(dobj/lambda-0.5*rp)*lambda/num;
    gap=pobj-dobj;
    return gap;
  }

};





BinaryLinearApplier * LinearTrainer::train_sdca
(LinearDataSet & trn, bool * labels, float * targets,
 LinearDataSet * tst, bool * tst_labels, float * tst_targets, int verbose_level)
{

  if (eps<0) eps=0;

  // checking max feature size and allocate array for appl
  BinaryLinearApplier *appl = new BinaryLinearApplier;
  appl->allocate(trn);
  appl->threshold=theta;

  int num=trn.size();

  use_sgd=use_sgd0;
  use_avg=use_avg0;

  if (use_sgd<0) {
    if (lambda2*num>=10) use_sgd=1;
    else use_sgd=0;
  }
  if (use_avg<0 || use_avg>2) {
    use_avg=2;
  }
  eps = (lambda1<1e-10)?0:(lambda1/(lambda2+1e-20));
  if (eps>1e-10 && use_avg) {
    use_avg=0;
    if (use_avg0>0) {
      cerr << "averaging is currently not supported when lambda1 >0 " <<endl;
    }
  }
  if (verbose_level>=1) {
    printTrainingParameters(cerr);
    cerr << "  using prox-SDCA" <<endl;
    cerr << "  min duality gap=" << min_dgap << endl;
    cerr << "  check duality gap every " << chk_interval << " epochs" <<endl;
    cerr << "  initialize with modified sgd=" << (use_sgd?"true":"false") <<endl;
    cerr << endl;
  }


  double *alpha_avg=0;
  if (use_avg) {
    alpha_avg= new double [num];
    if (use_sgd<=0) {
      for (int i=0; i<num; i++) alpha_avg[i]=0;
    }
  }

  AlphCalc alph(*this,appl,trn,labels,targets,update_b);

  // generate random ordering
  srand48(27432042);
  int * order = new_order(num);

  double startTime=clock();
  double cpuTime=0;
  
  int it =0;
  double dg=1e10;
  double pobj0=1e10, dobj0=-1e10;
  double pobj1=1e10, dobj1=-1e10;
  
  bool need_restore=false;

  if (use_sgd>0) {
    // initialize with modified sgd
    alph.burnIn=use_sgd;
    if (use_sgd<num/100 && use_sgd<1/(lambda2+1e-10)) alph.burnIn=num/100;
    permutation(order,num);
    int i,j;
    for (j=0; j<num; j++) {
      int k=order[j];
      alph.updateAlpha_modSGD(k,j+1);
      if (use_avg) {
	alpha_avg[k]=alph.alpha[k];
      }
    }
    cpuTime +=clock()-startTime;
    if (use_avg) {
      double fr=0;
      for (int kk=num; kk>=1; kk--) {
	if (alph.burnIn<=kk) fr +=1.0/kk;
	else fr +=1.0/alph.burnIn;
	alpha_avg[order[kk]]*=fr;
      }
    }

    startTime=clock();

    ParamVector *wp=& appl->weightVec;

    if (update_b) wp->b *=alph.shr_fact;

    for (i=0; i<wp->numFloatFeatures; i++) {
      wp->floatVec[i] *=alph.shr_fact;
    }
    for (i=0; i<wp->numBinaryFeatures; i++) {
      wp->binaryVec[i] *=alph.shr_fact;
    }
    alph.shr_fact=1.0;
    
    cpuTime += clock()-startTime;

    if (min_dgap>0) {
      dg=alph.duality_gap(pobj0,dobj0);
      if (verbose_level>=2) {
	cerr << "-- epoch " << it+1 << ": duality gap=" << dg  << "(p="<<pobj0 <<",d="<<dobj0<<")"<< "  ";
      }

      if (use_avg) {
	// compute averaged weigthts
	alph.assignWeights(alpha_avg);
	
	dg=alph.duality_gap(pobj1,dobj1);
	dobj1=dobj0;
	dg=pobj1-dobj0;
	if (verbose_level>=2) {
	  cerr <<  " avg: duality gap=" << dg  << "(p="<<pobj1 << ")"<< endl;
	}

	// need to restore non-averaged weights
	need_restore=true;
      }
      else {
	cerr << endl;
      }
    }

    if (verbose_level>=3 && tst && min_dgap>0) {
      if (labels) {// classification
	BinaryTestStat bts;
	for (int i=0; i<tst->size(); i++) {
	  bts.update(tst_labels[i],
		     appl->weightVec.trunc_dot(tst->ldps[i],eps));
	}
	cerr << "test set performance: " <<endl;
	bts.print(cerr,"  ");
      }
      else { // regression
	float * targets= tst_targets;
	double se=0;
	for (int i=0; i<tst->size(); i++) {
	  double err=targets[i]-appl->weightVec.trunc_dot(tst->ldps[i],eps);
	  se += err*err;
	}
	cerr << " root mean squared error = " << sqrt(se/(float)tst->size()) 
	     << endl;
      }
    }
    it++;
    startTime=clock();
  }

  alph.updateIx2();
  // doing regular dual coordinate ascent
  for (; it<iters; it++) {
    dg=min(pobj0,pobj1)-dobj0;
    if (use_avg==1) {
      dg=pobj1-dobj0;
    }
    if (dg>=0 && dg<min_dgap) break;

    if (need_restore) {
      alph.assignWeights(alph.alpha);
    }
    need_restore=false;
    permutation(order,num);
    int j;
    for (j=0; j<num; j++) {
      int k=order[j];
      double u1=min(0.8,sqrt(it)/(1.0+sqrt(it)));
      double u2=1.0-j/(double)num;
      if (use_avg) {
	alpha_avg[k]=u1*alpha_avg[k]+(1-u1)*(1-u2)*alph.alpha[k];
      }
      alph.updateAlpha(k);
      if (use_avg) {
	alpha_avg[k] += (1-u1)*u2*alph.alpha[k];
      }
    }
    cpuTime +=clock()-startTime;
    if (min_dgap>0 && it%chk_interval==0) {
      dg=alph.duality_gap(pobj0,dobj0);
      if (verbose_level>=2) {
	cerr << "-- epoch " << it+1 << ": duality gap=" << dg  << "(p="<<pobj0 <<",d="<<dobj0<<")"<< "  ";
      }
      
      if (use_avg) { 
	// compute avg weight
	alph.assignWeights(alpha_avg);

	dg=alph.duality_gap(pobj1,dobj1);
	dg=pobj1-dobj0;
	if (verbose_level>=2) {
	  cerr <<  " avg: duality gap=" << dg  << "(p="<<pobj1 <<")"<< endl;
	}
      
	// need to restore non-avg weight
	need_restore=true;
      }
      else {
	cerr << endl;
      }
    }      
    if (verbose_level>=3 && tst && min_dgap>0 && it%chk_interval==0) {
      if (labels) {// classification
	BinaryTestStat bts;
	for (int i=0; i<tst->size(); i++) {
	  float v=appl->weightVec.trunc_dot(tst->ldps[i],eps);
	  bts.update(tst_labels[i], v>appl->threshold);
	}
	cerr << "test set performance: " <<endl;
	bts.print(cerr, " ");
      }
      else { // regression
	float * targets= tst_targets;
	double se=0;
	for (int i=0; i<tst->size(); i++) {
	  double err=targets[i]-appl->weightVec.trunc_dot(tst->ldps[i],eps);
	  se += err*err;
	}
	cerr << " root mean squared error = " << sqrt(se/(float)tst->size()) 
	     << endl;
      }
    }
    startTime=clock();
  }

  if ((use_avg==2) && (pobj0<pobj1) && need_restore) {
    if (verbose_level>=2) {
      cerr << "final solution uses non-averaged weights" << endl;
    }
    alph.assignWeights(alph.alpha);
    need_restore=false;
  }
  else {
    if (need_restore && (verbose_level>=2)) {
      cerr << "final solution uses averaged weights" << endl;
    }
  }

  delete [] order;
  if (use_avg) {
    delete [] alpha_avg;
  }

  alph.destroy();

  // compute sparse weights;
  appl->weightVec.trunc(eps);

  if (verbose_level>=1) {
    cpuTime +=clock()-startTime;
    cpuTime /= (CLOCKS_PER_SEC);
    cerr << " cputime = " << cpuTime << endl;
  }

  if (loss_type==LOGISTIC) {
    appl->prob.type=BinaryLinearApplier::ProbabilityModel::EXP;
  }
  return appl;
}


