/********************************************************************
 *  File: linear_trainer.cc
 *  Copyright (C) 2012, 2013 Tong Zhang
 *
 *  Description: Common Routines for Linear classifier trainers
 *
 ********************************************************************/

#include "linear_trainer.hh"

static int lossFun(char * loss, double & gamma) {
  if (!strcmp(loss,"SVM")) 
    return LinearTrainer::SVM;
  else if (!strcmp(loss,"LS")) 
    return LinearTrainer::LS;
  else if (!strcmp(loss,"MOD_LS")) 
    return LinearTrainer::MOD_LS;
  else if (!strcmp(loss,"LOGISTIC")) 
    return LinearTrainer::LOGISTIC;
  else if (!strncmp(loss,"SmoothHinge,",strlen("SmoothHinge,"))) {
    gamma=atof(loss+strlen("SmoothHinge,"));
    return LinearTrainer::SM_HINGE;
  }
  return -1;
}

static const char *lossFun(LinearTrainer::loss_t type)
{
  if (type==LinearTrainer::SVM) 
    return "SVM";
  else if (type==LinearTrainer::LS) 
    return "LS";
  else if (type==LinearTrainer::MOD_LS) 
    return "MOD_LS";
  else if (type==LinearTrainer::SM_HINGE) 
    return "Smoothed Hinge";
  return "LOGISTIC";
}

static char * parse(char *str, const char *arg) {
  if (!strncmp(str,arg,strlen(arg))) { 
    return (str+strlen(arg));
  }
  return 0;
}


void LinearTrainer::printTrainingParameters(ostream & os)
{
  os <<  " training parameters : " <<endl;
  os << "  loss function = " << lossFun(loss_type);
  if (loss_type==LinearTrainer::SM_HINGE) os << " (gamma=" << loss_gamma << ")";
  os <<endl;
  os << "  regularization= " << lambda1 <<" * |w| + 0.5 * " << lambda2 << "* w*w" <<endl;
  os << "  max iterations (epochs)     =" << iters <<endl;
  if (update_b) {
    os << "  include intercept (bias)" <<endl;
  }
  else {
    os << "  exclude intercept (bias)" <<endl;
  }
  os <<endl;

    os << "  using Accelerated-prox-SDCA with inner iters=" << accl_iters <<endl;
}



void LinearTrainer::printCommandLineOptions(ostream & os)
{
  os << "   -lambda1=lambda1" <<endl;
  os << "     set L1 regularization parameter (default=0)"
     << endl;
  os << "   -lambda2=lambda2"<<endl;
  os << "     set L2 regularization parameter (default=1e-4)" <<endl;
  os << "     regularization term:  lambda1 |w| + 0.5 lambda2 w*w"
     <<endl;
  os << "   -max.iters=max_iterations" <<endl;
  os << "     run dual algorithms no more than max_iterations (epochs) over the training data."
     << endl;
  os << "   -chk.interval=m" <<endl;
  os << "     check and report duality gap every m epochs (default=" << chk_interval0 << ")."
     << endl;
  os << "   -loss=[SVM|SmoothHinge,gamma|MOD_LS|LOGISTIC|LS]" <<endl;
  os << "     set loss function" <<endl;
  os << "     SmoothHinge: smoothed hinge loss with 0.5 gamma * alpha^2 added to the dual of SVM" <<endl;
  os << "     SVM: hinge loss (SmoothHinge with gamma=0)" << endl;
  os << "     MOD_LS: squared hinge loss" <<endl;
  os << "     LOGISTIC: logistic regression" << endl;
  os << "     LS: least squares (classification or regression)" <<endl;
  os << "   -use.intercept=[0,1]" <<endl;
  os << "     whether to use intercept (or bias)" <<endl;
  os << "   -threshold=theta" <<endl;
  os << "     set binary decision threshold to theta" << endl <<endl;
  os << "   === the following options are for prox-SDCA ===" <<endl;
  os << "   -min.dgap=min_duality_gap" <<endl;
  os << "     terminate when duality gap is smaller than min_duality_gap."
     << endl;
  os << "   -sgd.init=[0,1]" <<endl;
  os << "     whether to use modified sgd for the first epoch" << endl;
  os << endl;
  os << "   === the following option is for FISTA ===" <<endl;
  os << "   -fista.eta=eta (default<0)" <<endl;
  os << "     learning rate for FISTA" << endl;
  os << "     if >=0 use fista with learning rate eta; if =0 automatically set eta to 1/smoothness" <<endl;
  os << endl;
  os << "   === the following options are for accelerated-prox-SDCA ===" <<endl;
  os << "   -accl.iters=iters (default<0)" <<endl;
  os << "     use acclerated prox-SDCA with iters number of inner iterations" << endl;
  os << "     if >=0 use acceleration; if =0 automatically set" <<endl;
  os << "   -accl.kappa=kappa (default=0.0001)" <<endl;
  os << "     accleration reg parameter kappa (add 0.5 kappa *||w-y||_2^2 to the regularizer)" << endl;
  os << "     if <=0 automatically set to 10/data-size" <<endl;
  os << "   -accl.beta=beta (default=0.9)" <<endl;
  os << "     accleration momentum coefficient beta" <<endl;
  os << "     if <=0 automatically set to (1-a)/(1+a) with a=sqrt(lambda2/(lambda2+2*kappa))" << endl;
  os << endl;
}




bool LinearTrainer::commandLineParse(char *str)
{
  char *s;
  if ((s=parse(str,"-lambda1="))) {
    lambda1=atof(s);
  }
  else if ((s=parse(str,"-lambda2="))) {
    lambda2=atof(s);
  }
  else if ((s=parse(str,"-max.iters="))) {
    iters=atoi(s);
  }
  else if ((s=parse(str,"-min.dgap="))) {
    min_dgap=atof(s);
  }
  else if ((s=parse(str,"-chk.interval="))) {
    chk_interval=atoi(s);
    if (chk_interval<1) chk_interval=1;
  }
  else if ((s=parse(str,"-loss="))) {
    int v=lossFun(s,loss_gamma);
    if (v<0) return false;
    loss_type=(loss_t) v;
  }
  else if ((s=parse(str,"-sgd.init="))) {
    use_sgd0=atoi(s);
  }
  else if ((s=parse(str,"-threshold="))) {
    theta=atof(s);
  }
  else if ((s=parse(str,"-use.intercept="))) {
    update_b=atoi(s);
  }
  else if ((s=parse(str,"-fista.eta="))) {
    fista_eta=atof(s);
  }
  else if ((s=parse(str,"-accl.iters="))) {
    accl_iters=atoi(s);
  }
  else if ((s=parse(str,"-accl.kappa="))) {
    accl_kappa=atof(s);
  }
  else if ((s=parse(str,"-accl.beta="))) {
    accl_beta=atof(s);
  }
  else {
    return false;
  }
  return true;
}


BinaryLinearApplier * LinearTrainer::train
(LinearDataSet & trn, bool * labels, float * targets,
 LinearDataSet * tst, bool * tst_labels, float * tst_targets, int verbose_level)
{
  BinaryLinearApplier *appl;
  if (fista_eta>0) {
    appl=train_fista(trn,labels,targets,tst,tst_labels,tst_targets,verbose_level);
  }
  else if (accl_iters>0) {
    appl=train_accl(trn,labels,targets,tst,tst_labels,tst_targets,verbose_level);
  }
  else {
    appl=train_sdca(trn,labels,targets,tst,tst_labels,tst_targets,verbose_level);
  }
  return appl;
}


double AlphCalcBasic::deltaAlpha(double sum, int k) 
{
    double dalph;
    
    if (labels) { // classification
      switch(trn.loss_type) {
      case LinearTrainer::SVM:
	dalph= (1-sum)*ix2[k];
	if (dalph+alpha[k]<0) {
	  dalph=-alpha[k];
	}
	else {
	  if (dalph+alpha[k]>pC2) dalph=pC2-alpha[k];
	}
	break;
      case LinearTrainer::SM_HINGE:
	dalph= (1- trn.loss_gamma*lambda*alpha[k] -sum)*ix2[k];
	if (dalph+alpha[k]<0) {
	  dalph=-alpha[k];
	}
	else {
	  if (dalph+alpha[k]>pC2) dalph=pC2-alpha[k];
	}
	break;
      case LinearTrainer::MOD_LS:
	dalph= (1- lambda*alpha[k] -sum)*ix2[k];
	if (dalph+alpha[k]<0) {
	  dalph=-alpha[k];
	}
	break;
      case LinearTrainer::LS:
	dalph= (1- lambda*alpha[k] -sum)*ix2[k];
	break;
      default: // LinearDualTrainer::LOGISTIC
	dalph= (1.0/(1.0+exp(sum))-lambda*alpha[k])*ix2[k];
	dalph += (1.0/(1.0+exp(sum+dalph*ix2[k]))-lambda*(alpha[k]+dalph))*ix2[k];
	if (alpha[k]+dalph<=lambda*1e-10) dalph=lambda*1e-10-alpha[k];
	if (alpha[k]+dalph>=lambda*(1-1e-10)) dalph=(1.0/lambda-alpha[k]);
      }
    }
    else { // regression
      dalph= (targets[k]-lambda*alpha[k]-sum)*ix2[k];
    }
    return dalph;
}


void AlphCalcBasic::initIx2()
{
  // compute ix2
  int i;
  for (i=0; i<num; i++) {
    register LinearDataPoint ldp=ldps[i];
    register int j;
    double tmp=ldp.numBinaryFeatures;
    for (j=0; j<ldp.numFloatFeatures; j++) {
      tmp += ldp.floatArray[j].value*ldp.floatArray[j].value;
    }
    tmp=(update_b?1.0:1e-5)+ldp.norm_factor*ldp.norm_factor*tmp;
    if (trn.loss_type==LinearTrainer::SVM) {
      ix2[i] = 1.0/tmp;
    }
    else {
      ix2[i]=tmp;
    }
  }
}

void AlphCalcBasic::updateIx2()
{
  if (trn.loss_type == LinearTrainer::SVM) return;
  double a0=(trn.loss_type == LinearTrainer::LOGISTIC)?0.25:1.0;
  for (int i=0; i<num; i++) {
    ix2[i]=1.0/(lambda+a0*ix2[i]);
  }
}


int * new_order(int num)
{
  int * order = new int [num];
  for (int i=0; i<num; i++) {
    order[i]=i;
  }
  return order;
}

/** compute permutation */
void permutation(int * order, int num)
{
  int i,j;
  for (i=0; i<num; i++) {
    j=i + (int)((num-i)*drand48());
    if (j>=num) j=num-1;
    if (j>i) {
      int tmp=order[i];
      order[i]=order[j];
      order[j]=tmp;
    }
  }
}
