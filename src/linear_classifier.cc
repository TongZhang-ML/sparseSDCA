/********************************************************************
 *  File: linear_classifier.cc
 *  Date: Aug 2012
 *  Author: Tong Zhang
 *  Description: Data structure for linear classifiers
 *
 ********************************************************************/

#include "linear_classifier.hh"


LinearDataPoint::LinearDataPoint(int _numFloatFeatures, int *_flt_ind, float * _flt_val,
				 int _numBinaryFeatures, int *_bin_ind, float _norm_fact)
{
  numFloatFeatures=_numFloatFeatures;
  floatArray=0;
  int i;

  numBinaryFeatures=_numBinaryFeatures;
  binaryIndexArray=0;
  if (numBinaryFeatures>0) {
    binaryIndexArray = new int [numBinaryFeatures];
    for (i=0; i<numBinaryFeatures; i++) {
      binaryIndexArray[i]=_bin_ind[i];
    }
  }

  if (numFloatFeatures>0) {
    floatArray = new  FltArray [numFloatFeatures];
    for (i=0; i<numFloatFeatures; i++) {
      floatArray[i].index=_flt_ind[i];
      floatArray[i].value=_flt_val[i];
    }
  }
  normalize(_norm_fact);
}

void LinearDataPoint::normalize(double norm)
{
  if (norm<=0) {
    norm_factor=1.0;
    return;
  }
  double ff=1e-10 + numBinaryFeatures;
  int j;
  for (j=0; j<numFloatFeatures; j++) {
    ff += floatArray[j].value*floatArray[j].value;
  }
  norm_factor=norm/sqrt(ff);  
}

int LinearDataPoint::max_float_index()
{
  int maxfeat=0;
  for (int i=0; i< numFloatFeatures; i++) {
    if (maxfeat <floatArray[i].index) maxfeat=floatArray[i].index;
  }
  return maxfeat;
}

int LinearDataPoint::max_binary_index() 
{
  int maxfeat=0;
  for (int i=0; i< numBinaryFeatures; i++) {
    if (maxfeat <binaryIndexArray[i]) maxfeat=binaryIndexArray[i];
  }
  return maxfeat;
}

void LinearDataPoint::destroy()
{
  if (floatArray) delete [] floatArray;
  floatArray=0;
  if (binaryIndexArray) delete [] binaryIndexArray;
  floatArray=0;
  numFloatFeatures=0;
  numBinaryFeatures=0;
  norm_factor=1.0;
}


void LinearDataSet::destroy()
{
  for (int i=0; i<num; i++) {
    ldps[i].destroy();
  }
  if (ldps) delete [] ldps;
  clear();
}

double BinaryLinearApplier::apply(LinearDataPoint &ldp) 
{
  return weightVec.dot_with_boundary_check(ldp);
}

static void write_arr(ostream & os, linfloat_t *arr, int num)
{
  int nz=0;
  int i;
  for (i=0; i<num; i++) {
    if (arr[i]>1e-5 || arr[i]<-1e-5) nz++;
  }
  if (nz*2.5>num) nz=-1;

  os << nz << ' ';
  if (nz>=0) {
    for (i=0; i<num; i++) {
      if (arr[i]>1e-5 || arr[i]<-1e-5) {
	os << i << ' ' << arr[i] << ' ';
      }
    }
  }
  else {
    for (i=0; i<num; i++) {
      os << arr[i] << ' ';
    }
  }
}

static void read_arr(istream & is, linfloat_t *arr, int num)
{
  int nz;
  is >> nz;
  int i;
  if (nz>=0) {
    for (i=0; i<num; i++) arr[i]=0.0;
    for (i=0; i<nz; i++) {
      int ii;
      is >> ii;
      is >> arr[ii];
    }
  }
  else {
    for (i=0; i<num; i++) {
      is >> arr[i];
    }
  }
}


void BinaryLinearApplier::write(ostream & os)
{
  os << threshold << ' ';
  os << prob.type << ' ';

  weightVec.write(os);
}

void BinaryLinearApplier::read(istream & is)
{
  destroy();

  is >> threshold;
  int v;
  is >> v; 
  prob.type=(ProbabilityModel::prob_t) v;

  weightVec.read(is);
}


void BinaryLinearApplier::destroy()
{
  weightVec.destroy();
}

void BinaryLinearApplier::allocate(LinearDataSet & trn) 
{
  int numFloatFeatures =1;
  int numBinaryFeatures =1;
  int i;
  for (i=0; i< trn.size(); i++) {
    int ft;
    ft= trn.ldps[i].max_float_index();
    if (numFloatFeatures<=ft) numFloatFeatures=ft+1;
    ft= trn.ldps[i].max_binary_index();
    if (numBinaryFeatures<=ft) numBinaryFeatures=ft+1;
  }
  weightVec.allocate(numFloatFeatures,numBinaryFeatures);
}

void LinearDataReader::push_label(double vv)
{
  int v;
  v=(int)(vv+0.50);
  label_vec.push_back(v);
  target_vec.push_back(vv);
}

void LinearDataReader::read_labels(istream & is)
{
  double vv;
  while (1) {
    is >>vv;
    if (is.eof()) break;
    push_label(vv);
  }
}


static int getline(istream & is, vector<char> & line)
{
  int nc=0;
  line.clear();
  while (1) {
    char c;
    c=is.get();
    if (is.eof()) {
      nc=-1;
      break;
    }
    if (c=='\n') break;
    line.push_back(c);
    nc++;
  }
  line.push_back(0);
  return nc;
}

void LinearDataReader::read_features(istream & is, bool read_label, float norm_fact)
{
  vector<int> flt_index;
  vector<float> flt_value;
  vector<int> bin_index;

  vector<char> line;
  int nc;
  while ((nc=getline(is,line))>=0) {
    flt_index.clear();
    flt_value.clear();
    bin_index.clear();

    int cpos=0, npos=0;
    bool is_label=read_label;
    while (cpos< nc-1) {
      // get rid of initial spaces
      if (isspace(line[cpos])) {
	cpos++;
	continue;
      }
      // find the next token
      npos=cpos;
      while (npos<nc&& !isspace(line[npos])) npos++;
      line[npos]=0;
      char * token= & line[cpos];
      
      /** whether the first token is label: if so read */
      if (is_label) {
	push_label(atof(token));
	cpos=npos+1;
	is_label=false;
	continue;
      }

      // check token format
      int i=0;
      for (i=0; i<npos-cpos; i++) {
	if (token[i]==':' || token[i]=='@') break;
      }
      if (i>0 && i<npos-cpos) {
	token[i]=0;
	flt_index.push_back(atoi(token));
	flt_value.push_back(atof(token+i+1));
      }
      else {
	  bin_index.push_back(atoi(token));
      }
      cpos=npos+1;
    }
    LinearDataPoint ldp(flt_index.size(),&flt_index[0],&flt_value[0],
			bin_index.size(),&bin_index[0],norm_fact);
    ldp_vec.push_back(ldp);
  }
}


int LinearDataReader::max_label()
{
  int ml=0;
  for (unsigned int i=0; i<label_vec.size(); i++) {
    if (label_vec[i]>ml) ml=label_vec[i];
  }
  return ml;
}


bool LinearDataReader::append_data(char * label_fn, char * feat_fn, float norm_fact)
{
  bool read_label=(label_fn==0);
  {
    ifstream is(feat_fn);
    if (!is.good()) {
      cerr << "cannot open feature file <" << feat_fn << ">" <<endl;
      exit(-1);
    }
    read_features(is,read_label,norm_fact);
    is.close();
  }
  if (!read_label) {
    ifstream is(label_fn);
    if (!is.good()) {
      cerr << "cannot open label file <" << label_fn << ">: assign zero values" <<endl;
      while (label_vec.size() < ldp_vec.size()) {
	label_vec.push_back(0);
      }
    }
    else {
      read_label=true;
      read_labels(is);
    }
    is.close();
  }
  if (label_vec.size() != ldp_vec.size()) {
    cerr << "error : number of labels does not match number of features" <<endl;
    cerr << "labels = " << label_vec.size() 
	 << ", features= " << ldp_vec.size() <<endl;
    exit(-1);
  }
  if (binaryLabels) delete [] binaryLabels;
  binaryLabels = new bool [size()];
  return read_label;
}

LinearDataReader::~LinearDataReader()
{
  if (binaryLabels) delete [] binaryLabels;
  for (unsigned int i=0; i<ldp_vec.size(); i++) {
    ldp_vec[i].destroy();
  }
}

int MultiLinearClassifier::classify(LinearDataPoint & ldp, double &scr) 
{
  int bl=0;
  scr=appl_vec[0]->apply(ldp);
  for (unsigned int cl=1; cl<appl_vec.size(); cl++) {
    double my_scr=appl_vec[cl]->apply(ldp);
    if (scr<my_scr) {
      scr=my_scr;
      bl=cl;
    }
  }
  return bl;
}

void MultiLinearClassifier::write(ostream & os) 
{
  os << appl_vec.size() << ' ';
  for (unsigned int cl=0; cl<appl_vec.size(); cl++) {
    appl_vec[cl]->write(os);
  }
}

void  MultiLinearClassifier::read(istream & is) 
{
  int nn;
  is >> nn;
  for (int cl=0; cl<nn; cl++) {
    appl_vec.push_back(new BinaryLinearApplier());
    appl_vec[cl]->read(is);
  }
}


void MultiLinearClassifier::destroy() 
{
  for (unsigned int cl=0; cl<appl_vec.size(); cl++) {
    appl_vec[cl]->destroy();
    delete appl_vec[cl];
  }
  appl_vec.clear();
}


void ParamVector::allocate(int nf, int nb) 
{
  clear();
  numFloatFeatures=nf;
  numBinaryFeatures=nb;
  floatVec=0;
  binaryVec=0;
  
  if (numFloatFeatures>0) {
    floatVec = new linfloat_t [numFloatFeatures];
    bzero(floatVec,numFloatFeatures*sizeof(linfloat_t));
  }
  if (numBinaryFeatures>0) {
    binaryVec= new linfloat_t [numBinaryFeatures];
    bzero(binaryVec,numBinaryFeatures*sizeof(linfloat_t));
  }
  setZero();
}

void ParamVector::clear() 
{
  numFloatFeatures=0;
  numBinaryFeatures=0;
  floatVec=0;
  binaryVec=0;
  b=0;
}

void ParamVector::setZero() 
{
  b=0;
  if (numFloatFeatures>0)
    bzero(floatVec,numFloatFeatures*sizeof(linfloat_t));
  if (numBinaryFeatures>0)
    bzero(binaryVec,numBinaryFeatures*sizeof(linfloat_t));
}

void ParamVector::saxpy(ParamVector & src, double a) 
{
  b += a * src.b;
  int i;
  for (i=0; i<numFloatFeatures; i++) {
    floatVec[i] += a* src.floatVec[i];
  }
  for (i=0; i<numBinaryFeatures; i++) {
    binaryVec[i] += a* src.binaryVec[i];
  }
}

void ParamVector:: saxpy(LinearDataPoint & ldp, double del, bool update_b)
{
  if (update_b) b += del;

  del *= ldp.norm_factor;
  int i;
  for (i=0; i<ldp.numFloatFeatures; i++) {
    floatVec[ldp.floatArray[i].index] += del * ldp.floatArray[i].value;
  }
  for (i=0; i<ldp.numBinaryFeatures; i++) {
    binaryVec[ldp.binaryIndexArray[i]] += del;
  }
}

void ParamVector::scaled_add(double alpha, ParamVector &x, double beta, ParamVector &y)
{
  b= alpha*x.b+ beta*y.b;
  int i;
  for (i=0; i<numFloatFeatures; i++) {
    floatVec[i] = alpha*x.floatVec[i]+beta*y.floatVec[i];
  }
  for (i=0; i<numBinaryFeatures; i++) {
    binaryVec[i] = alpha*x.binaryVec[i]+beta*y.binaryVec[i];
  }
}


void ParamVector::copyFrom(ParamVector & src) 
{
  assert(src.numFloatFeatures==numFloatFeatures && src.numBinaryFeatures==numBinaryFeatures);
  
  b=src.b;
  int i;
  for (i=0; i<numFloatFeatures; i++) {
    floatVec[i] = src.floatVec[i];
  }
  for (i=0; i<numBinaryFeatures; i++) {
    binaryVec[i] = src.binaryVec[i];
  }
}


double ParamVector::dot(LinearDataPoint & ldp)
{
  double sum=0;
  register int i;
  for (i=0; i<ldp.numFloatFeatures; i++) {
    sum += floatVec[ldp.floatArray[i].index]*ldp.floatArray[i].value;
  }
  for (i=0; i<ldp.numBinaryFeatures; i++) {
    sum += binaryVec[ldp.binaryIndexArray[i]];
  }
  sum = sum*ldp.norm_factor+b;
  return sum;
}

double ParamVector::dot_with_boundary_check(LinearDataPoint & ldp)
{
  double sum=0;
  register int i;
  for (i=0; i<ldp.numFloatFeatures; i++) {
    if (ldp.floatArray[i].index<numFloatFeatures)
      sum += floatVec[ldp.floatArray[i].index]*ldp.floatArray[i].value;
  }
  for (i=0; i<ldp.numBinaryFeatures; i++) {
    if (ldp.binaryIndexArray[i]<numBinaryFeatures)
      sum += binaryVec[ldp.binaryIndexArray[i]];
  }
  sum = sum*ldp.norm_factor+b;
  return sum;
}

double ParamVector::trunc_dot(LinearDataPoint & ldp, double eps)
{
  int i;
  double sum=0;
  if (eps<1e-10) {
    for (i=0; i<ldp.numFloatFeatures; i++) {
      sum += floatVec[ldp.floatArray[i].index]*ldp.floatArray[i].value;
    }
    for (i=0; i<ldp.numBinaryFeatures; i++) {
      sum += binaryVec[ldp.binaryIndexArray[i]];
    }
  }
  else {
    for (i=0; i<ldp.numFloatFeatures; i++) {
      register double ww=floatVec[ldp.floatArray[i].index];
      sum += (ww>eps?ww-eps:(ww<-eps?ww+eps:0))*ldp.floatArray[i].value;
    }
    for (i=0; i<ldp.numBinaryFeatures; i++) {
      register double ww=binaryVec[ldp.binaryIndexArray[i]];
      sum += (ww>eps?ww-eps:(ww<-eps?ww+eps:0));
    }
  }
  sum = sum*ldp.norm_factor+b;
  return sum;
}

void ParamVector::trunc(double eps)
{
  if (eps<1e-10)  return;
  int i;
  for (i=0; i<numFloatFeatures; i++) {
    register double ww=floatVec[i];
    floatVec[i] =
      (ww>eps?ww-eps:(ww<-eps?ww+eps:0));
  }
  for (i=0; i<numBinaryFeatures; i++) {
    register double ww=binaryVec[i];
    binaryVec[i]= (ww>eps?ww-eps:(ww<-eps?ww+eps:0));
  }
}

void ParamVector::destroy() 
{
  if (floatVec) {
    delete [] floatVec;
  }
  if (binaryVec) {
    delete [] binaryVec;
  }
  clear();
}


void ParamVector::write(ostream & os)
{
  os << b << ' ';
  os << numFloatFeatures << ' ';
  write_arr(os,floatVec,numFloatFeatures);

  os << numBinaryFeatures << ' ';
  write_arr(os,binaryVec,numBinaryFeatures);
}

void ParamVector::read(istream & is)
{
  is >> b;
  is >> numFloatFeatures; 
  if (numFloatFeatures>0) 
    floatVec = new linfloat_t [numFloatFeatures];
  else 
    floatVec=0;
  read_arr(is, floatVec, numFloatFeatures);

  is >> numBinaryFeatures;
  if (numBinaryFeatures>0) 
    binaryVec = new linfloat_t [numBinaryFeatures];
  else 
    binaryVec=0;
  read_arr(is, binaryVec, numBinaryFeatures);
}

