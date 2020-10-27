/********************************************************************
 *  File: predict.cc
 *  Copyright (C) 2012, 2013 Tong Zhang 
 * 
 *  Description: load linear model and predict
 *
 ********************************************************************/

#include "linear_classifier.hh"

float norm_fact=-1.0;
int verbose_level=1;

char * tstfn= (char *) "";
char * tstlabel=0;
char * modfn= (char *) "";
char * outfn= (char *) "";

int start_label=0;
int end_label=-1;



double theta=0;
bool use_theta=false;

void usage(char *nm) {
  cout << "usage: " <<nm << " model_file test_file output_file [option]... " <<endl;
  cout << "  model_file : model file name to load a previously trained model" <<endl ;
  cout << "  test_file  : test file name." <<endl;
  cout << "             sparse format: one data point per line; one nonzero-feature per column index@value" <<endl;
  cout << "             if no extra test_label_file is specified, then the 1st column is label" <<endl;
  cout << "  output_file: output predictions to file name." <<endl <<endl;
  cout << "options: " <<endl;
  cout << "  -tstlabel=test_label_file : whether to read test label from an extra file" <<endl;
  cout << "            (if this option is not used: label is the first column of training file)" <<endl;
  cout << "  -threshold=theta : set the decision threshold to theta." <<endl;
  cout <<endl;
  exit(0);
};


static char * parse(char *str, const char *arg) {
  if (!strncmp(str,arg,strlen(arg))) { 
    return (str+strlen(arg));
  }
  return 0;
}

void commandLineParse(int argc, char *argv[])
{
  int i;
  char *s;
  if (argc<=3){
    usage(argv[0]);
  }
  else {
    modfn=argv[1];
    tstfn=argv[2];
    outfn=argv[3];
  }
  for (i=4; i<argc; i++) {
    if ((s=parse(argv[i],"-threshold="))) {
      theta=atof(s);
      use_theta=true;
    }
    else if ((s=parse(argv[i],"-tstlabel="))) {
      tstlabel= s;
    }
    else if ((s=parse(argv[i],"-outfn="))) {
      outfn= s;
    }
    else {
      cerr << " invalid option " << argv[i] <<endl;
      usage(argv[0]);
    }
  }

  return;
}


int main(int argc, char * argv[])
{
  commandLineParse(argc,argv);
  cerr <<endl;

  LinearDataReader tst;

  MultiLinearClassifier multi_lin;


  cerr << "reading model from " << modfn << "..." <<endl;
  ifstream is(modfn);
  is >> norm_fact;
  is >> start_label;
  is >> end_label;
  multi_lin.read(is);
  is.close();

  if ((int) multi_lin.appl_vec.size() != end_label-start_label+1) {
    cerr << " error in model loading " <<endl;
    exit(0);
  }

  if (norm_fact>0) {
    cerr << "normalize data to: ||x||= " << norm_fact << endl;
  }    
  if (use_theta) {
    cerr << "decision threshold= " << theta << endl;
  }

  cerr << endl << "reading test set ... "<<endl;
  bool label_read= tst.append_data(tstlabel, tstfn, norm_fact);

  if (label_read) {
    if (start_label>=0) { // classification
      cerr << "testing binary classification performance " <<endl;
      BinaryTestStat micro_stat;
      for (int cl=start_label; cl<=end_label; cl++) {
	cerr << "-- class " << cl << " --" <<endl;

	// testing
	cerr << "testing" <<endl;
	BinaryTestStat bts;
	if (use_theta) {
	  bts.update(*multi_lin.appl_vec[cl-start_label], tst.get_dataset(), tst.get_binary_labels(cl),theta);
	}
	else {
	  bts.update(*multi_lin.appl_vec[cl-start_label], tst.get_dataset(), tst.get_binary_labels(cl));
	}
	micro_stat.update(bts);

	// printing stat
	bts.print(cerr);
	bts.clear();
      }
      if (end_label>start_label) {
	cerr << "===== micro stats from "<< start_label << " to "
	     << end_label << endl;
	micro_stat.print(cerr);
      }
	
      if (start_label==0 && end_label>start_label) {
	cerr << endl << "===== testing multi-category performance " <<endl;
	int cr=0;
	LinearDataSet tst_lds=tst.get_dataset();
	for (int i=0; i<tst.size(); i++) {
	  double scr;
	  
	  int cl=multi_lin.classify(tst_lds.ldps[i],scr);
	  if (tst.get_label(i)==cl) cr++;
	}
	cerr << " multi-category classification accuracy = " 
	     << cr/(float)tst.size()*100 << "\%" <<endl;
      }
      else if (start_label<0) {  // regression
	LinearDataSet tst_lds=tst.get_dataset();
	float * targets= tst.get_targets();
	double se=0;
	for (int i=0; i<tst.size(); i++) {
	  double err=targets[i]-multi_lin.appl_vec[0]->apply(tst_lds.ldps[i]);
	  se += err*err;
	}
	cerr << " root mean squared error = " << sqrt(se/(float)tst.size()) 
	     << endl;
      }
    }

    if (outfn[0]) {
      cout << " ==== outputing scores to " << string(outfn) <<endl;
      ofstream os(outfn);
      LinearDataSet tst_lds=tst.get_dataset();
      if (start_label==end_label) { // binary classification or regression
	for (int i=0; i<tst.size(); i++) {
	  os << multi_lin.appl_vec[0]->apply(tst_lds.ldps[i]) <<endl;
	}
      }
      else { // multi-category classification
	for (int i=0; i<tst.size(); i++) {
	  double scr;
	  int cl=multi_lin.classify(tst_lds.ldps[i], scr);
	  os << cl << " " << scr <<endl;
	}
      }
      os.close();
    }
  }
  multi_lin.destroy();
}


