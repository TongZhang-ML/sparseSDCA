/********************************************************************
 *  File: train.cc
 *  Copyright (C) 2012, 2013 Tong Zhang 
 * 
 *  Description: training a linear classifier
 *
 ********************************************************************/

#include "linear_trainer.hh"

float norm_fact=-1.0;
int verbose_level=1;

char * trnfn= (char *) "";
char * trnlabel=0;
char * tstfn= (char *) "";
char * tstlabel=0;
char * modfn= (char *) "";

int start_label=0;
int end_label=-1;



LinearTrainer trainer;


void usage(char *nm) {
  cout << "usage: " <<nm << " train_file model_file [option]... " <<endl;
  cout << "  train_file : training file name." <<endl;
  cout << "             sparse format: one data point per line; one nonzero-feature per column index@value" <<endl;
  cout << "             if no extra train_label_file is specified, then the 1st column is label" <<endl;
  cout << "  model_file : model file name to save the trained model" <<endl << endl;
  cout << "options: " <<endl;
  cout << "  -trnlabel=train_label_file : whether to read training label from an extra file" <<endl;
  cout << "            (if this option is not used: label is the first column of training file)" <<endl;
  cout << "  -label=class_label : class_label>=0 -- only train binary classifier for class_label as the positive class" <<endl;
  cout << "                       class_label<0 --  regression where targets are real numbers" <<endl;
  cout << "                       default: train multi-class classifier with class_label 0,1,2,..." << endl;
  cout << "  -norm=factor :  normalize the data so that 2-norm of each datum equals factor"<<endl;
  cout << "            factor <=0 implies no normalization (default factor=" <<norm_fact   << ")" <<endl;
  cout << "  -verbose=verbose_level : debug output verbose level" <<endl;
  cout << "                           >=1: print training parameters" <<endl;
  cout << "                           >=2: print duality gaps every few epochs" <<endl;
  cout << "                           >=3: print test accuracy on test data" <<endl;
  cout << "  -tstfile=test_file : test file name" << endl;
  cout << "                      if present and combined with verbose_level>=3: output test performance during training" <<endl;
  cout << "  -tstlabel=test_label_file : whether to read test label from an extra file" <<endl;
  cout << "            (if this option is not used: label is the first column of test file)" <<endl;
  cout <<endl;
  cout << "training options: " << endl;
  LinearTrainer::printCommandLineOptions(cout);
  cout <<endl<<endl;
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
  if (argc<=2){
    usage(argv[0]);
  }
  else {
    trnfn=argv[1];
    modfn=argv[2];
  }
  for (i=3; i<argc; i++) {
    if (!trainer.commandLineParse(argv[i])) {
      if ((s=parse(argv[i],"-norm="))) {
	norm_fact=atof(s);
      }
      else if ((s=parse(argv[i],"-verbose="))) {
	verbose_level=atoi(s);
      }
      else if ((s=parse(argv[i],"-trnlabel="))) {
	trnlabel= s;
      }
      else if ((s=parse(argv[i],"-tstfile="))) {
	tstfn= s;
      }
      else if ((s=parse(argv[i],"-tstlabel="))) {
	tstlabel= s;
      }
      else if ((s=parse(argv[i],"-label="))) {
        start_label=end_label= atoi(s);
      }
      else {
	cerr << " invalid option " << argv[i] <<endl;
	usage(argv[0]);
      }
    }
  }

  return;
}


int main(int argc, char * argv[])
{
  commandLineParse(argc,argv);
  cerr <<endl;

  LinearDataReader trn, tst;

  MultiLinearClassifier multi_lin;


  // read training set
  if (norm_fact>0) {
    cerr << "normalize data to: ||x||= " << norm_fact << endl;
  }

  cerr << "reading training set ... "<<endl;
  bool label_read= trn.append_data(trnlabel,trnfn,norm_fact);

  if (!label_read) {
    cerr << "no label information for training data" <<endl;
    exit(-1);
  }    
    
  LinearDataSet *tst_ptr=0;
  bool * tst_label=0;
  float * tst_targets=0;
  if (tstfn[0]) {
    cerr << endl << "reading test set ... "<<endl;
    bool label_read=
      tst.append_data(tstlabel,tstfn,norm_fact);
    if (!label_read) {
      cerr << "no label information for training data" <<endl;
      tst_ptr=0;
    }
    else {
      tst_ptr=& tst.get_dataset();
    }
  }

  // determine end label
  int ml=0;
  if (start_label<0) {
    end_label=start_label; // regression
  }
  else {
    ml=trn.max_label();
    if (ml<start_label) ml=start_label;
    if (end_label>ml || end_label<0) end_label=ml;
    if (start_label==0 && end_label==1) start_label=1; // binary classification
  }

  // the actual training
  cerr.precision(10);
  if (start_label>=0) { // classification
    for (int cl=start_label; cl<=end_label; cl++) {
      cerr << "-- training class " << cl << " --" <<endl;
      if (tst_ptr) tst_label=tst.get_binary_labels(cl);

      multi_lin.appl_vec.push_back(trainer.train(trn.get_dataset(),trn.get_binary_labels(cl),
							   0,tst_ptr,tst_label,0,verbose_level));
    }
  }
  else { // regression
    if (tst_ptr) tst_targets=tst.get_targets();
    multi_lin.appl_vec.push_back(trainer.train(trn.get_dataset(),0,trn.get_targets(),tst_ptr,0,tst_targets,verbose_level));
  }
  cerr << " ****** done training ****** " <<endl<<endl;

  if (modfn[0]) {
    cerr << "writing model to " << modfn << "..." << endl;
    ofstream os(modfn);
    os << norm_fact << ' ';
    os << start_label << ' ';
    os << end_label << ' ';
    multi_lin.write(os);
    os.close();
  }
  multi_lin.destroy();
}


