CXXFLAGS = -Wall  -O3 -fPIC
CXX = g++
CLIBS =  -lm 

.SUFFIXES: .o .cc

.cc.o: 
	$(CXX) $(CXXFLAGS) -c -o $@ $<

all: train predict 

train: train.o linear_classifier.o linear_trainer.o linear_dual.o linear_fista.o linear_accl.o 
	$(CXX) $(CXXFLAGS) -o train train.o  linear_classifier.o linear_trainer.o linear_dual.o linear_fista.o linear_accl.o $(LIBS)

predict: predict.o linear_classifier.o
	$(CXX) $(CXXFLAGS) -o predict predict.cc linear_classifier.o $(LIBS)

clean:
	rm -f *~ *.o train predict 
