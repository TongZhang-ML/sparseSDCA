#!/bin/sh

../src/train ./inputs/pv_on_txt ./models/simple_model -lambda1=1.92745e-07 -lambda2=1.92745e-07 -max.iters=50 -loss=LOGISTIC  -min.dgap=1e-6 -accl.kappa=6.42484e-05 -verbose=3 -chk.interval=1 -accl.iters=4 -accl.beta=0.9 -label=1

