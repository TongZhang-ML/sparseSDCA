#!/bin/sh
#

trn=trainset
tst=testset

for lam2 in 1e-3 1e-4 1e-5 1e-6
do 
    echo +++++++++++++++ lambda2= $lam2 +++++++++++++


    echo == running SDCA ==
    ../src/train ./inputs/${trn}.dat ./models/${trn}-sdca-$lam2   -max.iters=1000 -chk.interval=1 -min.dgap=1e-8 -verbose=3 -loss=SmoothHinge,1 -norm=1 -lambda2=$lam2 -lambda1=1e-3 -sgd.init=0 -label=1 | tee -i outputs/${trn}-sdca-$lam2

    echo == running FISTA ==
    ../src/train ./inputs/${trn}.dat ./models/${trn}-fista-$lam2  -max.iters=100 -chk.interval=1 -min.dgap=1e-8 -verbose=3 -loss=SmoothHinge,1 -norm=1 -lambda2=$lam2 -lambda1=1e-3 -fista.eta=1 -label=1 | tee -i outputs/${trn}-fista-$lam2

    echo == running ACCL-SDCA ==
    ../src/train ./inputs/${trn}.dat ./models/${trn}-accl-$lam2  -max.iters=25 -chk.interval=1 -min.dgap=1e-8 -verbose=3 -loss=SmoothHinge,1 -norm=1 -lambda2=$lam2 -lambda1=1e-3 -accl.iters=5 -accl.kappa=1e-4 -accl.beta=0.9 -label=1 | tee -i outputs/${trn}-accl-$lam2

done



