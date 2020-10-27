# sparseSDCA Version 0.1

### C++ code for prox-SDCA

************************************************************************

# 0. Changes

This is an old implementation in 2013, which is provided as is. This package has
not been updated since.

************************************************************************

# Contents:

1. [Introduction](#1-introduction)

2. [System Requirement](#2-system-requirement)

3. [Installation](#3-installation)

4. [Examples](#4-examples)

5. [Contact](#5-contact)

6. [Copyright](#6-copyright)

7. [References](#7-references)

# 1. Introduction

This software package provides a sample implementation of accelerated
Proximal Stochastic Dual Coordinate Ascent with L1-L2 regularization
described in [[1]](#7-references) for various loss functions. Please
cite the paper if you find the software useful.

The code has not been updated since 2013, and is provided as is.

# 2. System Requirement

The code has been tested on Linux, but should compile on other unix systems
with g++ and make.


```
git clone https://github.com/TongZhang-ML/sparseSDCA.git
```

# 3. Installation

The source files are located in the `src` directory.

To compile:
```
cd src/
make
```

This should compile into two binary programs `train` and `predict`

- **train**: train and save the model;
- **predict**: apply already trained model on test data.

Use `train -h` to see command line options for the training program

Use `predict -h` to see command line options for the prediction program


API documentations are in html/index.html


# 4. Examples

 Please go to the `example1` and  `example2` subdirectories and type `run.sh`.


# 5. Contact

tongzhang@tongzhang-ml.org

# 6. Copyright

The software is distributed under the **MIT license**. Please read the file [`LICENSE`](./LICENSE).

# 7. References

[1] Shai Shalev-Shwartz and Tong Zhang. [Accelerated Proximal
Stochastic Dual Coordinate Ascent for Regularized Loss Minimization](./papers/mathprog16-proxsdca.pdf),
Mathematical Programming, 155:105-145, 2016.

[2] Shai Shalev-Shwartz and Tong Zhang. [Stochastic Dual Coordinate Ascent Methods for Regularized Loss Minimization](./papers/jmlr13-sdca.pdf), JMLR 14:567-599, 2013.
