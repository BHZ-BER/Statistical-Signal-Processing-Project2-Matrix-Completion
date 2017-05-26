# Statistical-Signal-Processing-Project2-Matrix-Completion
The Source code and paper for Statistical Signal Processing Project2 Matrix Completion.

## File List
+ AGD.m, The source code for Alternating Gradient Descent method.
+ ALS.m, The source code for Alternating Least Square method.
+ LDMM.m, The source code for Low Dimensional Manifold Model method.
+ AGDOut.mat, The outcome (matrix X) of Alternating Gradient Descent method.
+ ALSOut.mat, The outcome (matrix X) of Alternating Least Square method.
+ LDMMOut.mat, The outcome (matrix X) of Low Dimensional Manifold Model method.
+ data_train.mat The source file for training and testing.
+ Other .m files are necessary functions for running LDMM.
In order to run LDMM, VLFeat toolbox [http://www.vlfeat.org/]  for matlab is necessary. 

## Usage

Simply run AGD, ALS or LDMM to get the predicted matrix X. 

Note that we are running ten fold cross validation, so the total time should be ten times the single processing procedure duration. (For ALS and LDMM, it may take several minutes. For AGD it may take several seconds.)

One may refer to report.pdf for detailed introduction for the three algorithms. 