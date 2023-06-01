#########################################################################
## Implementation of primal quadratic formulation of the SVM classifier##
#########################################################################

### Parameters ### 
param n >= 1 integer; # number of variables
param m >= 1 integer; # number of samples
param nu;              # hyperparameter 
param y{i in 1..m};    # predicted label vector
param A{1..m, 1..n};   # matrix of data 

### Variables ###
var w {1..n};		   # weights
var gamma;             # intercept
var s {1..m};          # slacks: soft constrains		

minimize fobj_primal: 
	1/2*(sum{i in 1..n} w[i]^2) + nu*(sum{i in 1..m} s[i]);
subject to classify{i in 1..m}: 
	-y[i]*(sum{j in 1..n} w[j]*A[i, j] + gamma) - s[i] + 1 <= 0;
subject to slacks{i in 1..m}:
	-s[i] <= 0;  
	
	