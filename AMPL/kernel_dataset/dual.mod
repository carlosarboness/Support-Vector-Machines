#########################################################################
## Implementation of dual quadratic formulation of the SVM classifier  ##
#########################################################################

### Parametres ### 
param n >= 1, integer; # number of variables
param m >= 1, integer; # number of samples
param nu;              # hyperparameter
param y{i in 1..m};    # predicted label vector
param A{1..m, 1..n};   # matrix of data 

### Variables ###
var la{1..m}; 	

maximize fobj_dual: 
	(sum{i in 1..m} la[i]) - (1/2*sum{i in 1..m, j in 1..m} la[i]*y[i]*la[j]*y[j]*(sum{k in 1..n} A[i,k]*A[j,k])); 
subject to h: 
	sum{i in 1..m} la[i]*y[i] = 0; 
subject to c{i in 1..m}: 
	0 <= la[i] <= nu;  
	
	