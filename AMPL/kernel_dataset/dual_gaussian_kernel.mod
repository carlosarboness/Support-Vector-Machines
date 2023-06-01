#########################################################################
## Implementation of dual quadratic formulation of the SVM classifier  ##
#########################################################################

### Parametres ### 
param n >= 1, integer; # number of variables
param m >= 1, integer; # number of samples
param nu;
param sigma;                
param y{i in 1..m};    # vector of {-1,1}
param A{1..m, 1..n};   # matrix of data 

param TEST_SIZE >= 1, integer; 
param test{1..TEST_SIZE, 1..n}; 

### Variables ###
var la{1..m}; 	

maximize fobj_dual: 
	(sum{i in 1..m} la[i]) - 
	(1/2*sum{i in 1..m, j in 1..m} la[i]*y[i]*la[j]*y[j]*exp(-sum{k in 1..n} ((A[i,k] - A[j,k])^2) / (2 * sigma^2))); 
subject to h: 
	sum{i in 1..m} la[i]*y[i] = 0; 
subject to c{j in 1..m}: 
	0 <= la[j] <= nu;  
	
	