# acMark
### acMark: General Generator for Attributed Graph with Community Structure  
Users can control the characteristics of generated graphs by acMark.

## Requirements
- numpy >= 1.14.5
- scipy >= 1.1.0

## Example
> $ python acmark.py  

'test.mat' is generated by the example.

## Usage
In a python code,
> import acmark  
A, X, C = acmark.acmark(n=1000, m=4000, d=100)

Other parameters are described below:

## Parameter (default)

outpath : path to output file (.mat)  
n (=1000) : number of nodes  
m (=4000) : number of edges  
d (=100)  : number of attributes  
k (=5)  : number of clusters  
k2 (=10)  : number of clusters for attributes  
alpha (=0.2)  : parameters for balancing inter-edges and intra-edges  
beta (=10)  : parameters of separability for attribute cluster proportions  
gamma (=1)  : parameters of separability for cluster transfer proportions  
node_d (=0) : choice for node degree (0:power law, 1:uniform, 2:normal)  
com_s (=0)  : choice for community size (0:power law, 1:uniform, 2:normal)  
phi_d (=3)  : parameters of exponent for power law distribution for node degree  
phi_c (=2)  : parameters of exponent for power law distribution for community size  
delta_d (=3)  : parameters for uniform distribution for node degree  
delta_c (=2)  : parameters for uniform distribution for community size  
sigma_d (=0.1)  : parameters for normal distribution for node degree  
sigma_c (=0.1)  : parameters for normal distribution for community size   
r (=10) : number of iterations for edge construction  
att_ber (=0.0) : ratio of attributes which takes discrete value  
att_pow (=0.0) : ratio of attributes which follow power law distributions  
att_uni (=0.0) : ratio of attributes which follow uniform distributions  
att_nor (=0.5) : ratio of attributes which follow normal distributions  
dev_power_max (=3)  : upper bound of deviations for power law distribution for random attributes  
dev_power_min (=2)  : lower bound of deviations for power law distribution for random attributes  
dev_normal_max (=0.3) : upper bound of deviations for normal distribution for random attributes  
dev_normal_min (=0.1) : lower bound of deviations for normal distribution for random attributes  
uni_att (=0.2)  : range of paramters for uniform distribution for random attributes


## Citation
