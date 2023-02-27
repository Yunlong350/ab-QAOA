
import numpy as np
from QAOA import QAOA
from scipy.optimize import minimize




def gradient_function(para,qaoa,epsilon):
    gra=np.zeros(len(para),dtype=np.float64)
    e2=np.zeros(len(para),dtype=np.float64)

    for j in range(len(para)):
        p_j=para[j].copy()
        para[j]=para[j]+epsilon   
        e2[j]=qaoa.get_expectation(para)
        para[j]=p_j
    
    e1=qaoa.get_expectation(para)
    gra=(e2-e1)/epsilon
    return gra,e1


  
#the input of adam_grad_desc
class GradientInput:
    
    def __init__(self,para,n,level,costH_object,bias_field,sys_para,H):

        self.para=para

        self.n=n

        self.level=level

        self.costH_object=costH_object
        
        self.bias_field=bias_field

        self.max_iters=sys_para['max_iters']

        self.learning_rate=sys_para['learning_rate']

        self.H=H

        
        
        
#the output of adam_grad_desc
class GradientOutput:

    def __init__(self,para_opt,energy,itera,Pf,bias_opt,others):
  
        self.para_opt=para_opt

        self.energy=energy

        self.itera=itera

        self.Pf=Pf

        self.bias_opt=bias_opt

        self.others=others


def adam_iters(gradient):

    #SAT precision -5 -3 -2
    #the small quantity for the calculation of gradient function
    para_precision=1e-5
    #the small quantity for convergence of cost function in optimization 
    final_precision=1e-3
    #the small quantity for convergence of points in optimization 
    theta_precision=1e-1

     
    qaoa=QAOA(gradient.n,
                gradient.level,
                gradient.costH_object,
                gradient.bias_field,
                gradient.learning_rate,
                gradient.H
                )

    #parameters in adam gradient descent algorithm
    beta1=0.9
    beta2=0.999
    alpha=0.02
    epsilon=10**(-8)   
    
    npoints=2*gradient.level
    

    #adam gradient descent algorithm

    m=np.zeros(npoints,dtype=np.float64)
    v=np.zeros(npoints,dtype=np.float64)   


    qaoa.get_mixing_info()




    theta=np.zeros(shape=(int(gradient.max_iters+1),npoints),dtype=np.float64)
    energy=np.zeros((int(gradient.max_iters+1)),dtype=np.float64)
    bias=np.zeros(shape=(int(gradient.max_iters+1),qaoa.n),dtype=np.float64)
    g=np.zeros(shape=(int(gradient.max_iters+1),npoints),dtype=np.float64)



    theta[0]=gradient.para.copy() 
    
    
    g[0],energy[0]=gradient_function(theta[0],qaoa,para_precision)  
    
    bias[0]=qaoa.bias

    
    t=0
    for t in range(1,gradient.max_iters+1):

        m=beta1*m+(1-beta1)*g[t-1]
        v=beta2*v+(1-beta2)*g[t-1]**2
        
        alphat=alpha*np.sqrt(1-beta2**t)/(1-beta1**t)
        theta_change=alphat*m/(np.sqrt(v)+epsilon)
        theta[t]=theta[t-1]-theta_change
        

        if qaoa.bias_flag==True:

            qaoa.update_bias()                       
            qaoa.get_mixing_info()

        
        bias[t]=qaoa.bias
        g[t],energy[t]=gradient_function(theta[t],qaoa,para_precision)

          

        if np.abs(energy[t]-energy[t-1]) < final_precision and np.linalg.norm(theta_change)<theta_precision: 

            break
    

    Pg,Pt=qaoa.get_probability()

    others={'theta':theta[:t+1],'probability':Pt,'energy_opt':energy[:t+1],'bias_opt':bias[:t+1],'gra_opt':g[:t+1]}

    return GradientOutput(theta[t],energy[t],t,Pg,bias[t],others)

    
   

  


    
        
