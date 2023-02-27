
#replace this file with Adaptive_para to run optimization free ab-QAOA

import numpy as np
from CostHamiltonian import CostHamiltonian
from Gradient import GradientInput,adam_iters
#parallel computation
from joblib import Parallel,delayed
from QAOA import QAOA

import time
#input parameters of adaptive_qaoa
class AdaptiveQaoaInput:

    def __init__(self,seed,level_list,R,n,costH,sys_para,H):
        #random number seed
        self.seed=seed
        #QAOA levels
        self.level_list=level_list
        #sample parameters 
        self.R=R
        #number of qubits
        self.n=n
        #cost Hamiltonian 
        self.costH=costH
        #parameters needed in the algorithm implementation
        self.sys_para=sys_para
        #operators from TensorProductXYZ.py
        self.H=H


#output information of adaptive_qaoa
class AdaptiveQaoaOutput:
        
    def __init__(self,energy,Pf,itera,others):
        #energys in different levels
        self.energy=energy
        #final ground state fidelities in different levels
        self.Pf=Pf
        #average iteration over R samples in different level
        self.itera=itera
        #other outputs
        self.others=others
        

def adaptive_qaoa(input_para):
    #QAOA levels
    L=len(input_para.level_list)
    #sample parameters
    R=input_para.R
    #number of qubits
    n=input_para.n
    #empirical parameters for initilization
    alpha=0.6
    
    costH=input_para.costH
    #number of states we care about
    l=len(costH.degeneracy)
    
    
    np.random.seed(input_para.seed)
    
    #in the following arrays,2 means this array contains both standard (0) and adaptive bias(1) QAOA
    Pf=np.zeros(shape=(2,L,l),dtype=np.float64)      

    ave_iteration=np.zeros(shape=(2,L),dtype=np.float64)
    opt_iterationR=np.zeros((2,R),dtype=np.float64)
    
    best_energy=np.zeros((2,L),dtype=np.float64)
    opt_energyR=np.zeros((2,L,R),dtype=np.float64)
    opt_PfR=np.zeros((2,L,R),dtype=np.float64)


 
    best_other_opt=np.empty((2,L),dtype=object)

   
    t1=time.perf_counter()
    biasR=np.zeros((2,R,n),dtype=np.float64) 
    biasR[1]=np.sign(np.random.uniform(-1,1,(R,n)))
    
    for ith in range(L): 

        
        level=input_para.level_list[ith]
              
        delta_t=0.6
        initial_pointR=np.zeros((R,2*level))

        for i in range(level):
            initial_pointR[0][2*i]=i/level*delta_t
            initial_pointR[0][2*i+1]=(1-i/level)*delta_t
        
        for k in range(1,R):
            initial_pointR[k]=initial_pointR[0]+alpha*np.random.normal(0,initial_pointR[0]**2) 

        
        
        for k in range(R):
        
            qaoa=QAOA(n,
                level,
                costH,
                {'flag':True,'bias':biasR[1][k]},
                input_para.sys_para['learning_rate'],
                input_para.H
                )
            
            qaoa.get_mixing_info()
            qaoa.get_expectation(initial_pointR[0])
            
            res=qaoa.get_bias_state()
            opt_energyR[1][ith][k]=res[0]
            opt_PfR[1][ith][k]=res[1]
            
            qaoa.update_bias()
            biasR[1][k]=qaoa.bias

        for j in range(1,2):

            ma=np.where(opt_energyR[j][ith]==np.min(opt_energyR[j][ith]))
  
            p_ma=ma[0][0]       

            best_energy[j][ith]=opt_energyR[j][ith][p_ma]
            
            Pf[j][ith]=opt_PfR[j][ith][p_ma]

        
        

        t2=time.perf_counter()
        print('Level',level,'energy',best_energy[:,ith],'Pf',Pf[:,ith][:,0],'iterations',ave_iteration[:,ith],'time',t2-t1)
        t1=t2
    

    result = AdaptiveQaoaOutput(best_energy,
                                Pf,
                                ave_iteration,
                                best_other_opt)
 
    return result

