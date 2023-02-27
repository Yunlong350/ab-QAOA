import numpy as np
from CostHamiltonian import CostHamiltonian
from Gradient import GradientInput,adam_iters
#parallel computation
from joblib import Parallel,delayed


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
    
    opt_method=input_para.sys_para['opt_method']
    
    np.random.seed(input_para.seed)
    
    #in the following arrays,2 means this array contains both standard (0) and adaptive bias(1) QAOA
    Pf=np.zeros(shape=(2,L,l),dtype=np.float64)      

    ave_iteration=np.zeros(shape=(2,L),dtype=np.float64)
    opt_iterationR=np.zeros((2,R),dtype=np.float64)
    
    best_energy=np.zeros((2,L),dtype=np.float64)
    opt_energyR=np.zeros((2,L,R),dtype=np.float64)
    opt_PfR=np.zeros((2,L,R),dtype=np.float64)

    #the bias for output
    best_bias=np.zeros(shape=(2,L,n),dtype=np.float64)
 
    best_other_opt=np.empty((2,L),dtype=object)

   
    t1=time.perf_counter()

    for ith in range(L): 

        biasR=np.zeros((2,R,n),dtype=np.float64) 
        level=input_para.level_list[ith]
              
        if opt_method=='ModifiedTQA':
            
            delta_t=0.6#0.2
            initial_pointR=np.zeros((R,2*level))
    
            for i in range(level):
                initial_pointR[0][2*i]=i/level*delta_t
                initial_pointR[0][2*i+1]=(1-i/level)*delta_t
            
            for k in range(1,R):
                initial_pointR[k]=initial_pointR[0]+alpha*np.random.normal(0,initial_pointR[0]**2) 
            
    
            biasR[1]=np.sign(np.random.uniform(-1,1,(R,n)))
            
            adam_input=[GradientInput(
                                      initial_pointR[k],
                                      n,
                                      level,
                                      costH,
                                      {'flag':flag,'bias':biasR[flag][k]},
                                      input_para.sys_para,
                                      input_para.H
                                      ) 
                        for flag in range(2) for k in range(R)]  
        
        elif opt_method=='ModifiedFourier':

            if ith==0:
    
                
                initial_pointR=np.random.uniform(0,np.pi,(R,2*level))       
                biasR[1]=np.sign(np.random.uniform(-1,1,(R,n)))
                adam_input=[GradientInput(
                                          initial_pointR[k],
                                          n,
                                          level,
                                          costH,
                                          {'flag':flag,'bias':biasR[flag][k]},
                                          input_para.sys_para,
                                          input_para.H
                                          ) 
                            for flag in range(2) for k in range(R)]
              
            else:      
               
                
                initial_pointR=np.zeros(shape=(2,R,2*level),dtype=np.float64)
                
                biasR[1]=np.sign(np.random.uniform(-1,1,(R,n)))
     
                for j in range(0,2):
                    initial_pointR[j][0][:2*input_para.level_list[ith-1]]=best_point[j]
                    for k in range(1,R):   
                        initial_pointR[j][k]=initial_pointR[j][0]+alpha*np.random.normal(0,initial_pointR[j][0]**2)                 

                #input 
                adam_input=[GradientInput(
                                          initial_pointR[flag][k],
                                          n,
                                          level,
                                          costH,
                                          {'flag':flag,'bias':biasR[flag][k]},
                                          input_para.sys_para,
                                          input_para.H
                                          ) 
                            for flag in range(2) for k in range(R)]
        

        #output
        gradient_result=Parallel(n_jobs=20)(delayed(adam_iters)(adam) for adam in adam_input)

        for j in range(2):
            for k in range(0,R):
                opt_energyR[j][ith][k]=gradient_result[j*R+k].energy  
                opt_PfR[j][ith][k]=gradient_result[j*R+k].Pf
                opt_iterationR[j][k]=gradient_result[j*R+k].itera


        best_point=np.zeros(shape=(2,2*level),dtype=np.float64) 
  
        for j in range(2):
            
            ma=np.where(opt_energyR[j][ith]==np.min(opt_energyR[j][ith]))
   
            p_ma=ma[0][0]            
            best_energy[j][ith]=opt_energyR[j][ith][p_ma]
  
            best_point[j]=gradient_result[j*R+p_ma].para_opt  
            ave_iteration[j][ith]=np.mean(opt_iterationR[j],axis=0)
            Pf[j][ith]=gradient_result[j*R+p_ma].Pf
            best_bias[j][ith]=gradient_result[j*R+p_ma].bias_opt

            best_other_opt[j][ith]=gradient_result[j*R+p_ma].others

        
        t2=time.perf_counter()
        print('Level',level,'energy',best_energy[:,ith],'Pf',Pf[:,ith][:,0],'iterations',ave_iteration[:,ith],'time',t2-t1)
        t1=t2
        


    result = AdaptiveQaoaOutput(best_energy,
                                Pf,
                                ave_iteration,
                                best_other_opt)
 
    return result

