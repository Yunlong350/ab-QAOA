import numpy as np
from TensorProductXYZ import generate_HZ,generate_HX,generate_HY
from scipy import sparse as se

from CostHamiltonian import CostHamiltonian
import time
from global_var import *


HX=generate_HX(qubits)
HZ=generate_HZ(qubits)
HY=generate_HY(qubits)


HX=[HX.get_ith(i) for i in range(qubits)]
HZ=[HZ.get_ith(i) for i in range(qubits)]
HY=[HY.get_ith(i) for i in range(qubits)]
HI=se.identity(2**(qubits),dtype='float64',format='csc')


HXYZ={'X':HX,'Y':HY,'Z':HZ,'I':HI}


start=time.perf_counter()

import numpy as np
from CostHamiltonian import CostHamiltonian
if Problem=='ExactCover3(p1-3-SAT)':
    
    #s represhent standard QAOA while ab represents ab-QAOA
    
    #residual energy 
    E_s=np.zeros(shape=(alpha_max,realizations,L))
    E_ab=np.zeros(shape=(alpha_max,realizations,L))
    #whether the problem instance is SAT (1) or not (0)
    sat_s=np.zeros(shape=(alpha_max,realizations))
    sat_ab=np.zeros(shape=(alpha_max,realizations))
    #success probability
    success_s=np.zeros(shape=(alpha_max,realizations,L))
    success_ab=np.zeros(shape=(alpha_max,realizations,L))  
    #the other outputs
    other_s=np.empty(shape=(alpha_max,realizations,L),dtype='object')
    other_ab=np.empty(shape=(alpha_max,realizations,L),dtype='object')
    #ground state fidelity
    Pg_s=np.zeros(shape=(alpha_max,realizations,L))
    Pg_ab=np.zeros(shape=(alpha_max,realizations,L))
    #number of iterations 
    ite_s=np.zeros(shape=(alpha_max,realizations,L))
    ite_ab=np.zeros(shape=(alpha_max,realizations,L))
    
    #ground energy of the Hamiltonian    
    ground_energy=np.zeros(shape=(alpha_max,realizations))
    ham_error=np.zeros(shape=(alpha_max,realizations))
    #ground state degeneracy
    degeneracy=np.zeros(shape=(alpha_max,realizations,max_e))
    
    

if Problem=='ExactCover3(p1-3-SAT)':
    for j in range(alpha_max):
        
        r=num_clause[j]
        text='clause'
   
        for i in range(realizations):
            print('qubits',qubits, text,r,'realizations',i, Problem, 'QAOA')
            
            pro_solve={'clause':pro_info[str(r)][i],'Problem':Problem,'excited':max_e}
            Hc=CostHamiltonian(pro_solve)
            l=Hc.calculate_cost(HXYZ)
            ground_energy[j][i]=Hc.ground_energy
            ham_error[j][i]=Hc.calculate_violation(HXYZ)
            degeneracy[j][i]=Hc.degeneracy
    
    

    '''
            para=AdaptiveQaoaInput (seed,
                                    level_list,
                                    R_qaoa,
                                    qubits,
                                    Hc,
                                    {'opt_method':OptMethod,'max_iters':max_iters,'learning_rate':learning_rate},
                                    HXYZ
                                    )
        
            
            res=adaptive_qaoa(para)
            

            E_s[j][i]=res.energy[0]-ground_energy[j][i]
            E_ab[j][i]=res.energy[1]-ground_energy[j][i]
            
            sat_s[j][i]=(ground_energy[j][i]<E_th)+0
            sat_ab[j][i]=(ground_energy[j][i]<E_th)+0
        
            
            for k in range(L):
                success_s[j][i][k]=(res.energy[0][k]<E_th and ground_energy[j][i]<E_th or res.energy[0][k]>E_th and ground_energy[j][i]>E_th )+0
                success_ab[j][i][k]=(res.energy[1][k]<E_th and ground_energy[j][i]<E_th or res.energy[1][k]>E_th and ground_energy[j][i]>E_th)+0
                
            
            Pg_s[j][i]=res.Pf[0][:,0]
            ite_s[j][i]=res.itera[0]
            
            Pg_ab[j][i]=res.Pf[1][:,0]
            ite_ab[j][i]=res.itera[1]

            other_s[j][i]=res.others[0]
            other_ab[j][i]=res.others[1]
            '''
            
e1=[]
e2=[]


                 
end=time.perf_counter()
print('time',end-start)


