
import numpy as np
import networkx as nx
import random
from itertools import combinations as com
from scipy.special import comb

#spin 1/2 Pauli X,Y,Z operator
sigmax=np.array([[0,1],[1,0]],dtype=np.float64)
sigmay=np.array([[0,-1j],[1j,0]],dtype='complex128')
sigmaz=np.array([[1,0],[0,-1]],dtype=np.float64)
sigmai=np.array([[1,0],[0,1]],dtype=np.float64)

#positive 1 in 3 SAT problems
ProblemList=['ExactCover3(p1-3-SAT)']

Problem=ProblemList[0]

OptMethod_list=['ModifiedTQA','ModifiedFourier']
OptMethod=OptMethod_list[0]

#number of qubits
qubits=6
#seed of random numers
seed=10
#qaoa levels
level_list=[i for i in range(1,25)]#4,8,16,24]

L=len(level_list)
#sample parameters R
R_qaoa=10
#which excited is considered , 1 represents ground state 2 represents ground state and 1st excited state
max_e=1
#\ell for updating bias fields
learning_rate=0.4

#iterations in the optimization
max_iters=4000

np.random.seed(seed)
random.seed(seed)



if Problem=='ExactCover3(p1-3-SAT)':
    #threshold of the desicion version
    E_th=0.5
    #number of random problem instances
    realizations=100
    #number of the clauses to solve
    num_clause=np.array([i for i in range(3,11)]+[2*i+12 for i in range(10)])#+[10*i for i in range(4,12)])
    num_clause=np.ceil(num_clause*qubits/10)
    
    num_clause=np.array(num_clause,dtype=np.int)
    
    
    alpha_max=len(num_clause)
    
    alpha_list=num_clause/qubits
    
    def generate_clause(variables):

        variable_list=[i for i in range(1,variables+1)]
        clause_list=[]
        for i in range(variables):
            for j in range(i+1,variables):           
                for k in range(j+1,variables):
                    v1=variable_list[i]
                    v2=variable_list[j]
                    v3=variable_list[k]

                    clause_list.append([v1,v2,v3])
                    
        return clause_list
                
    clause_total=generate_clause(qubits) 
    
    pro_info={}

    for m in num_clause:
    
        clause_info=np.zeros(shape=(realizations,m,3))
        
        for i in range(realizations):
            clause_random=clause_total.copy()
            random.shuffle(clause_random)
            clause_info[i]=np.array(clause_random[:m])
            
        pro_info[str(m)]=clause_info
    


#pro_info={str(10*i):pro_info[str(10*i)] for i in range(4,12)}

import numpy as np
np.save('./definition/n='+str(qubits)+'.npy',pro_info)




