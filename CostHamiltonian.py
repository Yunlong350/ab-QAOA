#This file contains all the operations we need in QAOA 
import numpy as np



def matrix_to_qubit(v,n,Z):
 
    qubit=np.array([np.real(Z[i].dot(v).dot(v))for i in range(n)])
    
    qubit=(1-qubit)/2
        
    
    return qubit




#to find the eigenvalue,eigenvector, ground energy, ground state of cost Hamiltonian, used as costH_object    
class CostHamiltonian:
    
    def __init__(self,pro_solve):
        self.constraints=pro_solve['clause']
        self.pro=pro_solve['Problem']
        self.excited=pro_solve['excited']
        

    def calculate_cost(self,HXYZ):        
        
        max_excited=self.excited
        
        self.costH=generate_costH(self.constraints,HXYZ,self.pro)        
  
        self.costH_eigen_values=self.costH.diagonal()


        self.e=np.sort(np.round(self.costH_eigen_values,8))
        self.e_index=np.argsort(np.round(self.costH_eigen_values,8))
        

        self.degeneracy=np.zeros(max_excited,'int32')
        for i in range(max_excited):
            a=int(np.sum(self.degeneracy))
            if a<len(self.costH_eigen_values):
                self.degeneracy[i]=np.sum(self.e==self.e[a])

        m=self.e_index[:int(np.sum(self.degeneracy))]


        
        self.ground_energy=self.e[0]
        costH_eigen_vectors=np.zeros(shape=(len(m),len(self.costH_eigen_values)))
        for j in range(len(m)):
            costH_eigen_vectors[j][m[j]]=1
        self.psi_final=costH_eigen_vectors
        self.psi_ground=self.psi_final[:self.degeneracy[0]]
    
        return len(m)        

        

        




def generate_costH(constraints,HXYZ,Problem):
    Z=HXYZ['Z']
    I=HXYZ['I']
    

    if Problem=='ExactCover3(p1-3-SAT)':
        for k in range(len(constraints)):
    
            v1=int(constraints[k][0])
            v2=int(constraints[k][1])
            v3=int(constraints[k][2])
          
            H1=(Z[v1-1]+Z[v2-1]+Z[v3-1]-I)/2
    
            if k==0:
                H=H1.dot(H1)
            else:
                H+=H1.dot(H1)


    return H

    

    

    
    