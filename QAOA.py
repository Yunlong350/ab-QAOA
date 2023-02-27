#This file contains all the operations we need in QAOA 
#from numba import jit
import numpy as np
from global_var import sigmax,sigmaz,seed
from scipy import sparse as se
import qutip as qp






def ground_state(h):
    #h=0
    f=h-np.sqrt(1+h**2)
    
    s=np.array([1,f])/(np.sqrt(1+f**2))

    return s


def mixing_ground(bias,n):
    for i in range(0,n):

        v=ground_state(bias[i])
        if i==0:
            v_g=v
        else:
            v_g=np.kron(v_g,v)
        
    return v_g




def cost_evolution(p,QAOA_object,psi0):

    e=QAOA_object.costHe
    D=np.exp(-1j*p*e)
    psit=D*psi0

    return psit





def mixing_evolution(p,QAOA_object,psi0):

    HX=QAOA_object.HX
    HZ=QAOA_object.HZ


    bias=QAOA_object.bias

    n=QAOA_object.n

    a=np.sqrt(1+bias**2)
    

    cos_a=np.cos(p*np.ones(n))
    sin_a=np.sin(p)/a
    
    psit=cos_a[0]*psi0-1j*sin_a[0]*HX[0].dot(psi0)+1j*sin_a[0]*bias[0]*HZ[0].dot(psi0)

    
    for i in range(1,n):
        psit=cos_a[i]*psit-1j*sin_a[i]*HX[i].dot(psit)+1j*sin_a[i]*bias[i]*HZ[i].dot(psit)



    return psit



def period(para,period,center):
    while para<center-period/2 or para>center+period/2:
        if para<center-period/2:
            para=para+period
        elif para>center+period/2:
            para=para-period

    return para

def total_evolution(para,QAOA_object):
    
    level=QAOA_object.level

    for k in range(2*level):
        if k%2==0:
            para[k]=period(para[k],2*np.pi,0)
        else:
            para[k]=period(para[k],np.pi,0)
    

    gamma=np.array([para[2*j] for j in range(level)])
    beta=np.array([para[2*j+1] for j in range(level)])

    psit=cost_evolution(gamma[0],QAOA_object,QAOA_object.psi0)

    psit=mixing_evolution(beta[0],QAOA_object,psit)

    for j in range(1,level):
        psit=cost_evolution(gamma[j],QAOA_object,psit)   
        psit=mixing_evolution(beta[j],QAOA_object,psit)
    return psit


def matrix_to_qubit(v,n,Z):
 
    qubit=np.array([np.real(Z[i].dot(v).dot(v))for i in range(n)])
    return qubit


def qubit_to_matrix(q,n):
    state_0=np.array([1,0])
    state_1=np.array([0,1])
    
    for i in range(n):
        if q[i]==1:
            s=state_0
        elif q[i]==-1:
            s=state_1
        if i==0:
            v=s
        else:
            v=np.kron(v,s)
    return v


class QAOA:
    
    def __init__(self,n,level,costH_object,bias_field,learning_rate,H):

        self.n=n
 
        self.level=level
        #bias_flag is False, standard QAOA; bias_flag is True, adaptive bias QAOA 
        self.bias_flag=bias_field['flag']
 
        self.bias=bias_field['bias']


        self.learning_rate=learning_rate

        
        self.HX=H['X']
        self.HY=H['Y']
        self.HZ=H['Z']
        self.HI=H['I']
        

        self.costH_object=costH_object
        self.costHe=costH_object.costH_eigen_values

        self.psi_final=costH_object.psi_final
        self.psi_ground=costH_object.psi_ground
        

    def get_mixing_info(self):
        
        self.psi0=mixing_ground(self.bias,self.n)
        
        return 1  
    
     
    def get_expectation(self,para):
        
        f1=total_evolution(para,self)
              
 
        f2=self.costHe*f1
        f3=np.conjugate(f1)
        f4=np.dot(f3,f2)
                  
        self.psit=f1  
        return np.round(np.real(f4),14)


    
    def update_bias(self):
        bias_change=np.zeros(self.n)
        for k in range(self.n):
            a=self.HZ[k].dot(self.psit)
            
            fa=np.conjugate(self.psit)      
            ez=np.real(np.dot(fa,a))
            bias_change[k]= ez-self.bias[k]
            self.bias[k]+=self.learning_rate*bias_change[k]
        return bias_change
        
      
    def get_probability(self):
        

        d=self.costH_object.degeneracy
        P_exc=np.zeros(len(d))

        P=np.abs(np.dot(self.psi_final,self.psit))**2
        for j in range(len(d)):
            a1=int(np.sum(d[:j+1]))
            a2=int(np.sum(d[:j]))
            P_exc[j]=np.sum(P[:a1])-np.sum(P[:a2])
        
 
        return P_exc,P
        
        
    
    def get_bias_state(self):
        if self.bias_flag==True:
            bias_qubit=np.sign(self.bias)
            bias_state=qubit_to_matrix(bias_qubit,self.n)
            e=self.costHe*bias_state
            e=np.dot(bias_state,e)
            #e=e/self.costH_object.ground_energy
            
            P=np.sum(np.abs(np.dot(self.psi_ground,bias_state))**2)
            
        else:
            e=0
            P=0
        return [e,P]
    
    