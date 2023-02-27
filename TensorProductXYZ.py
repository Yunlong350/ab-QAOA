import numpy as np
from global_var import sigmax,sigmay,sigmaz
from scipy import sparse as se



sigmax=se.csr_matrix(sigmax)
sigmay=se.csr_matrix(sigmay)
sigmaz=se.csr_matrix(sigmaz)

class generate_HX:
    def __init__(self,nl):
        self.nl=nl
        
    def get_ith(self,i):
        if i==0:
            H_X3=se.kron(sigmax,se.identity(2**(self.nl-1),dtype='float64',format='csr'),format='csr')
        elif i==self.nl-1:
            H_X3=se.kron(se.identity(2**(self.nl-1),dtype='float64',format='csr'),sigmax,format='csr')
        else:
            H_X1=se.identity(2**i,dtype='float64',format='csr')
            H_X2=se.identity(2**(self.nl-i-1),dtype='float64',format='csr')
            H_X3=se.kron(se.kron(H_X1,sigmax),H_X2,format='csr')
        
        return H_X3
    
    
    
    
class generate_HY:
    def __init__(self,nl):
        self.nl=nl
        
    def get_ith(self,i):
        if i==0:
            H_Y3=se.kron(sigmay,se.identity(2**(self.nl-1),dtype='float64',format='csr'),format='csr')
        elif i==self.nl-1:
            H_Y3=se.kron(se.identity(2**(self.nl-1),dtype='float64',format='csr'),sigmay,format='csr')
        else:
            H_Y1=se.identity(2**(i),dtype='float64',format='csr')
            H_Y2=se.identity(2**(self.nl-i-1),dtype='float64',format='csr')
            H_Y3=se.kron(se.kron(H_Y1,sigmay),H_Y2,format='csr')
        
        return H_Y3  
    
class generate_HZ:
    def __init__(self,nl):
        self.nl=nl

    def get_ith(self,i):
        if i==0:
            H_Z3=se.kron(sigmaz,se.identity(2**(self.nl-1),dtype='float64',format='csr'),format='csr')
        elif i==self.nl-1:
            H_Z3=se.kron(se.identity(2**(self.nl-1),dtype='float64',format='csr'),sigmaz,format='csr')
        else:
            H_Z1=se.identity(2**(i),dtype='float64',format='csr')
            H_Z2=se.identity(2**(self.nl-i-1),dtype='float64',format='csr')
            H_Z3=se.kron(se.kron(H_Z1,sigmaz),H_Z2,format='csr')
        
        return H_Z3  








    




            


            






