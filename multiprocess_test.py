from multiprocessing import Pool
import time
import numpy as np

class A:
    def __init__(self, a):
        self.a=a

    def f(self):
        np.random.seed()
        print("hey")
        self.a= np.random.rand()
        return self

    def run(self):
        with Pool(3) as p:
            print([i.a for i in p.map(A.f, [self,self,self])])


A(1).run()





























