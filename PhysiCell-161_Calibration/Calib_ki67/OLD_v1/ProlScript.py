import numpy as np
import matplotlib.pyplot as plt
import shlex,subprocess
from subprocess import DEVNULL, STDOUT, check_call
from rk4 import rk4
# import sys
# sys.path.append('../')
from MCMC import ABC_MCMC

def KI67_Basic ( t, rf, Par ):
  Q = rf[0]
  K = rf[1]
  
  r01 = Par[0] #1.0/(4.59*60.0)
  r10 = Par[1] #1.0/(15.5*60.0)
  
  dQdt =  -r01 * Q + 2*r10 * K
  dKdt = r01 * Q - r10 * K
  
  drfdt = np.array ( [ dQdt, dKdt] )

  return drfdt

def Model(Par):
  Po2 = np.array([0.0864051, 0.0890906, 0.0929483, 0.0988798, 0.107035, 0.116854, 0.129048, 0.144141, 0.162745, 0.184965, 0.211073, 0.241984, 0.279285, 0.324073, 0.377286, 0.438583, 0.512567, 0.602552, 0.706005, 0.831655, 0.97609, 1.15126, 1.3635, 1.60932, 1.90144, 2.25161, 2.67429, 3.17616, 3.76633, 4.46569, 5.30917, 6.32467, 7.53771, 8.94839, 10.6579, 12.746, 15.1701, 18.1281, 21.5589, 25.7389, 30.8287, 36.7742, 45.94])
  tspan = np.array ( [ 0.0, 6000.0 ] )
  InitialCond = np.array ( [ 56827, 0.0] )
  n = 300
  QOI1 = np.zeros((len(Po2),n+1))
  QOI2 = np.zeros((len(Po2),n+1))
  ParLocal = np.copy(Par)
  #sigma_S = ParLocal[2]
  #sigma_T = ParLocal[3]
  sigma_S = 38.0
  sigma_T = 6.0
  for index in range( 0,len(Po2) ):
    rate = 1.0
    if(Po2[index] <= sigma_T): rate = 0.0
    else: 
      if(Po2[index] < sigma_S): rate = ((Po2[index] - sigma_T)/(sigma_S - sigma_T))**Par[2]
    ParLocal[0] = Par[0]*rate
    t, ODE_out = rk4 ( KI67_Basic, tspan, InitialCond, n, ParLocal )
    QOI1[index,:] = 100*ODE_out[:,0]/(ODE_out[:,0]+ODE_out[:,1])
    QOI2[index,:] = 100*ODE_out[:,1]/(ODE_out[:,0]+ODE_out[:,1])
  QOI2 = np.delete(QOI2,-5,0)
  return QOI2[-19:,-1]


#Data
X, Y, Ystd = [], [], []
for line in open('./Calib_ki67/DataProl.dat', 'r'):
  values = [float(s) for s in line.split()]
  X.append(values[0])
  Y.append(values[1])
  Ystd.append(values[2])
X = np.array(X)
Y = np.array(Y)
Ystd = np.array(Ystd)

#UpperLimit = np.array([0.01,0.002,20.0,0.5])
#LowLimit = np.array([0.0,0.0,10.0,0.0])
UpperLimit = np.array([0.01,0.002,2.0])
LowLimit = np.array([0.0,0.0,0.0])

N=19
qoi = 55*Y[-N:]
print(str(LowLimit)+" "+str(UpperLimit)+"\n")
ABC_MCMC(Model, qoi, LowLimit, UpperLimit,'./Calib_ki67/CalibProl.dat',30.0,200)
