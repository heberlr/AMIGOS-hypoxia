import numpy as np
import matplotlib.pyplot as plt
import shlex,subprocess
from subprocess import DEVNULL, STDOUT, check_call
from MCMC import ABC_MCMC

def Model(Par):
  run = "./proliferation2D 3 "+str(Par[0])+" "+str(Par[1])+" "+str(Par[2])+" "+str(Par[3])
  args = shlex.split(run)
  p = subprocess.Popen(args,stdout=DEVNULL)
  if p.wait() != 0:
    print("There was an error")
  output = np.loadtxt("./output0003", dtype='f', delimiter='\t') 
  KI67 = np.array(output[:,1])
  KI67 = np.delete(KI67,-5,0)
  KI67[np.isnan(KI67)] = 0
  return KI67

#Data
Dist, KI67, KI67Std = [], [], []
for line in open('DataProl.dat', 'r'):
  values = [float(s) for s in line.split()]
  Dist.append(values[0])
  KI67.append(values[1])
  KI67Std.append(values[2])
Dist = np.array(Dist)
KI67 = np.array(KI67)
KI67Std = np.array(KI67Std)

UpperLimit = np.array([55.0,1.2,160.0,7.0])
LowLimit = np.array([45.0,0.8,140.0,5.0])
qoi = KI67

ABC_MCMC(Model, qoi, LowLimit, UpperLimit,'CalibProl.dat',40.0,1)