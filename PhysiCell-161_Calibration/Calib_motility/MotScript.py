import numpy as np
import matplotlib.pyplot as plt
import shlex,subprocess
from subprocess import DEVNULL, STDOUT, check_call
from MCMC import ABC_MCMC

def ModelRed(Par):
  run = "./motility2D.exe 2 15.0 "+str(Par[0])+" "+str(Par[1])
  #print(run)
  args = shlex.split(run)
  p = subprocess.Popen(args,stdout=DEVNULL)
  if p.wait() != 0:
    print("There was an error")
  output = np.loadtxt("./output0002", dtype='f', delimiter='\t')
  speed = np.array(output[:,1])
  return speed
  
def ModelGreen(Par):
  run = "./motility2D.exe 2 15.0 "+str(Par[0])+" "+str(Par[1])
  #print(run)
  args = shlex.split(run)
  p = subprocess.Popen(args,stdout=DEVNULL)
  if p.wait() != 0:
    print("There was an error")
  output = np.loadtxt("./output0002", dtype='f', delimiter='\t')
  speed = np.array(output[:,1])
  return speed

#Data
Temp, DsRedDisplacement, DsRedDisplacementSTD, GFPDisplacement, GFPDisplacementSTD = [], [], [], [], []
for line in open('./Calib_motility/Data_disp.dat', 'r'):
  values = [float(s) for s in line.split()]
  Temp.append(values[0])
  DsRedDisplacement.append(values[1])
  DsRedDisplacementSTD.append(values[2])
  GFPDisplacement.append(values[3])
  GFPDisplacementSTD.append(values[4])
Temp = np.array(Temp)
DsRedDisplacement = np.array(DsRedDisplacement)
DsRedDisplacementSTD = np.array(DsRedDisplacementSTD)
GFPDisplacement = np.array(GFPDisplacement)
GFPDisplacementSTD = np.array(GFPDisplacementSTD)

UpperLimit = np.array([0.5,0.5])#1.0
LowLimit = np.array([0.0,0.0])
qoiR = DsRedDisplacement
qoiG = GFPDisplacement

ABC_MCMC(ModelRed, qoiR,LowLimit, UpperLimit,'./Calib_motility/CalibMotR.dat',15.0,200)#50.0
ABC_MCMC(ModelGreen, qoiG,LowLimit, UpperLimit,'./Calib_motility/CalibMotG.dat',20.0,200)#50.0
#print(ModelGreen([0.2]))
#input('Press ENTER to exit')
