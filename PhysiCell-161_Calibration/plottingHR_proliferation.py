from pyMCDS import pyMCDS
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./Calib_ki67')
from rk4 import rk4

def KI67_Basic ( t, rf, Par ):
  Q = rf[0]
  K = rf[1]
  
  r01 = Par[0] #1.0/(4.59*60.0)
  r10 = Par[1] #1.0/(15.5*60.0)
  
  dQdt =  -r01 * Q + 2*r10 * K
  dKdt = r01 * Q - r10 * K
  
  drfdt = np.array ( [ dQdt, dKdt] )

  return drfdt

def Model(Par,Oxygen):
  Po2 = Oxygen
  tspan = np.array ( [ 0.0, 12000.0 ] )
  InitialCond = np.array ( [ 497.0, 0.0] )
  n = 50
  time = np.zeros((len(Po2),n+1))
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
    QOI1[index,:] = ODE_out[:,0]/(ODE_out[:,0]+ODE_out[:,1])
    QOI2[index,:] = ODE_out[:,1]/(ODE_out[:,0]+ODE_out[:,1])
    time = t
  MeanQoi1 = QOI1.mean(axis = 0)
  StdQoi1 = QOI1.std(axis = 0)
  MeanQoi2 = QOI2.mean(axis = 0)
  StdQoi2 = QOI2.std(axis = 0)   
  return time,MeanQoi1,MeanQoi2,StdQoi1,StdQoi2

initial_index = 0;
last_index = 200;
Ki67neg_count = np.zeros( last_index+1 );
Ki67pos_count = np.zeros( last_index+1 );
dead_count = np.zeros( last_index+1 );
times = np.zeros( last_index+1 );
Oxygen = np.zeros(50)

# Col1, Col2, Col3 = [], [], []
# for line in open('DataProl.dat', 'r'):
  # values = [float(s) for s in line.split()]
  # Col1.append(values[0])
  # Col2.append(values[1])
  # Col3.append(values[2])

for n in range( initial_index,last_index+1 ):
  filename='output'+"%08i"%n+'.xml'
  mcds=pyMCDS(filename,'output')
  times[n]= mcds.get_time()
  
  cycle = mcds.data['discrete_cells']['cycle_model']
  cycle = cycle.astype( int )
  current_phase = mcds.data['discrete_cells']['current_phase']
  current_phase = current_phase.astype(int)
  elapsed_time_in_phase = mcds.data['discrete_cells']['elapsed_time_in_phase']
  cell_type = mcds.data['discrete_cells']['cell_type']
  cell_type = cell_type.astype(int)
  Ki67neg = np.argwhere( (cycle < 100) & (cell_type==0) & (current_phase == 3) ).flatten()
  Ki67pos = np.argwhere( (cycle < 100) & (cell_type==0) & (current_phase == 2)).flatten()
  dead = np.argwhere( (cycle >= 100) & (cell_type==0) ).flatten()
  
  print("Ki67-: "+str(len(Ki67neg))+"  Ki67+: "+str(len(Ki67pos))+"  dead: "+str(len(dead)) +"  Total: "+ str(len(cell_type)))
  total = len(Ki67neg)+len(Ki67pos)
  Ki67neg_count[n] = len(Ki67neg)/total
  Ki67pos_count[n] = len(Ki67pos)/total
  #dead_count[n] = len(dead)/len(cell_type)
  for i in range( 0,len(Oxygen) ):
    y_pos = i*20
    Oxygen[i] = mcds.get_concentrations_at(x=0., y=y_pos, z=0.)

Par = np.array([0.0010,0.0017, 0.2857])
timeModel, Qoi1, Qoi2, StdQoi1, StdQoi2 = Model(Par,Oxygen)
   
plt.ylabel('Cell fraction')
plt.xlabel('Time (hours)')
plt.plot( times/60.0,Ki67neg_count,c='blue',label='Ki67-')
plt.plot( times/60.0,Ki67pos_count,c='black',label='Ki67+')
#plt.plot( times/60.0,dead_count,c='red',label='Dead')
plt.errorbar( timeModel/60.0,Qoi1,yerr=StdQoi1,c='blue',linestyle='dashed',label='ODE_Ki67-')
plt.errorbar( timeModel/60.0,Qoi2,yerr=StdQoi2,c='black',linestyle='dashed',label='ODE_Ki67+')
plt.legend()
plt.show()
