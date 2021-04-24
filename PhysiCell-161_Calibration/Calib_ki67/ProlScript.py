import numpy as np
import matplotlib.pyplot as plt
from rk4 import rk4
from MCMC import ABC_MCMC
import sys
sys.path.append('../')
from pyMCDS import pyMCDS
from scipy.optimize import minimize

def KI67_Basic ( t, rf, Par ):
  Kn = rf[0]
  Kp = rf[1]
  
  r01 = Par[0]
  r10 = Par[1]
  
  dKn_dt =  -r01 * Kn + 2*r10 * Kp
  dKp_dt = r01 * Kn - r10 * Kp
  
  drf_dt = np.array ( [ dKn_dt, dKp_dt] )

  return drf_dt
  
def KI67_Advanced ( t, rf, Par ):
  Kn = rf[0]
  Kp1 = rf[1]
  Kp2 = rf[2]
  
  r01 = Par[0]
  r10 = Par[1]
  r12 = Par[2]
  r20 = Par[3]
  
  dKn_dt =  -r01 * Kn + r10 * Kp2
  dKp1_dt = r01 * Kn - r12 * Kp1
  dKp2_dt = 2*r12 * Kp1 - r20 * Kp2
  
  drf_dt = np.array ( [ dKn_dt, dKp1_dt, dKp2_dt] )

  return drf_dt

def Model_OLD(Par):
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
      if(Po2[index] < sigma_S): rate = ((Po2[index] - sigma_T)/(sigma_S - sigma_T))
    ParLocal[0] = Par[0]*rate
    t, ODE_out = rk4 ( KI67_Basic, tspan, InitialCond, n, ParLocal )
    QOI1[index,:] = ODE_out[:,0]/(ODE_out[:,0]+ODE_out[:,1])
    QOI2[index,:] = ODE_out[:,1]/(ODE_out[:,0]+ODE_out[:,1])
  return QOI1[-20,:], QOI2[-20,:]
  
  
def Model(Par):
  pho = 674/(347*347*15) # cell density
  cell_radius = 8.413 # 8.413
  tumor_radius = 3000.0 # 2000.0
  R_NC = 550.0 # necrotic core radius -- 570.0
  T = tumor_radius - R_NC # viable rim thickness
  f = pho*(4.0/3.0)*np.pi*(cell_radius**3) # confluence (pho*(4/3)*pi*R^3)
  print("confluence: "+str(f))
  Lambda = 0.2 # uptake rate in the confluent fraction of the viable rim
  Lambda_b = 0.001 # uptake rate in non-confluent portion of the rim
  D = 100000.0 # diffusion coefficient
  sigma_T = 6.0 # PO2 threshold viable rim
  lambdaViable = f*Lambda + (1-f)*Lambda_b # uptake rate in the viable rim
  lambdaCore = 0.1*f*Lambda + (1-f)*Lambda_b
  #lambdaCore = Lambda_b # uptake rate in the necrotic core
  Lv = np.sqrt(D/lambdaViable) # length scale in the viable rim
  Ln = np.sqrt(D/lambdaCore)# length scale in the necrotic core
  sigma_boundary = sigma_T*( np.cosh(T/Lv) + (Lv/Ln)*np.tanh(R_NC/Ln)*np.sinh(T/Lv))
  Po2 = ( Lv*sigma_T/T )*( np.sinh(T/Lv) + (Lv/Ln)*np.tanh(R_NC/Ln)*(np.cosh(T/Lv) - 1) )
  print("pho: "+str(pho))
  print("<sigma>: "+str(Po2))
  print("sigma_B: "+str(sigma_boundary))
  # Ki67 dynamic
  tspan = np.array ( [ 0.0, 6000.0 ] )
  InitialCond = np.array ( [ 28138.0, 0.0] )
  NumInterv = 300
  ParLocal = np.copy(Par)
  sigma_S = 38.0
  ParLocal[0] = Par[0]*((Po2 - sigma_T)/(sigma_S - sigma_T))
  t, ODE_out = rk4 ( KI67_Basic, tspan, InitialCond, NumInterv, ParLocal )
  # Cell fraction
  Ki67neg = np.zeros(NumInterv+1)
  Ki67pos = np.zeros(NumInterv+1)
  Ki67neg = ODE_out[:,0]/(ODE_out[:,0]+ODE_out[:,1])
  Ki67pos = ODE_out[:,1]/(ODE_out[:,0]+ODE_out[:,1])
  print("Times: "+str(t[-1])+" and "+str(t[-2]))
  print("Rate1: "+str(Ki67neg[-1]/Ki67neg[-2])+"  Rate2: "+str(Ki67pos[-1]/Ki67pos[-2]))
  return t,Ki67neg,Ki67pos,ODE_out, Po2

def Model2(Par):
  pho = 674/(347*347*50) # cell density
  cell_radius = 8.413 # 8.413
  tumor_radius = 2000.0 # 2000.0
  R_NC = 570.0 # necrotic core radius -- 570.0
  T = tumor_radius - R_NC # viable rim thickness
  f = pho*(4.0/3.0)*np.pi*(cell_radius**3) # confluence (pho*(4/3)*pi*R^3)
  Lambda = 1.2530 # uptake rate in the confluent fraction of the viable rim
  Lambda_b = 0.01 # uptake rate in non-confluent portion of the rim
  D = 100000.0 # diffusion coefficient
  sigma_T = 6.0 # PO2 threshold viable rim
  lambdaViable = f*Lambda + (1-f)*Lambda_b # uptake rate in the viable rim
  lambdaCore = 0.1*f*Lambda + (1-f)*Lambda_b
  #lambdaCore = Lambda_b # uptake rate in the necrotic core
  Lv = np.sqrt(D/lambdaViable) # length scale in the viable rim
  Ln = np.sqrt(D/lambdaCore)# length scale in the necrotic core
  sigma_boundary = sigma_T*( np.cosh(T/Lv) + (Lv/Ln)*np.tanh(R_NC/Ln)*np.sinh(T/Lv))
  Po2 = ( Lv*sigma_T/T )*( np.sinh(T/Lv) + (Lv/Ln)*np.tanh(R_NC/Ln)*(np.cosh(T/Lv) - 1) )
  # Ki67 dynamic
  tspan = np.array ( [ 0.0, 6000.0 ] )
  InitialCond = np.array ( [ 28138.0, 0.0] )
  NumInterv = 300
  Par = 1.0/(60.0*Par)
  ParLocal = np.array([Par[0],1.0/(60.0*27.1)])
  sigma_S = 38.0
  ParLocal[0] = Par[0]*((Po2 - sigma_T)/(sigma_S - sigma_T))
  t, ODE_out = rk4 ( KI67_Basic, tspan, InitialCond, NumInterv, ParLocal )
  # Cell fraction
  Ki67neg = np.zeros(NumInterv+1)
  Ki67pos = np.zeros(NumInterv+1)
  Ki67neg = ODE_out[:,0]/(ODE_out[:,0]+ODE_out[:,1])
  Ki67pos = ODE_out[:,1]/(ODE_out[:,0]+ODE_out[:,1])
  #print(Ki67pos[-1])
  return np.array([np.abs(Ki67pos[-1]- 0.37)])
  
r01 = 1.0/(9.03*60.0) # 8.51 min
r10 = 1.0/(19*60.0) # 19 min

time, Ki67neg, Ki67pos, ODE_out, Po2 = Model([r01,r10])
# Ki67negOLD, Ki67posOLD = Model_OLD([r01,r10])

f_Ki67p = 0.37
Std_f_Ki67p = 0.09

# Calibration
# UpperLimit = np.array([9.1,19.0])
# LowLimit = np.array([9.0,19.0])
# print(str(LowLimit)+" "+str(UpperLimit)+"\n")
# ABC_MCMC(Model2, np.array([f_Ki67p]), LowLimit, UpperLimit,'CalibProl.dat',0.0005,200)

# Optimization
x0 = np.array([20.0])
res = minimize(Model2, x0, method='nelder-mead', options={'maxiter': 2000, 'disp': True})
print(res)
exit()
print("Data: "+ str(f_Ki67p)+" Model: ",str(Ki67pos[-1]))

initial_index = 0;
last_index = 100;
Ki67neg_count = np.zeros( last_index+1 );
Ki67pos_count = np.zeros( last_index+1 );
dead_count = np.zeros( last_index+1 );
times = np.zeros( last_index+1 );

for n in range( initial_index,last_index+1 ):
  filename='output'+"%08i"%n+'.xml'
  mcds=pyMCDS(filename,'../outputCalibParameters')
  times[n]= mcds.get_time()
  
  cycle = mcds.data['discrete_cells']['cycle_model']
  cycle = cycle.astype( int )
  current_phase = mcds.data['discrete_cells']['current_phase']
  current_phase = current_phase.astype(int)
  elapsed_time_in_phase = mcds.data['discrete_cells']['elapsed_time_in_phase']
  cell_type = mcds.data['discrete_cells']['cell_type']
  cell_type = cell_type.astype(int)
  Ki67negModel = np.argwhere( (cycle < 100) & (cell_type==0) & (current_phase == 3) ).flatten()
  Ki67posModel = np.argwhere( (cycle < 100) & (cell_type==0) & (current_phase == 2)).flatten()
  dead = np.argwhere( (cycle >= 100) & (cell_type==0) ).flatten()
  
  print("Ki67-: "+str(len(Ki67negModel))+"  Ki67+: "+str(len(Ki67posModel))+"  dead: "+str(len(dead)) +"  Total: "+ str(len(cell_type)))
  total = len(Ki67negModel)+len(Ki67posModel)
  Ki67neg_count[n] = len(Ki67negModel)/total
  Ki67pos_count[n] = len(Ki67posModel)/total

#Model
plt.plot( times,Ki67neg_count,c='red',label='Ki67-')
plt.plot( times,Ki67pos_count,c='black',label='Ki67+')

# plt.plot( time,Ki67negOLD,linestyle='-.',c='red',label='Ki67-Old')
# plt.plot( time,Ki67posOLD,linestyle='-.',c='black',label='Ki67+Old')

#Calibration
plt.plot( time,Ki67neg,c='red',linestyle='dashed',label='<Ki67->')
plt.plot( time,Ki67pos,c='black',linestyle='dashed',label='<Ki67+>')
#plt.hlines(y=f_Ki67p, xmin=-50.0, xmax=6000, linewidth=1.0, color='blue',label='Data_Ki67+')
#plt.hlines(y=Ki67pos.mean(), xmin=-50.0, xmax=6000, linewidth=1.0, color='green',label='Data_Ki67+')
plt.ylabel('Cell fraction')
plt.xlabel('Time (minutes)')
plt.legend(loc='center', bbox_to_anchor=(0.5, 1.07),
          fancybox=True, shadow=True, ncol=5)

plt.figure(2,figsize=(4, 8))
plt.subplots_adjust(left=0.26,right=0.74)
plt.xlim(0,1.5)
plt.errorbar(0.5, f_Ki67p, yerr=Std_f_Ki67p, marker='>',
         mec='black', ms=10, mew=4, lw=6, c='black',label='Data') 
plt.errorbar(1.0, Ki67pos.mean(), yerr=Ki67pos.std(), marker='<',
         mec='red', ms=10, mew=4, lw=6, c='red',label='Model') 
#plt.errorbar(1.5, Ki67pos_count.mean(), yerr=Ki67pos_count.std(), marker='<', mec='red', ms=10, mew=4, lw=6, c='b',label='Model') 
plt.xticks([])
plt.xlabel('Cell fraction (Ki67+)')
#plt.ylabel('Percentage (%)')
plt.legend(loc='center', bbox_to_anchor=(0.5, 1.07),
          fancybox=True, shadow=True, ncol=5)
plt.show()