import numpy as np
from scipy.integrate import odeint
from ABC import ABC_SMC,ABC_MCMC
import matplotlib.pyplot as plt
import seaborn as sns
 
def OldModel(protein, t, par):
    # u steps from 0 to 1 at t=10
    if t < par[2]:
        gene1 = 1
        gene2 = 0
    else:
        gene1 = 0
        gene2 = 1
    proteinRed=gene1*(par[0] + par[1]) - protein[0]*(gene1*par[0] + par[1])
    proteinGreen= gene2*(par[0] + par[1]) - protein[1]*(gene2*par[0] + par[1])
    
    return [proteinRed,proteinGreen]

def NewModel(protein, t, par):
    # u steps from 0 to 1 at t=10
    if t < par[2]:
        gene1 = 1
        gene2 = 0
    else:
        gene1 = 0
        gene2 = 1
    proteinRed= gene1*par[0]*(1.0 - protein[0]) - protein[0]*par[1]*(1.0 - gene1)
    proteinGreen= gene2*par[0]*(1.0 - protein[1]) - protein[1]*par[1]*(1.0 - gene2)
    
    return [proteinRed,proteinGreen]

def PlotFigure(Par):    
    # Define time interval, parameter value, and initial condition
    times = np.arange(0,14400,0.5)

    # a = 4.8e-4
    # b = 6.8e-5
    # c = 100.0
    
    a = Par[0]
    b = Par[1]
    c = Par[2]

    #Initial Condition
    proteinRed_ic = 1
    proteinGreen_ic = 0

    # Solving the ODE model
    # y = odeint(OldModel, t=times, y0=[proteinRed_ic,proteinGreen_ic], args=tuple([[a,b,c]]))
    y2 = odeint(NewModel, t=times, y0=[proteinRed_ic,proteinGreen_ic], args=tuple([[a,b,c]]))

    # print(times[5760])
    # print(y2[5760,0])
    # print(y2[5760,1])
    
    # Data observational
    DataObs = []
    for line in open('Protein.dat', 'r'):
        values = [float(s) for s in line.split()]
        DataObs.append(values)
    DataObs = np.array(DataObs)
    Data = np.concatenate((DataObs[:,1], DataObs[:,2]), axis=None)
    DataObs[:,0] = DataObs[:,0]*1440.0
    
    # Plotting the tumor solution
    fig, ax = plt.subplots(dpi=120)
    plt.scatter(DataObs[:,0], DataObs[:,1], label = 'Data_DsRed+', color = 'red' )
    plt.scatter(DataObs[:,0], DataObs[:,2], label = 'Data_GFP+', color = 'green' )
    # plt.plot(times, y[:,0], label='GFP+', color='red',linewidth=2.0)
    # plt.plot(times, y[:,1], label='DsRed+', color='green',linewidth=2.0)
    plt.plot(times, y2[:,0], label='DsRed+', color='red',linewidth=2.0,linestyle='dashed')
    plt.plot(times, y2[:,1], label='GFP+', color='green',linewidth=2.0,linestyle='dashed')
    plt.legend()
    plt.xlabel('Time (min)')
    plt.ylabel('Protein expression');
    #plt.savefig("solution.pdf")
    plt.show()

def CalibrationMaxLikEst():
    def log_likelihood(theta, times, y):
        cl_alpha, cal_beta, cal_switchTime, cal_sigma = theta
        model = odeint(control_tumor, t=times, y0=[1,0], args=tuple([[cal_r,cal_K, cal_switchTime]]))
        variance = cal_sigma*cal_sigma
        return -0.5 * np.sum(np.log(2*np.pi)+np.log(variance) + (y - model) ** 2 / variance)
    
    nll = lambda *args: -log_likelihood(*args)
    initial = [0.01, 0.01, 600.0]
    bounds = Bounds([0.0, 0.0, 60], [1.0, 1.0, 10000.0])

    ml_alpha, ml_beta, ml_switchTime, ml_sigma = minimize(nll, initial, args=(times, DataObs), bounds=bounds)
    print("Maximum likelihood estimates:")
    print("alpha = %f" % ml_alpha)
    print("beta = %f" % ml_beta)
    print("swith time = %f" % ml_switchTime)
    print("Standard deviation = %f" % ml_sigma)

def ABCCalibration():
  UpperLimit = np.array([3e-4,2e-4, 2500.0])
  LowLimit = np.array([2e-4,1e-4, 2000.0])
  # Data observational
  DataObs = []
  for line in open('Protein.dat', 'r'):
    values = [float(s) for s in line.split()]
    DataObs.append(values)
  DataObs = np.array(DataObs)
  Data = np.concatenate((DataObs[:,1], DataObs[:,2]), axis=None)
  ABC_MCMC(ModelToABC,Data,LowLimit,UpperLimit,tol=0.3,NumAccept=5000)


def ModelToABC(Par, Nqoi=8):
    times = ([0, 5*1440, 7*1440, 10*1440])
    y2 = odeint(NewModel, t=times, y0=[1,0], args=tuple([[Par[0],Par[1],Par[2]]]))
    QOI = np.concatenate((y2[:,0], y2[:,1]), axis=None)
    return QOI

def PlotPosterior(file, Npar, color, plot = True): 
  # Read file
  input = np.loadtxt(file, dtype='f', delimiter=' ')
  # Store in a matrix
  Par_ant = np.array(input[:,0])
  for i in range(1, Npar):    
    MatrixPar = np.column_stack((Par_ant,np.array(input[:,i])))
    Par_ant = MatrixPar
  # Plot
  sns.set()
  sns.set_style('white')
  MAP = np.zeros(MatrixPar.shape[1])  
  fig, ax = plt.subplots(1,MatrixPar.shape[1])
  for i in range(0, MatrixPar.shape[1]): 
    
    value = sns.distplot(MatrixPar[:,i],color=color,ax=ax[i]).get_lines()[0].get_data()
    MAP[i] = value[0][np.argmax(value[1])]
    ax[i].set_title("MAP = %2.4f" % (MAP[i]), fontsize=18)
    ax[i].set_xlabel("Parameter %d" % (i+1),fontsize=18)
    ax[i].set_ylabel('Density',fontsize=18)
  if (plot): plt.show()
  else: plt.close()
  return MAP
  
    

#ABCCalibration()
MAP = PlotPosterior('CalibMCMC.dat',3,'gray')
print(MAP)
PlotFigure(MAP)


