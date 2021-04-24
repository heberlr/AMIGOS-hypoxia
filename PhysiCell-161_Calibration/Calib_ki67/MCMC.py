#! /usr/bin/env python3
#
def ABC_MCMC(Model, data, LowLimit, UpperLimit, FILE='CalibMCMC.dat', tol = 100, NumAccept = 1000, max_iterations=100000, var_trasition=0.2):
  #*****************************************************************************
  #
  ## Markov chain Monte Carlo without likelihoods - Bayesian inference
  #
  #  Modified:
  #
  #  16 May 2020
  #
  #  Author:
  #
  #    Heber L. Rocha
  #
  #  Input:
  #
  #    function Model(Par): model with outpu compatible to observational data.
  #
  #    real data[n]: contains the observational data
  #
  #    real LowLimit[numpar]: lower limit for parameters.
  #
  #    real UpperLimit[numpar]: upper limit for parameters.
  #
  #    string FILE: output file name.
  #
  #    real tol: tolerance between observational and model data.
  #    
  #    int NumAccept: the number of accepted parameters that it will generate a posterior distribution.
  #
  #    int max_iterations: the number max of execution of model.
  #     
  #    real var_trasition: variance of the normal distribution for sampling.
  
  import numpy as np

  file = open(FILE,"w") 
  count = 0
  Npar = UpperLimit.shape[0]
  theta_star = np.zeros(Npar)
  for j in range(0, Npar):
    theta_star[j] = np.random.uniform(LowLimit[j],UpperLimit[j])
  for i in range(0, max_iterations):
    output_model = Model(theta_star)
    distance = np.sqrt(np.sum([(a - b)**2 for a, b in zip(output_model, data)]))
    print(str(count)+"/"+str(i)+" -- distance: "+str(distance)+" "+ str(theta_star)+"\n")
    if (distance < tol or count == 0):
        theta_star1 = theta_star
        count = count + 1
        for j in range(0, Npar):
          file.write(str(theta_star[j])+" ")
        file.write(str(count)+" "+str(i)+" "+str(distance)+"\n")
    if (count == NumAccept):
        break
    cond = True
    #print(distance)
    while(cond):
      noise = np.random.normal(0, var_trasition*(UpperLimit-LowLimit))
      theta_star = theta_star1 + noise
      cond = [False for k in range(0,Npar) if theta_star[k]>UpperLimit[k] or theta_star[k]<LowLimit[k]]
  file.close()