#! /usr/bin/env python3
#
def ABC_MCMC(Model, data, LowLimit, UpperLimit, FILE='CalibMCMC.dat', tol = 100, NumAccept = 100, max_iterations=100000, var_trasition=0.2):
  #*****************************************************************************
  #
  ## Markov chain Monte Carlo without likelihoods - Bayesian inference
  #  (Marjoram et al. - 2003 - Markov chain Monte Carlo without likelihoods)
  #
  #  Modified:
  #
  #  7 May 2020
  #
  #  Author:
  #
  #    Heber L. Rocha
  #
  #  Input:
  #
  #    function Model(Par): model with output compatible to observational data.
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
  theta = np.zeros((NumAccept,Npar))
  for j in range(0, Npar):
    theta_star[j] = np.random.uniform(LowLimit[j],UpperLimit[j])
  for i in range(0, max_iterations):
    output_model = Model(theta_star)
    distance = np.sqrt(np.sum([(a - b)**2 for a, b in zip(output_model, data)]))
    print(str(count+1)+"/"+str(i+1)+" -- distance: "+str(distance)+" "+ str(theta_star)+"\n")
    if (distance < tol or count == 0):
        theta[count,:] = theta_star
        count = count + 1
        for j in range(0, Npar):
          file.write(str(theta_star[j])+" ")
        file.write(str(count)+" "+str(i+1)+" "+str(distance)+"\n")
    if (count == NumAccept):
        break
    cond = True
    while(cond):
      noise = np.random.normal(0, var_trasition*(UpperLimit-LowLimit))
      theta_star = theta[count-1,:] + noise
      cond = [False for k in range(0,Npar) if theta_star[k]>UpperLimit[k] or theta_star[k]<LowLimit[k]]
  file.close()
  return theta


def ABC_SMC(Model, data, LowLimit, UpperLimit, FILE='CalibSMC.dat', tol = ([98,99,100]), NumAccept = 100, max_iterations=100000, var_trasition=0.2):
  #*****************************************************************************
  #
  ## Approximate  Bayesian Computation - Sequential Monte Carlo - Bayesian inference
  #  (Del Moral et al. - 2006 - Sequential Monte Carlo samplers)
  #
  #  Modified:
  #
  #  29 May 2020
  #
  #  Author:
  #
  #    Heber L. Rocha
  #
  #  Input:
  #
  #    function Model(Par): model with output compatible to observational data.
  #
  #    real data[n]: contains the observational data
  #
  #    real LowLimit[numpar]: lower limit for parameters.
  #
  #    real UpperLimit[numpar]: upper limit for parameters.
  #
  #    string FILE: output file name.
  #
  #    array real tol: tolerance vector to evaluate the distance between observational and model data (must be in ascending order). The size of this vector is the number of populations.
  #    
  #    int NumAccept: the number of accepted parameters that it will generate a posterior distribution (for each population).
  #
  #    int max_iterations: the number max of execution of model for each population.
  #     
  #    real var_trasition: variance of the normal distribution for sampling.
  
  import numpy as np
  from scipy.stats import multivariate_normal
  
  file = open(FILE,"w") 
  Npar = UpperLimit.shape[0]
  Npop = tol.shape[0]
  theta_star = np.zeros(Npar)
  theta_ant = np.zeros((NumAccept,Npar))
  weight_prev = np.ones(NumAccept)
  weight = np.ones(NumAccept)
  Kernel = np.zeros(NumAccept)
  count = np.zeros((NumAccept,2), dtype=int)
  dist = np.zeros(NumAccept)
  count_total = 0
  #Loop populations
  for k in range(0, Npop):
    count_pop = 0
    # Generate posterior of the population k
    for i in range(0, max_iterations):
        cond = True
        while(cond):
            if (k == 0):
              for j in range(0, Npar):
                theta_star[j] = np.random.uniform(LowLimit[j],UpperLimit[j])
            else:
              index = np.random.choice(NumAccept,p=weight_prev)
              theta_star = theta_ant[index,:] + np.random.normal(0, var_trasition*(UpperLimit-LowLimit))
            cond = [False for k in range(0,Npar) if theta_star[k]>UpperLimit[k] or theta_star[k]<LowLimit[k]]
        output_model = Model(theta_star)
        distance = np.sqrt(np.sum([(a - b)**2 for a, b in zip(output_model, data)]))
        print(str(count_pop+1)+"/"+str(i+1)+" -- distance: "+str(distance)+" "+ str(theta_star)+"\n")
        # Number total of executions
        count_total = count_total + 1
        if (distance < tol[k]):
            if (k == 0): weight[count_pop] = 1.0
            else:
              for j in range(0, NumAccept): Kernel[j] = np.linalg.norm(multivariate_normal.pdf(theta_ant[j,:], mean=theta_star, cov=var_trasition))
              weight[count_pop] = 1.0/np.sum(weight_prev*Kernel)
            # Add to vector count and distance
            count[count_pop,:] = np.array([count_pop+1,count_total])
            dist[count_pop] = distance
            # Add sample to population
            theta_ant[count_pop,:] = theta_star
            count_pop = count_pop + 1
        if (count_pop == NumAccept):
            break
    # Normalize weights 
    weight /= np.sum(weight)
    weight[np.isnan(weight)] = 0.0
    weight[-1] += np.abs(1.0-np.sum(weight))
    weight_prev = np.copy(weight)

  # Print last population
  for i in range(0, NumAccept):
    for j in range(0, Npar):
      file.write(str(theta_ant[i,j])+" ")
    file.write(str(count[i,0])+" "+str(count[i,1])+" "+str(dist[i])+"\n")  
  file.close()
  return theta_ant