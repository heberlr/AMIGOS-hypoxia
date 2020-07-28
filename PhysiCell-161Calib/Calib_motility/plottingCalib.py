import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Data Red
with open('CalibMotR.dat', 'r') as f:
  RP1 = []
  RP2 = []
  #RP3 = []
  Ind = []
  AcT = []
  distance = []
  for each in f:
    each1 = each.split()
    RP1.append(float(each1[0]))
    RP2.append(float(each1[1]))
    #RP3.append(float(each1[2]))
    Ind.append(int(each1[2]))
    AcT.append(int(each1[3]))
    distance.append(float(each1[4]))

RPar1 = np.array(RP1)
RPar2 = np.array(RP2)
#RPar3 = np.array(RP3)

#Data Green
with open('CalibMotG.dat', 'r') as f:
  GP1 = []
  GP2 = []
  #GP3 = []
  Ind = []
  AcT = []
  distance = []
  for each in f:
    each1 = each.split()
    GP1.append(float(each1[0]))
    GP2.append(float(each1[1]))
    #GP3.append(float(each1[2]))
    Ind.append(int(each1[2]))
    AcT.append(int(each1[3]))
    distance.append(float(each1[4]))

GPar1 = np.array(GP1)
GPar2 = np.array(GP2)
#GPar3 = np.array(GP3)

#Plotting

#figure, axes = plt.subplots(nrows=1, ncols=3,figsize=(12,6))
sns.set()
sns.set_style("white")
plt.subplot(121)
value1 = sns.distplot(RPar1,color='red').get_lines()[0].get_data()
maxPar1 = value1[0][np.argmax(value1[1])]
# plt.title("MAP = %2.4f" % maxPar1, fontsize=12)
plt.title("MAP = %2.4f" % (maxPar1), fontsize=12)
plt.xlabel('Bias')
plt.ylabel('Density')

plt.subplot(122)
value2 = sns.distplot(RPar2,color='red',).get_lines()[0].get_data()
maxPar2 = value2[0][np.argmax(value2[1])]
#plt.title("MAP = %2.4f and Min = %2.4f" % (maxPar1,RPar1[np.argmin(distance)]), fontsize=12)
plt.title("MAP = %2.4f" % (maxPar2), fontsize=12)
plt.xlabel('Speed')

# plt.subplot(133)
# sns.set_style('darkgrid')
# sns.distplot(Par3,color='green')
# value3 = sns.distplot(Par3).get_lines()[0].get_data()
# maxPar3 = value3[0][np.argmax(value3[1])]
# plt.title("MAP = %2.4f" % maxPar3,fontsize=12)
# plt.xlabel('Speed')
# figure.tight_layout(pad=1.0)
plt.savefig('RedMotility.png')

plt.clf()
#figure, axes = plt.subplots(nrows=1, ncols=3,figsize=(12,6))
sns.set()
sns.set_style("white")
plt.subplot(121)
value1 = sns.distplot(GPar1,color='green').get_lines()[0].get_data()
maxPar1 = value1[0][np.argmax(value1[1])]
plt.title("MAP = %2.4f" % maxPar1, fontsize=12)
plt.xlabel('Bias')
plt.ylabel('Density')

plt.subplot(122)
value2 = sns.distplot(GPar2,color='green').get_lines()[0].get_data()
maxPar2 = value2[0][np.argmax(value2[1])]
#plt.title("MAP = %2.4f and Min = %2.4f" % (maxPar1,GPar1[np.argmin(distance)]), fontsize=12)
plt.title("MAP = %2.4f" % (maxPar2), fontsize=12)
plt.xlabel('Speed')

# plt.subplot(133)
# sns.distplot(Par3,color='green')
# value3 = sns.distplot(Par3).get_lines()[0].get_data()
# maxPar3 = value3[0][np.argmax(value3[1])]
# plt.title("MAP = %2.4f" % maxPar3,fontsize=12)
# plt.xlabel('Speed')
# figure.tight_layout(pad=1.0)
plt.savefig('GreenMotility.png')

# Plotting the results
#Data
Temp, DsRedDisplacement, DsRedDisplacementSTD, GFPDisplacement, GFPDisplacementSTD = [], [], [], [], []
for line in open('./Data_disp.dat', 'r'):
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

output = np.loadtxt("./output0010", dtype='f', delimiter='\t')
mapR = np.array(output[:,1])
mapRstd = np.array(output[:,2])
# output = np.loadtxt("./output0011", dtype='f', delimiter='\t')
# maxR = np.array(output[:,1])
# maxRstd = np.array(output[:,2])
output = np.loadtxt("./output0020", dtype='f', delimiter='\t')
mapG = np.array(output[:,1])
mapGstd = np.array(output[:,2])
# output = np.loadtxt("./output0021", dtype='f', delimiter='\t')
# maxG = np.array(output[:,1])
# maxGstd = np.array(output[:,2])
  
plt.clf()
plt.errorbar(Temp*15.0, DsRedDisplacement, yerr=DsRedDisplacementSTD,fmt='o',color='gray',alpha=0.5,label='Data')
plt.xticks(np.arange(0, 920, step=100.0))
plt.plot(Temp*15.0, mapR,color='red',label='Model')
plt.fill_between(Temp*15.0, mapR - mapRstd, mapR + mapRstd, alpha=0.2, color='red',label='Std of the model')
#plt.plot(Temp*15.0, maxR,color='black')
plt.xlabel('Time (min)')
plt.ylabel('Displacement ($\mu m$)')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center', ncol=3)
print("Distance RFP: "+str(np.linalg.norm(DsRedDisplacement-mapR)))
plt.savefig('ResultR.png')

plt.clf()
plt.xticks(np.arange(0, 920, step=100.0))
plt.errorbar(Temp*15.0, GFPDisplacement, yerr=GFPDisplacementSTD,fmt='o',color='gray',alpha=0.5,label='Data')
plt.plot(Temp*15.0, mapG,color='green',label='Model')
plt.fill_between(Temp*15.0, mapG - mapGstd, mapG + mapGstd, alpha=0.2, color='green',label='Std of the model')
#plt.plot(Temp*15.0, maxG,color='black')
plt.xlabel('Time (min)')
plt.ylabel('Displacement ($\mu m$)')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center', ncol=3)
print("Distance GFP: "+str(np.linalg.norm(GFPDisplacement-mapG)))
plt.savefig('ResultG.png')