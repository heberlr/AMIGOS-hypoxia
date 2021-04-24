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

figure, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2,figsize=(8,8))
sns.set()
sns.set_style("white")
#plt.subplot(121)
value1 = sns.distplot(RPar1,color='red',ax=ax1).get_lines()[0].get_data()
maxPar1 = value1[0][np.argmax(value1[1])]
# plt.title("MAP = %2.4f" % maxPar1, fontsize=12)
ax1.set_title("MAP = %2.2f" % (maxPar1),fontsize='large')
ax1.set_xlabel('Bias',fontsize=18)
ax1.set_ylabel('Density',fontsize=18)
ax1.tick_params(labelsize=18)

#plt.subplot(122)
value2 = sns.distplot(RPar2,color='red',ax=ax2).get_lines()[0].get_data()
maxPar2 = value2[0][np.argmax(value2[1])]
#plt.title("MAP = %2.4f and Min = %2.4f" % (maxPar1,RPar1[np.argmin(distance)]), fontsize=12)
ax2.set_title("MAP = %2.2f $\mu m /min$" % (maxPar2), fontsize='large')
ax2.set_xlabel('Speed ($\mu m /min$)', fontsize=18)
ax2.tick_params(labelsize=18)


sns.set()
sns.set_style("white")
#plt.subplot(121)
value1 = sns.distplot(GPar1,color='green',ax=ax3).get_lines()[0].get_data()
maxPar1 = value1[0][np.argmax(value1[1])]
ax3.set_title("MAP = %2.2f" % maxPar1, fontsize='large')
ax3.set_xlabel('Bias', fontsize=18)
ax3.set_ylabel('Density', fontsize=18)
ax3.tick_params(labelsize=18)
from matplotlib.ticker import FormatStrFormatter
ax3.yaxis.set_major_formatter(FormatStrFormatter('%d'))

#plt.subplot(122)
value2 = sns.distplot(GPar2,color='green',ax=ax4).get_lines()[0].get_data()
maxPar2 = value2[0][np.argmax(value2[1])]
#plt.title("MAP = %2.4f and Min = %2.4f" % (maxPar1,GPar1[np.argmin(distance)]), fontsize=12)
ax4.set_title("MAP = %2.2f $\mu m /min$" % (maxPar2),fontsize='large')
ax4.set_xlabel('Speed ($\mu m /min$)', fontsize=18)
ax4.tick_params(labelsize=18)

plt.subplots_adjust(left=0.10,right=0.95,bottom=0.08,top=0.95,wspace=0.3,hspace=0.35)

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
  
figure, ((ax1),(ax2)) = plt.subplots(nrows=2, ncols=1,figsize=(5,8))
p1 = ax1.errorbar(Temp*15.0, DsRedDisplacement, yerr=DsRedDisplacementSTD,fmt='o',color='gray',alpha=0.5,label='Data')
ax1.set_xticks(np.arange(0, 920, step=300.0))
p2 = ax1.plot(Temp*15.0, mapR,color='red',label='Model')
ax1.fill_between(Temp*15.0, mapR - mapRstd, mapR + mapRstd, alpha=0.2, color='red',label='Std_model')
p3 = ax1.fill(np.NaN, np.NaN, color='red', alpha=0.2)
#plt.plot(Temp*15.0, maxR,color='black')
ax1.set_xlabel('Time (min)',fontsize=18)
ax1.set_ylabel('Displacement ($\mu m$)',fontsize=18)
ax1.legend([(p3[0], p2[0]),p1 ], ['Model','Exp. data'], bbox_to_anchor=(0., 1.02, 1., .102), loc='center', ncol=3,fontsize='large')
print(p1)
ax1.tick_params(labelsize=18)
print("Distance RFP: "+str(np.linalg.norm(DsRedDisplacement-mapR)))

ax2.set_xticks(np.arange(0, 920, step=300.0))
p1 = ax2.errorbar(Temp*15.0, GFPDisplacement, yerr=GFPDisplacementSTD,fmt='o',color='gray',alpha=0.5,label='Data')
p2 = ax2.plot(Temp*15.0, mapG,color='green',label='Model')
ax2.fill_between(Temp*15.0, mapG - mapGstd, mapG + mapGstd, alpha=0.2, color='green',label='Std_model')
p3 = ax1.fill(np.NaN, np.NaN, color='green', alpha=0.2)
#plt.plot(Temp*15.0, maxG,color='black')
ax2.set_xlabel('Time (min)',fontsize=18)
ax2.set_ylabel('Displacement ($\mu m$)',fontsize=18)
ax2.legend([(p3[0], p2[0]),p1 ], ['Model','Exp. data'], bbox_to_anchor=(0., 1.02, 1., .102), loc='center', ncol=3,fontsize='large')
ax2.tick_params(labelsize=18)
print("Distance GFP: "+str(np.linalg.norm(GFPDisplacement-mapG)))

plt.subplots_adjust(left=0.18,right=0.96,bottom=0.08,top=0.95,hspace=0.40)
plt.show()