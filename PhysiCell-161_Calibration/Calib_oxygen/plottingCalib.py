import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Data
with open('CalibOxy.dat', 'r') as f:
  P1 = []
  Ind = []
  AcT = []
  distance = []
  for each in f:
    each1 = each.split()
    P1.append(float(each1[0]))
    Ind.append(int(each1[1]))
    AcT.append(int(each1[2]))
    distance.append(float(each1[3]))

Par1 = np.array(P1)


sns.set()
sns.set_color_codes("deep")
sns.distplot(Par1,color='blue')
value1 = sns.distplot(Par1).get_lines()[0].get_data()
maxPar1 = value1[0][np.argmax(value1[1])]
#plt.title("MAP = %2.4f and Min = %2.4f" % (maxPar1,Par1[np.argmin(distance)]), fontsize=12)
plt.title("MAP = %2.4f" % (maxPar1), fontsize=12)
plt.xlabel('Cell uptake')
plt.savefig('Oxygen.png')

#Data
Dist, Oxygen, OxygenStd = [], [], []
for line in open('DataOxy.dat', 'r'):
  values = [float(s) for s in line.split()]
  Dist.append(values[0])
  Oxygen.append(values[1])
  OxygenStd.append(values[2])
Dist = np.array(Dist)
Oxygen = np.array(Oxygen)
OxygenStd = np.array(OxygenStd)

output = np.loadtxt("./output0010", dtype='f', delimiter='\t')
map1 = np.array(output[:,1])
# output = np.loadtxt("./output0011", dtype='f', delimiter='\t')
# max1 = np.array(output[:,1])

plt.clf()
plt.ylabel('$PO_2$ (mmHg)')
plt.xlabel('Distance from tumor core (mm)')
plt.errorbar(Dist, Oxygen, yerr=OxygenStd,fmt='o',color='gray')
plt.plot(Dist, map1,color='blue')
#plt.plot(Dist, max1,color='black')
plt.savefig('Result.png')