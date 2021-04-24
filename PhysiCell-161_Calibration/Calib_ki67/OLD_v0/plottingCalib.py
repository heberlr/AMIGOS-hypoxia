import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def Model(Par):   
  #Po2 = np.array([4.60234, 4.74557, 4.95134, 5.26767, 5.70256, 6.22612, 6.87626, 7.68088, 8.67263, 9.85707, 11.2488, 12.8964, 14.8848, 17.2725, 20.1098, 23.3787, 27.324, 32.1182, 37.6101, 45.94])
  Po2 = np.array([0.0864051, 0.0890906, 0.0929483, 0.0988798, 0.107035, 0.116854, 0.129048, 0.144141, 0.162745, 0.184965, 0.211073, 0.241984, 0.279285, 0.324073, 0.377286, 0.438583, 0.512567, 0.602552, 0.706005, 0.831655, 0.97609, 1.15126, 1.3635, 1.60932, 1.90144, 2.25161, 2.67429, 3.17616, 3.76633, 4.46569, 5.30917, 6.32467, 7.53771, 8.94839, 10.6579, 12.746, 15.1701, 18.1281, 21.5589, 25.7389, 30.8287, 36.7742, 45.94])
  QOI = np.ones(20)
  QOI = (Po2 - Par[1])/(Par[0] - Par[1])
  QOI[QOI < 0] = 0.0
  QOI[QOI > 1] = 1.0
  return QOI
  
#Data
with open('CalibMCMC.dat', 'r') as f:
  P1 = []
  P2 = []
  Ind = []
  AcT = []
  distance = []
  for each in f:
    each1 = each.split()
    P1.append(float(each1[0]))
    P2.append(float(each1[1]))
    Ind.append(int(each1[2]))
    AcT.append(int(each1[3]))
    distance.append(float(each1[4]))

Par1 = np.array(P1)
Par2 = np.array(P2)

print("Min distance: "+str(distance[np.argmin(distance)])+" - Par:"+str(Par1[np.argmin(distance)])+" "+str(Par2[np.argmin(distance)])+"\n")

#Plotting
# figure, axes = plt.subplots(nrows=1, ncols=3)
# plt.subplot(131)
# n, bins, patches = plt.hist(x=Par1, bins='auto', color='#009900',
              # alpha=0.7, rwidth=0.85)
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('$Persistence$ $time$')
# maxfreq1 = n.max()
# plt.ylim(ymax=np.ceil(maxfreq1 / 10) * 10 if maxfreq1 % 10 else maxfreq1 + 10)
# ProbPar1 = bins[np.argmax(n)]

# plt.subplot(132)
# n, bins, patches = plt.hist(x=Par2, bins='auto', color='#009900',
              # alpha=0.7, rwidth=0.85)
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('$Bias$')
# maxfreq2 = n.max()
# plt.ylim(ymax=np.ceil(maxfreq2 / 10) * 10 if maxfreq2 % 10 else maxfreq2 + 10)
# #plt.xlim((0.0, 0.035)) 
# ProbPar2 = bins[np.argmax(n)]

# plt.subplot(133)
# n, bins, patches = plt.hist(x=Par3, bins='auto', color='#009900',
              # alpha=0.7, rwidth=0.85)
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('$Speed$')
# maxfreq2 = n.max()
# plt.ylim(ymax=np.ceil(maxfreq2 / 10) * 10 if maxfreq2 % 10 else maxfreq2 + 10)
# #plt.xlim((0.0, 0.035)) 
# ProbPar3 = bins[np.argmax(n)]
# plt.suptitle("MAP = %2.4f ; %2.4f ; %2.4f" % (ProbPar1, ProbPar2, ProbPar3),fontsize=12)
# figure.tight_layout(pad=2.0)
# plt.savefig('HistPar.png')

figure, axes = plt.subplots(nrows=1, ncols=2,figsize=(12,6))

sns.set()
sns.set_color_codes("deep")
plt.subplot(121)
sns.distplot(Par1,color='blue')
value1 = sns.distplot(Par1).get_lines()[0].get_data()
maxPar1 = value1[0][np.argmax(value1[1])]
plt.ylabel('Frequency')
plt.title("MAP = %2.4f and Min = %2.4f" % (maxPar1,Par1[np.argmin(distance)]), fontsize=12)
plt.xlabel('Prol_saturation')

plt.subplot(122)
sns.set_style('darkgrid')
sns.distplot(Par2,color='blue')
value2 = sns.distplot(Par2).get_lines()[0].get_data()
maxPar2 = value2[0][np.argmax(value2[1])]
plt.title("MAP = %2.4f and Min = %2.4f" % (maxPar2,Par2[np.argmin(distance)]), fontsize=12)
plt.xlabel('Prol_threshold')
figure.tight_layout(pad=1.0)
plt.savefig('Ki67.png')

plt.clf()
fig = plt.figure()
X, Y, Ystd = [], [], []
for line in open('DataProl.dat', 'r'):
  values = [float(s) for s in line.split()]
  X.append(values[0])
  Y.append(values[1])
  Ystd.append(values[2])
plt.errorbar(X, Y, yerr=Ystd,ls='none')
plt.plot(X,Model([maxPar1,maxPar2]),color='blue')
plt.plot(X,Model([Par1[np.argmin(distance)],Par2[np.argmin(distance)]]),color='red')
plt.xlabel('Distance from the tumor core (mm)')
plt.ylabel('%Ki67')
plt.savefig('Result.png')