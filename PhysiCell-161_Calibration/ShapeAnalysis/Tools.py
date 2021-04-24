import numpy as np
#import math
#import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection
import seaborn as sns
from pyMCDS import pyMCDS
import numpy as np
from astroML.correlation import two_point
from astroML.correlation import bootstrap_two_point
import cv2

def ReadFractionGreenCells(perct,initial_index = 0, last_index = 130):
  NumberGreen = np.zeros( (last_index-initial_index)+1 );
  NumberGreenResp = np.zeros( (last_index-initial_index)+1 );
  Fraction = np.zeros( (last_index-initial_index)+1 );
  for n in range( initial_index,last_index+1 ):
    filename='output'+"%08i"%n+'.xml'
    mcds=pyMCDS(filename,'output')
    geneY = mcds.data['discrete_cells']['genes_y']
    BiasMig = mcds.data['discrete_cells']['migration_bias'];
    cycle = mcds.data['discrete_cells']['cycle_model']
    biasDef = np.argwhere( (geneY > 0.5) & (BiasMig < 0.2) & (cycle < 100) ).flatten()
    biasRes = np.argwhere( (geneY > 0.5) & (BiasMig > 0.2) & (cycle < 100) ).flatten()
    NumberGreen[n] = len(biasDef)+len(biasRes)
    NumberGreenResp[n] = len(biasRes)
    if (NumberGreen[n] != 0.0 ): Fraction[n] = NumberGreenResp[n]/NumberGreen[n]
  print(Fraction[-1])
  X = np.linspace(initial_index,last_index,(last_index-initial_index)+1)
  Plot(X,Fraction,perct)
  return

def ReadPositionGreenCells2Points():
  initial_index = 72;
  last_index = 72;
  for n in range( initial_index,last_index+1 ):
    filename='output'+"%08i"%n+'.xml'
    mcds=pyMCDS(filename,'output1')
    geneY = mcds.data['discrete_cells']['genes_y']
    cx = mcds.data['discrete_cells']['position_x'];
    cy = mcds.data['discrete_cells']['position_y'];
    cycle = mcds.data['discrete_cells']['cycle_model']
    GreenCells = np.argwhere( (geneY > 0.5) & (cx >= 0) & (cy >= 0) & (cycle < 100)).flatten()
    Position = np.array([cx[GreenCells],cy[GreenCells]]).T
    print(Position.shape)
    Twopointfunc(Position)
  return

def ReadCells(index,folder="output"):
  filename='output'+"%08i"%index+'.xml'
  mcds=pyMCDS(filename,folder)
  geneY = mcds.data['discrete_cells']['genes_y']
  cx = mcds.data['discrete_cells']['position_x'];
  cy = mcds.data['discrete_cells']['position_y'];
  cycle = mcds.data['discrete_cells']['cycle_model']
  GreenCells = np.argwhere( (geneY > 0.5) & (cycle < 100)).flatten()
  NecCells = np.argwhere((cycle > 100)).flatten()
  Position = np.array([cx[GreenCells],cy[GreenCells]]).T
  return Position, len(NecCells)/len(cx)
  
def Twopointfunc(Position,NumBins=10):
  bins = np.linspace(0,500 , NumBins)
  #corr = two_point(Position, bins)
  corr, dcorr = bootstrap_two_point(Position, bins, Nbootstrap=5)
  #print(np.allclose(corr, 0, atol=0.02))
  #print(np.allclose(corr, 0, atol=2 * dcorr))
  print(corr)
  plt.plot(np.linspace(0,500 , NumBins-1),corr)
 
  plt.show()

def ImageOpenCV(file="output1/snapshot00000000.jpg",plot=True):
  img = cv2.imread(file)
  blur = cv2.blur(img,(5,5))
  img = img[100:-70, 0:-1]
  img0 = img.copy()
  # Red filter
  channelred = img[:,:,0]
  _,thresh_val1 = cv2.threshold(channelred,15 ,255,cv2.THRESH_BINARY)
  thresh_val1 = cv2.bitwise_not(thresh_val1)
  # Green filter
  channelgreen = img[:,:,1]
  _,thresh_val2 = cv2.threshold(channelgreen,70 ,255,cv2.THRESH_BINARY)
  thresh_val2 = cv2.bitwise_not(thresh_val2)
  # Blue filter
  channelblue = img[:,:,2]
  _,thresh_val3 = cv2.threshold(channelblue,35 ,255,cv2.THRESH_BINARY)
  thresh_val3 = cv2.bitwise_not(thresh_val3)  
  # Combine
  mask = cv2.bitwise_and(thresh_val1,thresh_val2)
  mask2 = cv2.bitwise_and(thresh_val1,~mask)
 
 # cv2.imshow("RedMask",thresh_val1)
  # cv2.imshow("GreenMask",mask2)
  # cv2.imshow("BlueMask",thresh_val3)
  # cv2.waitKey()
  # return

  element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 5))

  morph_img = thresh_val1.copy()
  cv2.morphologyEx(src=thresh_val1, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img)
  
  morph_img2 = mask2.copy()
  cv2.morphologyEx(src=mask2, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img2)
  
  morph_img3 = thresh_val3.copy()
  cv2.morphologyEx(src=thresh_val3, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img3)

  contours,_ = cv2.findContours(morph_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  contours2,_ = cv2.findContours(morph_img2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  contours3,_ = cv2.findContours(morph_img3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

  areas = [cv2.contourArea(c) for c in contours]
  sorted_areas = np.sort(areas)
  areas2 = [cv2.contourArea(c) for c in contours2]
  sorted_areas2 = np.sort(areas2)
  areas3 = [cv2.contourArea(c) for c in contours3]

  #Ellipses
  if (len(sorted_areas) > 0): 
    cntA=contours[areas.index(sorted_areas[-1])] #the biggest contour
    ellipseA = cv2.fitEllipse(cntA)
    cv2.ellipse(img,ellipseA,(0,0,0),2)    
  if (len(sorted_areas2) > 0): 
    cntA2=contours2[areas2.index(sorted_areas2[-1])] #the biggest contour
    epsilon = 0.001 * cv2.arcLength(cntA2, True)
    approx = cv2.approxPolyDP(cntA2, epsilon, True)
    cv2.drawContours(img, [approx], -1, (0, 255, 0), 4)
    ellipseA2 = cv2.fitEllipse(cntA2)
    cv2.ellipse(img,ellipseA2,(0,0,255),2) 
  
  #Necrotic cells
  necrotic = False
  if( (np.sum(areas3)/np.sum(areas)) > 0.05):
    necrotic = True
  
  #Scapping cells
  scape = False  
  morph_img4 = morph_img2.copy()
  (h,k),(ma,MA),angle = ellipseA
  ma = ma + 0.05*ma
  MA = MA + 0.05*MA 
  ellipseB = (h,k),(ma,MA),angle 
  cv2.ellipse(morph_img4,ellipseB,(0,0,0),-1)
  contours4,_ = cv2.findContours(morph_img4,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  areas4 = [cv2.contourArea(c) for c in contours4]
  if (np.sum(areas4) > 7.0):
    scape = True
    for ind in range(0,len(contours4)): 
      x = np.mean(contours4[ind][:,0,0])
      y = np.mean(contours4[ind][:,0,1])
      cv2.circle(img,(int(x),int(y)),4,(0,0,0),-1)

  
  #Plumes test
  plumes = False
  if (len(sorted_areas2) > 0): 
    (h,k),(ma,MA),angle = ellipseA2
    angle = angle*np.pi/180.0
    for i in range( 0,approx.shape[0]):
      (x,y) = approx[i,0,:]
      p = (((x - h)*np.cos(angle) + (y-k)*np.sin(angle))**2 / ((0.5*ma)**2)) +  (((x - h)*np.sin(angle) - (y-k)*np.cos(angle))**2 / ((0.5*MA)**2))
      if (p > 1.0): 
        dist = estimate_distance(x, y, 0.5*MA, 0.5*ma, x0=h, y0=k, angle=0, error=1e-5)
        tol =15.0
        if (dist >  tol):
          #print("Point: "+str(x)+" "+str(y)+" P: "+str(p)+" Dist: "+str(dist))
          cv2.circle(img,(int(x),int(y)),4,(0,200,0),-1)
          plumes = True
    # t = np.linspace(0, 2*np.pi, 100)
    # plt.plot( h+0.5*ma*np.cos(t) , k+0.5*MA*np.sin(t) )
    # plt.plot(approx[:,0,0],approx[:,0,1])
    # plt.show()
  # if (plumes): print("There is plumes!")
  # else: print("There isn't plumes!")
  
  #Area Calculation
  # (h,k),(ma,MA),angle = ellipseA 
  # Area = (0.5*ma)*(0.5*MA)*np.pi
  # Fraction = 250.0/85.0
  # Real_Area = (Fraction**2)*Area
  # # Compatibilization
  # (x,y),radius = cv2.minEnclosingCircle(cntA)
  # print(radius)
  # center = (int(x),int(y))
  # radius = int(radius)
  # cv2.circle(img,center,radius,(0,0,0),2)     
  # print("Real ellipse area: "+str(Real_Area)+" micrometers")  
  
  if (plot): 
    #cv2.imshow("morph_img",morph_img)
    #cv2.imshow("morph_img2",morph_img2)
    #cv2.imshow("morph_img3",morph_img3)
    cv2.imshow("img0", img0)
    cv2.imshow("img", img)
    cv2.waitKey()
  
  return plumes, scape, necrotic

def Response_rank(fileName="snapshot00000130.jpg",folder="output1"):
  file = folder+'/'+fileName
  print(file)
  Fraction = 85.2388916015625/250.0
  #Plumes
  plumes, ellipseA = ImageOpenCV(file,plot=False)
  (h,k),(ma,MA),angle = ellipseA
  
  img2 = cv2.imread(file)
  img2 = img2[100:-70, 0:-1]
  
  #Scape
  scape=False
  fileName1 = fileName.split("snapshot")
  fileName2 = fileName1[1].split(".jpg")
  index = int(fileName2[0])
  PositionGC,FracNecrotic = ReadCells(index,folder)
  necroticCore = False
  print(FracNecrotic)
  for i in range( 0,PositionGC.shape[0]):
    x = PositionGC[i,0]*Fraction + 1500.0*Fraction
    y = PositionGC[i,1]*Fraction + 1500.0*Fraction
    cv2.circle(img2,(int(x),int(y)),3,(0,0,0),-1)
    p = ((x*np.cos(angle) + y*np.sin(angle))**2 / ((0.5*ma)**2)) +  ((x*np.sin(angle) - y*np.cos(angle))**2 / ((0.5*MA)**2))
    if (p > 1.0): 
      dist = estimate_distance(x, y, 0.5*MA, 0.5*ma, x0=h, y0=k, angle=0, error=1e-5)
      tol =15.0
      if (dist >  tol):
        print("Point: "+str(x)+" "+str(y)+" P: "+str(p)+" Dist: "+str(dist))
        scape = True
  TolFracNec = 0.1
  if (TolFracNec < FracNecrotic): necroticCore = True
  cv2.ellipse(img2,ellipseA,(0,0,0),2) 
  cv2.imshow("img2", img2)
  cv2.waitKey()
  return plumes,scape,necroticCore
 
def Plot(X,Data,Percent):
  Perc = Percent*np.ones(Data.shape[0])
  plt.plot(X,Data)
  plt.plot(X,Perc)
  plt.show()
  
def PlotStochastic(mean,Fraction,color):
  theta = np.linspace(0, 2*np.pi, 100)
  r = mean
  # fig, ax = plt.subplots(1)
  # ax.plot(r*np.cos(theta), r*np.sin(theta),'--',color='gray',alpha=1.0,linewidth=0.7)
  # theta = 2.0*np.pi/Fraction.shape[0]
  # x_0 = 1 
  # y_0=0
  # x = x_0
  # y = y_0
  # for i in range( 0,Fraction.shape[0] ):
    # ax.arrow(0,0, Fraction[i]*(x), Fraction[i]*(y), head_width=0.05, head_length=0.1,color='Blue')
    # x = np.cos(theta)*x_0 -np.sin(theta)*y_0
    # y = np.sin(theta)*x_0 +np.cos(theta)*y_0
    # x_0 = x
    # y_0 = y
  # ax.set_aspect(1)
  # plt.xlim(-1.0*mean - 0.5*mean,1.0*mean + 0.5*mean)
  # plt.ylim(-1.0*mean - 0.5*mean,1.0*mean + 0.5*mean)
  
  # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center', ncol=2,)
  # plt.text(270, -305, '$\mu m$', fontsize=10) 
  # plt.text(-320, 270, '$\mu m$', fontsize=10)   
  # plt.xticks(np.arange(-250, 255, step=100.0))
  # plt.yticks(np.arange(-250, 255, step=100.0))
  #plt.show()
  

  fig, ax = plt.subplots()
  origin = np.zeros(2)
  patches = []
  # theta0 = 0 - (360.0/Fraction.shape[0])*0.5
  # thetaf = 0 + (360.0/Fraction.shape[0])*0.5
  theta0 = 0
  thetaf = (360.0/Fraction.shape[0])
  for i in range( 0,Fraction.shape[0] ):
    wedge = Wedge(origin,  Fraction[i], theta0, thetaf)
    patches.append(wedge)
    theta0 = thetaf
    thetaf = thetaf + (360.0/Fraction.shape[0])
  
  #colors = np.linspace(0,len(patches),len(patches))
  p = PatchCollection(patches, alpha=0.8)
  #p.set_array(np.array(colors))
  p.set_color(color)
  ax.add_collection(p)
  #Circle
  c = Circle(origin, r, fill=False, edgecolor='gray',ls=':')
  p1 = PatchCollection([c], match_original=True)
  ax.add_collection(p1)
  plt.xlim(-1.0*mean - 0.5*mean,1.0*mean + 0.5*mean)
  plt.ylim(-1.0*mean - 0.5*mean,1.0*mean + 0.5*mean)
  ax.set_aspect(1)

  plt.show()

def Stochastic(folder="outputStochastic1"):
  index = 100
  subdiv = 10
  numfiles = 10
  filename = 'output'+"%08i"%index+'.xml'
  RedFrac = np.zeros((subdiv, numfiles))
  GreenFrac = np.zeros((subdiv, numfiles))
  NecFrac = np.zeros((subdiv, numfiles))
  for n in range(numfiles):
    path = folder+"/%02i"%n
    mcds=pyMCDS(filename,path)
    geneY = mcds.data['discrete_cells']['genes_y']
    cx = mcds.data['discrete_cells']['position_x'];
    cy = mcds.data['discrete_cells']['position_y'];
    cycle = mcds.data['discrete_cells']['cycle_model']
    GreenCells = np.argwhere( (geneY > 0.5) & (cycle < 100)).flatten()
    RedCells = np.argwhere( (geneY < 0.5) & (cycle < 100)).flatten()
    NecCells = np.argwhere((cycle > 100)).flatten()
    # norm = np.sqrt(cx*cx + cy*cy)
    # cx = cx/norm
    # cy = cy/norm
    theta = np.arctan((cy/cx))
    theta[np.isnan(theta)] = 0.0
    quadrant1 = np.argwhere( (cx >= 0.0) & (cy >= 0.0)).flatten()
    quadrant2 = np.argwhere( (cx < 0.0) & (cy >= 0.0)).flatten()
    quadrant3 = np.argwhere( (cx <= 0.0) & (cy < 0.0)).flatten()
    quadrant4 = np.argwhere( (cx > 0.0) & (cy < 0.0)).flatten()
    theta[quadrant2] += np.pi
    theta[quadrant3] += np.pi
    theta[quadrant4] += 2*np.pi
    #firstDiv = np.argwhere( (theta >= (2*np.pi - (2*np.pi/subdiv)*0.5))).flatten() 
    #theta[firstDiv] -= 2*np.pi
    # theta0 = 0.0 - (2*np.pi/subdiv)*0.5
    # thetaf = 0.0 + (2*np.pi/subdiv)*0.5
    theta0 = 0.0
    thetaf = (2*np.pi/subdiv)
    for angle in range(subdiv):
      indR = np.argwhere( (theta >= theta0) & (theta < thetaf) & (geneY < 0.5) & (cycle < 100)).flatten()
      indG = np.argwhere( (theta >= theta0) & (theta < thetaf) & (geneY > 0.5) & (cycle < 100)).flatten()
      indN = np.argwhere( (theta >= theta0) & (theta < thetaf) & (cycle > 100)).flatten()
      Numcells = len(RedCells) + len(GreenCells) + len(NecCells)
      if (len(RedCells) > 0): RedFrac[angle,n] = len(indR)/Numcells
      if (len(GreenCells) > 0): GreenFrac[angle,n] = len(indG)/Numcells
      if (len(NecCells) > 0): NecFrac[angle,n] = len(indN)/Numcells
      # plt.figure()
      # plt.scatter(cx[indR],cy[indR],color='red')
      # plt.scatter(cx[indG],cy[indG],color='green')
      # plt.scatter(cx[indN],cy[indN],color='blue')
      # x = np.linspace(-1500.0,1500.0,100)
      # plt.plot(x,np.tan(theta0)*x)
      # plt.plot(x,np.tan(thetaf)*x)
      # plt.xlim(-1500.0,1500.0)
      # plt.ylim(-1500.0,1500.0)
      theta0 = thetaf
      thetaf = thetaf + (2*np.pi/subdiv)
  # plt.show()
  # PlotStochastic(np.mean(RedFrac),RedFrac[:,0],'#cc0000')
  # PlotStochastic(np.mean(GreenFrac),GreenFrac[:,0],'#007f00')
  # PlotStochastic(np.mean(NecFrac),NecFrac[:,3],'#6336de')
  return RedFrac,GreenFrac,NecFrac

def Cases():
  RedFrac1,GreenFrac1,NecFrac1 = Stochastic("outputStochastic1")
  RedFrac2,GreenFrac2,NecFrac2 = Stochastic("outputStochastic05")
  RedFrac3,GreenFrac3,NecFrac3 = Stochastic("outputStochastic01791")
  PlotStochasticTog(RedFrac1,GreenFrac1,NecFrac1)
  PlotStochasticTog(RedFrac2,GreenFrac2,NecFrac2)
  PlotStochasticTog(RedFrac3,GreenFrac3,NecFrac3)
  
def RankSim(Cut,par):
  Folders = ['0_01_000','0_01_050','0_01_130','0_1_000','0_1_050','0_1_130','0_05_000','0_05_050','0_05_130','1_01_000','1_01_050','1_01_130','1_1_000','1_1_050','1_1_130','1_05_000','1_05_050','1_05_130','05_01_000','05_01_050','05_01_130','05_1_000','05_1_050','05_1_130','05_05_000','05_05_050','05_05_130']
  Output = np.zeros((len(Folders),3), dtype=bool)
  Parameter = np.zeros((len(Folders),3))
  for n in range(len(Folders)):
    folder = 'outputRank/'+Folders[n]
    if (n < 9): Parameter[n,0] = 0.0
    if ( (n>=9) & (n<18) ): Parameter[n,0] = 1.0
    if (n>=18): Parameter[n,0] = 0.5
    if (n%9 < 3): Parameter[n,1] = 0.1
    if ( (n%9 >= 3) & (n%9 < 6) ): Parameter[n,1] = 1.0
    if (n%9 >= 6): Parameter[n,1] = 0.5
    if (n%3 == 0): Parameter[n,2] = 0.0
    if (n%3 == 1): Parameter[n,2] = 50.0
    if (n%3 == 2): Parameter[n,2] = 130.0
    Output[n,:] = Response_rank("snapshot00000100.jpg",folder)

  ind = np.argwhere( Parameter[:,par] == Cut)
  print(Output[ind,:])
  print(Parameter[ind,:])
  # X = Parameter[ind,par-2]
  # Y = Parameter[ind,par-1]
  # Z = Output[ind,par]
  # plt.xticks([0,0.5,1.0])
  # plt.yticks([0.1,0.5,1.0])
  # plt.scatter(X, Y, c=Z,marker='s',s=100)
  # plt.show()

  

def PlotStochasticTog(Fraction1,Fraction2,Fraction3):
  origin = np.zeros(2)
  patches1 = []
  patches2 = []
  patches3 = []
  opacity = 0.5
  
  #Red
  theta0 = 0
  thetaf = (360.0/Fraction3.shape[0])
  for i in range( 0,Fraction1.shape[0] ):
    wedge = Wedge(origin,np.mean(Fraction1[i,:]), theta0, thetaf)
    patches1.append(wedge)
    theta0 = thetaf
    thetaf = thetaf + (360.0/Fraction1.shape[0])
  p1 = PatchCollection(patches1, alpha=opacity)
  p1.set_color('#cc0000')
  
  # Green
  theta0 = 0
  thetaf = (360.0/Fraction3.shape[0])
  for i in range( 0,Fraction2.shape[0] ):
    wedge = Wedge(origin, np.mean(Fraction2[i,:]), theta0, thetaf)
    patches2.append(wedge)
    theta0 = thetaf
    thetaf = thetaf + (360.0/Fraction2.shape[0])
  p2 = PatchCollection(patches2, alpha=opacity)
  p2.set_color('#007f00')
  
  # Necrotic
  theta0 = 0
  thetaf = (360.0/Fraction3.shape[0])
  for i in range( 0,Fraction3.shape[0] ):
    wedge = Wedge(origin, np.mean(Fraction3[i,:]), theta0, thetaf)
    patches3.append(wedge)
    theta0 = thetaf
    thetaf = thetaf + (360.0/Fraction3.shape[0])
  p3 = PatchCollection(patches3, alpha=opacity)
  p3.set_color('#6336de')
  
  fig = plt.figure(figsize=(12,6))
  ax = fig.add_subplot(1,2,1)
  ax.set_title('Average fraction for each slice')
  tics = np.linspace(-0.10,0.10,num=5)
  ax.set_xticks(tics)
  ax.set_yticks(tics)
  ax.set_xlim(tics[0],tics[-1])
  ax.set_ylim(tics[0],tics[-1])
  ax.ticklabel_format(style='sci',axis='both')
  p = [p1,p2,p3]
  Average = np.array([np.mean(Fraction1),np.mean(Fraction2),np.mean(Fraction3)])
  ind = np.argsort(Average)
  ax.add_collection(p[ind[2]])
  ax.add_collection(p[ind[1]])
  ax.add_collection(p[ind[0]])
  # Label
  #LabelRadius = np.max(Average) + 0.01
  Fraction = [Fraction1,Fraction2,Fraction3]
  LabelRadius = np.mean(Fraction[ind[2]],axis=1) + 0.01
  LabelTheta = np.linspace((np.pi/Fraction3.shape[0]),2*np.pi-(np.pi/Fraction3.shape[0]),10)
  LabelPosition = np.zeros((Fraction3.shape[0],2))
  LabelPosition[:,0] = LabelRadius*np.cos(LabelTheta)
  LabelPosition[:,1] = LabelRadius*np.sin(LabelTheta)
  Label = []
  [Label.append("%i"%i) for i in range(1,Fraction3.shape[0] + 1)]
  for i in range(Fraction3.shape[0]):
    ax.text(LabelPosition[i,0] , LabelPosition[i,1], Label[i], fontsize=12, color='gray')
  ax.set_aspect(1)
  
  x = np.arange(1,Fraction1.shape[0]+1)
  ax2 = fig.add_subplot(3,2,2)
  ax2.errorbar(x,np.mean(Fraction1,axis=1),yerr=np.std(Fraction1,axis=1),fmt='o',color='#cc0000')#'#7a0000'
  # ax2.errorbar(x,np.mean(AFraction1,axis=1),yerr=np.std(Fraction1,axis=1),fmt='o',color='#a30000')
  # ax2.errorbar(x,np.mean(BFraction1,axis=1),yerr=np.std(Fraction1,axis=1),fmt='o',color='#cc0000')
  ax2.set_xticks(x)
  ax2.set_xlabel('Slices')
  ax2.set_ylabel('Fraction')
  ax2.ticklabel_format(style='sci',scilimits=(-2,4),axis='y',useMathText=True)
  
  ax3 = fig.add_subplot(3,2,4)
  ax3.errorbar(x,np.mean(Fraction2,axis=1),yerr=np.std(Fraction2,axis=1),fmt='o',color='#007f00')#'#001900'
  # ax3.errorbar(x,np.mean(AFraction2,axis=1),yerr=np.std(Fraction2,axis=1),fmt='o',color='#004c00')
  # ax3.errorbar(x,np.mean(BFraction2,axis=1),yerr=np.std(Fraction2,axis=1),fmt='o',color='#007f00')
  ax3.set_xticks(x)
  ax3.set_xlabel('Slices')
  ax3.set_ylabel('Fraction')
  ax3.ticklabel_format(style='sci',scilimits=(-2,4),axis='y',useMathText=True)
  
  ax4 = fig.add_subplot(3,2,6)
  ax4.errorbar(x,np.mean(Fraction3,axis=1),yerr=np.std(Fraction3,axis=1),fmt='o',color='#6336de')
  # ax4.errorbar(x,np.mean(AFraction3,axis=1),yerr=np.std(Fraction3,axis=1),fmt='o',color='#6336de')
  # ax4.errorbar(x,np.mean(BFraction3,axis=1),yerr=np.std(Fraction3,axis=1),fmt='o',color='#6336de')
  ax4.set_xticks(x)
  ax4.set_xlabel('Slices')
  ax4.set_ylabel('Fraction')
  ax4.ticklabel_format(style='sci',scilimits=(-2,4),axis='y',useMathText=True)
  
  plt.subplots_adjust(wspace=0.4,hspace=0.5)
  plt.show()
  
from math import sin, cos, atan2, pi, fabs
def ellipe_tan_dot(rx, ry, px, py, theta):
    '''Dot product of the equation of the line formed by the point
    with another point on the ellipse's boundary and the tangent of the ellipse
    at that point on the boundary.
    '''
    return ((rx ** 2 - ry ** 2) * cos(theta) * sin(theta) -
            px * rx * sin(theta) + py * ry * cos(theta))


def ellipe_tan_dot_derivative(rx, ry, px, py, theta):
    '''The derivative of ellipe_tan_dot.
    '''
    return ((rx ** 2 - ry ** 2) * (cos(theta) ** 2 - sin(theta) ** 2) -
            px * rx * cos(theta) - py * ry * sin(theta))


def estimate_distance(x, y, rx, ry, x0=0, y0=0, angle=0, error=1e-5):
    '''Given a point (x, y), and an ellipse with major - minor axis (rx, ry),
    its center at (x0, y0), and with a counter clockwise rotation of
    `angle` degrees, will return the distance between the ellipse and the
    closest point on the ellipses boundary.
    '''
    x -= x0
    y -= y0
    if angle:
        # rotate the points onto an ellipse whose rx, and ry lay on the x, y
        # axis
        angle = -pi / 180. * angle
        x, y = x * cos(angle) - y * sin(angle), x * sin(angle) + y * cos(angle)

    theta = atan2(rx * y, ry * x)
    while fabs(ellipe_tan_dot(rx, ry, x, y, theta)) > error:
        theta -= ellipe_tan_dot(
            rx, ry, x, y, theta) / \
            ellipe_tan_dot_derivative(rx, ry, x, y, theta)

    px, py = rx * cos(theta), ry * sin(theta)
    return ((x - px) ** 2 + (y - py) ** 2) ** .5
