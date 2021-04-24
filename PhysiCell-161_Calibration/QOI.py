from pyMCDS import pyMCDS
import numpy as np
import matplotlib.pyplot as plt

initial_index = 0;
last_index = 336;

Reds_Vol = np.zeros( last_index+1 );
Green_Vol = np.zeros( last_index+1 );
Doub_Vol = np.zeros( last_index+1 );
PorcentReds = np.zeros( last_index+1 );
PorcentGreen = np.zeros( last_index+1 );
PorcentDoub = np.zeros( last_index+1 );
times = np.zeros( last_index+1 );

ConcentDefR = 1.0
ConcentDefG = 0.95
FrameVol = 3000*3000*3000
Delta = 0.4

for n in range( initial_index,last_index+1 ):
  filename='output'+"%08i"%n+'.xml'
  filenameOut='Figs/output'+"%08i"%n+'.png'
  mcds=pyMCDS(filename,'output720')
  times[n]= mcds.get_time()
  
  cx = mcds.data['discrete_cells']['position_x'];
  cy = mcds.data['discrete_cells']['position_y'];
  cycle = mcds.data['discrete_cells']['cycle_model']
  cycle = cycle.astype( int )
  cell_type = mcds.data['discrete_cells']['cell_type']
  cell_type = cell_type.astype(int)
  proteinsX = mcds.data['discrete_cells']['proteins_x']
  proteinsY = mcds.data['discrete_cells']['proteins_y']
  volume = mcds.data['discrete_cells']['total_volume']
  
  Reds = np.argwhere( (cycle < 100) & (cell_type==0) & (proteinsX-proteinsY >= Delta)).flatten()
  Green = np.argwhere( (cycle < 100) & (cell_type==0) & (proteinsY-proteinsX >= Delta)).flatten()
  Doub = np.argwhere( (cycle < 100) & (cell_type==0) & (abs(proteinsX-proteinsY) < Delta)).flatten()
  
  Reds_Vol[n] = volume[Reds].sum()
  Green_Vol[n] = volume[Green].sum()
  Doub_Vol[n] = volume[Doub].sum()
  
  PorcentReds[n] = Reds_Vol[n]/FrameVol
  PorcentGreen[n] = Green_Vol[n]/FrameVol
  PorcentDoub[n] = Doub_Vol[n]/FrameVol

  #print("#Reds " + str(len(volume[Reds])) + " Red Volume " + str(Reds_Vol[n])+"  #Green " + str(len(volume[Green])) + " Green Volume " + str(Green_Vol[n])+"  #Doub " + str(len(volume[Doub])) + " Doub Volume " + str(Doub_Vol[n])+ " \n")
  
  # rgb_var = np.vstack((0.5*proteinsX[live],0.5*proteinsY[live],0*proteinsX[live])).T
	
  # plt.clf()
  # fig = plt.figure()
  # o2 = mcds.get_concentrations( 'VEGF' );
  # X,Y = mcds.get_2D_mesh();
  # plt.contourf(X,Y,o2[:,:,0],cmap='binary');
  # plt.colorbar()
  # plt.scatter( cx[live],cy[live],c=rgb_var,s=Lcell_size);
  # plt.axis('image')
  # plt.xlim(-1500, 1500)
  # plt.ylim(-1500, 1500)
  # plt.title( '#LC:'+str("%04i"%(live_count[n]))+'  #DC:'+str("%04i"%(dead_count[n]))+'  #EC:'+str("%04i"%endo_count[n])+ '  Time:' +str("%8.2f"%(n/24)) + ' days', size=10)
  # plt.scatter( cx[dead],cy[dead],c='moccasin',s=Dcell_size );
  # plt.scatter( cx[endo],cy[endo],c='r',s=Ecell_size );
  # fig.savefig(filenameOut)
plt.figure(1)
# plt.subplot(121)
plt.xlabel('Time (min)')
plt.ylabel('Volume')
plt.plot(times,Reds_Vol,color='red')
plt.plot(times,Green_Vol,color='green');
plt.plot(times,Doub_Vol,color='orange');
# plt.figure(2)
# plt.subplot(122)
# plt.xlabel('Time (min)')
# plt.ylabel('%Volume')
# plt.plot(times,PorcentReds,color='red')
# plt.plot(times,PorcentGreen,color='green');
# plt.plot(times,PorcentDoub,color='orange');
plt.show()