from pyMCDS import pyMCDS
import numpy as np
import matplotlib.pyplot as plt

initial_index = 1;
last_index = 1;
Lcell_size = 5;
Dcell_size = 1;
Ecell_size = 5;
live_count = np.zeros( last_index+1 );
dead_count = np.zeros( last_index+1 );
endo_count = np.zeros( last_index+1 );
times = np.zeros( last_index+1 );

for n in range( initial_index,last_index+1 ):
  filename='output'+"%08i"%n+'.xml'
  filenameOut='Figs/output'+"%08i"%n+'.png'
  mcds=pyMCDS(filename,'output')
  times[n]= mcds.get_time()
  
  cx = mcds.data['discrete_cells']['position_x'];
  cy = mcds.data['discrete_cells']['position_y'];
  cycle = mcds.data['discrete_cells']['cycle_model']
  cycle = cycle.astype( int )
  current_phase = mcds.data['discrete_cells']['current_phase']
  current_phase = current_phase.astype(int)
  elapsed_time_in_phase = mcds.data['discrete_cells']['elapsed_time_in_phase']
  cell_type = mcds.data['discrete_cells']['cell_type']
  cell_type = cell_type.astype(int)
  live = np.argwhere( (cycle < 100) & (cell_type==0) ).flatten()
  dead = np.argwhere( (cycle >= 100) & (cell_type==0) ).flatten()
  endo = np.argwhere( cell_type==1 ).flatten()
  live_count[n] = len(live)
  dead_count[n] = len(dead)
  endo_count[n] = len(endo)
  
  SizeSubdiv = 50.0
  RadiusSize = 3000.0
  binPhaseKi67Pos = np.zeros((int(RadiusSize/SizeSubdiv),), dtype=int)
  binPhaseKi67Neg = np.zeros((int(RadiusSize/SizeSubdiv),), dtype=int)
    
  #proteinsX = mcds.data['discrete_cells']['proteins_x']
  #proteinsY = mcds.data['discrete_cells']['proteins_y']
  #rgb_var = np.vstack((0.5*proteinsX[live],0.5*proteinsY[live],0*proteinsX[live])).T

  plt.clf()
  o2 = mcds.get_concentrations( 'oxygen' );
  X,Y = mcds.get_2D_mesh();
  #plt.clim(0,o2.max())
  v = np.linspace(0, o2.max(), 10, endpoint=True)
  plt.contourf(X,Y,o2[:,:,0],v,cmap='binary');
  x = plt.colorbar(ticks=v)
  #plt.scatter( cx[live],cy[live],c=rgb_var,s=Lcell_size);
  plt.axis('image')
  plt.xlim(-RadiusSize, RadiusSize)
  plt.ylim(-RadiusSize, RadiusSize)
  plt.title( '#LC:'+str("%04i"%(live_count[n]))+'  #DC:'+str("%04i"%(dead_count[n]))+ '  Time:' +str("%8.2f"%(n/24)) + ' days', size=10)
  #plt.scatter( cx[dead],cy[dead],c='moccasin',s=Dcell_size );
  #plt.scatter( cx[endo],cy[endo],c='r',s=Ecell_size );
  #figure.savefig(filenameOut)
  
  plt.figure(2)
  Oxygen = np.zeros((int(RadiusSize/SizeSubdiv)))
  index = np.zeros((int(RadiusSize/SizeSubdiv)))
  for i in range( 0,len(Oxygen) ):
    index[i] = i*(SizeSubdiv) + (SizeSubdiv/2.0)
    Oxygen[i] = mcds.get_concentrations_at(x=0., y=index[i], z=0.)
  plt.ylabel('Percentage of live cells')
  plt.xlabel('Pressure of oxygen (mmHg)')
  print(Oxygen)
  print(index)
  plt.plot(index,Oxygen,c='black')

  plt.show()
  plt.close()

