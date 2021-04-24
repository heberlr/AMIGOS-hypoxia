from pyMCDS import pyMCDS
import numpy as np
import matplotlib.pyplot as plt

initial_index = 0;
last_index = 50;
Lcell_size = 5;
Dcell_size = 1;
Ecell_size = 5;
live_count = np.zeros( last_index+1 );
dead_count = np.zeros( last_index+1 );
endo_count = np.zeros( last_index+1 );
times = np.zeros( last_index+1 );

Col1, Col2, Col3 = [], [], []
for line in open('DataProl.dat', 'r'):
  values = [float(s) for s in line.split()]
  Col1.append(values[0])
  Col2.append(values[1])
  Col3.append(values[2])

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

  for i in range( 0,len(cx) ):
    distance = np.linalg.norm([cx[i],cy[i]])
    index = int(distance/SizeSubdiv)
    #print('Pos: '+str(cx[i])+'; '+str(cy[i])+ ' -- Dist: '+str(distance[i])+ ' -- INDEX: '+str(index))
    if (current_phase[i] == 2):
        binPhaseKi67Pos[index] = binPhaseKi67Pos[index] + 1
    if (current_phase[i] == 3):
        binPhaseKi67Neg[index] = binPhaseKi67Neg[index] + 1
    

  # proteinsX = mcds.data['discrete_cells']['proteins_x']
  # proteinsY = mcds.data['discrete_cells']['proteins_y']
  # rgb_var = np.vstack((0.5*proteinsX[live],0.5*proteinsY[live],0*proteinsX[live])).T

  plt.clf()
  figure, axes = plt.subplots(nrows=1, ncols=3,figsize=(12,3))
  plt.subplot(131)
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
  plt.subplot(132)
  plt.ylabel('Percentage of live cells')
  plt.xlabel('Distance from the tumor core (mm)')
  plt.ylim(-0.01, 1.01)
  plt.xlim(0, RadiusSize/1000.0)
  plot1 = (1.0*binPhaseKi67Pos/(binPhaseKi67Pos+binPhaseKi67Neg))
  plot2 = (1.0*binPhaseKi67Neg/(binPhaseKi67Pos+binPhaseKi67Neg))
  plot1[np.isnan(plot1)] = 0
  plot2[np.isnan(plot2)] = 0
  #print( np.linspace(SizeSubdiv/2000.0, (RadiusSize/1000.0) - (SizeSubdiv/2000.0), int(RadiusSize/SizeSubdiv)))
  plt.errorbar(np.array(Col1)/4, Col2, yerr=Col3)
  plt.plot( np.linspace(SizeSubdiv/2000.0, (RadiusSize/1000.0) - (SizeSubdiv/2000.0), int(RadiusSize/SizeSubdiv)),plot1,c='black',label='Ki67+');
  plt.plot( np.linspace(SizeSubdiv/2000.0, (RadiusSize/1000.0) - (SizeSubdiv/2000.0), int(RadiusSize/SizeSubdiv)),plot2,c='r',label='Ki67-');
  plt.legend()
  
  plt.subplot(133)
  Oxygen = np.zeros((int(RadiusSize/SizeSubdiv),))
  for i in range( 0,len(Oxygen) ):
    indice = i*(SizeSubdiv) + (SizeSubdiv/2.0)
    Oxygen[i] = mcds.get_concentrations_at(x=0., y=indice, z=0.)
  plt.ylabel('Percentage of live cells')
  plt.xlabel('Pressure of oxygen (mmHg)')
  plt.ylim(-0.01, 1.01)
  plt.xlim(0, 50.0)
  #print(Oxygen)
  plt.plot( Oxygen,plot1,c='black')
  figure.tight_layout(pad=1.0)
  figure.savefig(filenameOut)
  plt.close()


#rgb_var = np.vstack((0.5*proteinsX[live],0.5*proteinsY[live],0*proteinsX[live])).T
#print(len(cx))
#for i in range(len(cx)):
#  if cell_type[i] != 0:
#    print(cell_type[i]);

# plt.figure(1) 
# plt.clf()
# plt.scatter(cx[live],cy[live],c=rgb_var, s=Lcell_size);
# plt.scatter( cx[dead],cy[dead],c='moccasin',s=Dcell_size );
# plt.title( 'Cells colored by oncoprotein value' , size=10)
# plt.axis( 'image' )
# plt.xlim(-1500, 1500)
# plt.ylim(-1500, 1500)
# plt.xlabel( 'x' , size=10 )
# plt.ylabel( 'y', size=10 )


#plt.clf()
#plt.figure(2)
#plt.xlim(-1500, 1500)
#plt.ylim(-1500, 1500) 
#o2 = mcds.get_concentrations( 'oxygen' );
#vegf = mcds.get_concentrations( 'VEGF' );
#ecm = mcds.get_concentrations( 'ECM' );
#X,Y = mcds.get_2D_mesh();
#plt.contourf(X,Y,o2[:,:,0],cmap='binary');
#plt.colorbar()
#plt.axis( 'image' )

# plt.figure(3)
# plt.contourf(X,Y,vegf[:,:,0],cmap='binary');
# plt.colorbar()
# plt.axis( 'image' )

# plt.figure(4)
# plt.contourf(X,Y,ecm[:,:,0],cmap='binary');
# plt.colorbar()
# plt.axis( 'image' )


#plt.clf()
#plt.figure(5) 
#mcds.get_substrate_names();
#o2 = mcds.get_concentrations( 'oxygen' );
#X,Y = mcds.get_2D_mesh();
#plt.contourf(X,Y,o2[:,:,0],cmap='binary');
#plt.colorbar()
#plt.scatter( cx,cy,c='r',s=Lcell_size);
#plt.axis('image')
#plt.xlim(-1500, 1500)
#plt.ylim(-1500, 1500)
#plt.title( 'Live cells colored at t=' +str(t/60) + ' hr', size=10)
# let's plot dead cells as black
#plt.scatter( cx[dead],cy[dead],c='moccasin',s=Dcell_size );
#plt.scatter( cx[endo],cy[endo],c='r',s=Dcell_size );
#plt.show()
