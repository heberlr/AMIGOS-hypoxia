import numpy as np
import random as rand

def Generate2DInitialCond():
    tumor_radius = 2000
    cell_radius = 8.413
    volume_experiment = 15.0*347.0*347.0
    numCell_experim = 674
    density = numCell_experim/volume_experiment
    confluence = density*(4.0/3.0)*np.pi*(cell_radius**3)
    density2D = confluence/(np.pi*(cell_radius**2))
    area_total = np.pi*tumor_radius*tumor_radius
    numCell_sample = round(density2D*area_total)
    numCell_temp = 0

    rad = tumor_radius*rand.random()
    theta = 2*np.pi*rand.random()
    position = np.array([[rad*np.cos(theta),rad*np.sin(theta),0.0]])
    
    while numCell_temp < numCell_sample:
      include = True
      rad = tumor_radius*rand.random()
      theta = 2*np.pi*rand.random()
      sample = np.array([rad*np.cos(theta),rad*np.sin(theta),0.0])
      for i in range(len(position)):
        d = np.linalg.norm(position[i,:] - sample, axis=0)
        if (d < 0.95*2.0*cell_radius):
          include = False
          break
      if (include == True):
        position = np.append(position,np.reshape(sample, (1, 3)),axis=0)
        numCell_temp += 1
      
    with open('Posfile.txt','w') as f:
      np.savetxt(f, position, fmt='%e')


def Generate2DInitialCondOrg():
    tumor_radius = 2000
    cell_radius = 8.413
    cell_spacing = 0.95 * 2.0 * cell_radius
    x = 0.0
    x_outer = tumor_radius
    y = 0.0
    n = 0
    
    position = []
    while( y < tumor_radius ):
        x = 0.0 
        if( n % 2 == 1 ):
          x = 0.5*cell_spacing
        x_outer = np.sqrt( tumor_radius*tumor_radius - y*y ) 
        while( x < x_outer ):
          position.append([x , y , 0.0 ])
          if( np.abs( y ) > 0.01 ):
            position.append([ x , -y , 0.0 ])
          if( np.abs( x ) > 0.01 ):
            position.append([ -x , y , 0.0 ])
            if( np.abs( y ) > 0.01 ):
              position.append([ -x , -y , 0.0 ])
          x += cell_spacing
        y += cell_spacing * np.sqrt(3.0)/2.0
        n+=1
    position = np.array(position)
    
    volume_experiment = 15.0*347.0*347.0
    numCell_experim = 674
    density = numCell_experim/volume_experiment
    confluence = density*(4.0/3.0)*np.pi*(cell_radius**3)
    density2D = confluence/(np.pi*(cell_radius**2))
    area_total = np.pi*tumor_radius*tumor_radius
    numCell_sample = round(density2D*area_total)
    
    while( position.shape[0] > numCell_sample ):
      index = rand.randint(0, position.shape[0]-1)
      position = np.delete(position, index, 0)
    
    with open('Posfile.txt','w') as f:
      np.savetxt(f, position, fmt='%e')

            

def Generate3DInitialCond():
    tumor_radius = 400
    cell_radius = 8.413
    volume_total = (4.0/3.0)*np.pi*tumor_radius*tumor_radius*tumor_radius
    volume_experiment = 15.0*347.0*347.0
    numCell_experim = 674
    density = numCell_experim/volume_experiment
    numCell_sample = round(volume_total*density)
    numCell_temp = 0

    rad = tumor_radius*rand.random()
    theta = np.pi*rand.random()
    phi = 2*np.pi*rand.random()
    position = np.array([[rad*np.sin(theta)*np.cos(phi),rad*np.sin(theta)*np.sin(phi),rad*np.cos(theta)]])

    print(numCell_sample)

    while numCell_temp < numCell_sample:
      include = True
      rad = tumor_radius*rand.random()
      theta = np.pi*rand.random()
      phi = 2*np.pi*rand.random()
      sample = np.array([rad*np.sin(theta)*np.cos(phi),rad*np.sin(theta)*np.sin(phi),rad*np.cos(theta)])
      for i in range(len(position)):
        d = np.linalg.norm(position[i,:] - sample, axis=0)
        if (d < 0.95*2*cell_radius):
          include = False
          break
      if (include == True):
        position = np.append(position,np.reshape(sample, (1, 3)),axis=0)
        numCell_temp += 1
      
    with open('Posfile3D.txt','w') as f:
      np.savetxt(f, position, fmt='%e')

Generate2DInitialCondOrg()