#! /usr/bin/env python3
#
def rk4 ( dydt, tspan, y0, n, Par ):

#*****************************************************************************80
#
## RK4 approximates the solution to an ODE using the RK4 method.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    22 April 2020
#
#  Author:
#
#    John Burkardt
#
#  Input:
#
#    function dydt: points to a function that evaluates the right
#    hand side of the ODE.
#
#    real tspan[2]: contains the initial and final times.
#
#    real y0[m]: an array containing the initial condition.
#
#    integer n: the number of steps to take.
#
#  Output:
#
#    real t[n+1], y[n+1,m]: the times and solution values.
#
  import numpy as np

  if ( np.ndim ( y0 ) == 0 ):
    m = 1
  else:
    m = len ( y0 )

  tfirst = tspan[0]
  tlast = tspan[1]
  dt = ( tlast - tfirst ) / n
  t = np.zeros ( n + 1 )
  y = np.zeros ( [ n + 1, m ] )
  t[0] = tspan[0]
  y[0,:] = y0

  for i in range ( 0, n ):

    f1 = dydt ( t[i],            y[i,:], Par )
    f2 = dydt ( t[i] + dt / 2.0, y[i,:] + dt * f1 / 2.0, Par )
    f3 = dydt ( t[i] + dt / 2.0, y[i,:] + dt * f2 / 2.0, Par )
    f4 = dydt ( t[i] + dt,       y[i,:] + dt * f3, Par )

    t[i+1] = t[i] + dt
    y[i+1,:] = y[i,:] + dt * ( f1 + 2.0 * f2 + 2.0 * f3 + f4 ) / 6.0

  return t, y
