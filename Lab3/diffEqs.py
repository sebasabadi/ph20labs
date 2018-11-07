# Ph20, Lab 3
#
# This program models and plots a spring's velocity and position using the
# explicit Euler's method, plots the error that comes from using that method,
# plots how those errors change as a function of step size, and plots the
# energy implied by those positions and velocities as a function of time. This
# program then models the spring with the implicit Euler's method, and plots
# the errors that come from using that method as well as the energy of the
# system that is implied by the values for position and velocity it gives.
# This program then plots the phase-space geometries for the springs as
# modelled by the explicit and implicit Euler's method, as well as by the
# symplectic Euler's method, and compares those to a plot of the analytic
# solution. Finally, this program plots the energy of the system as modelled
# using the sympletic Euler's method.
#
# Author: Sebastien Abadi
#
# Date: 10/30/18
#

import sys
import numpy as np
import pylab as pl
import math
import matplotlib.pyplot as plt
import scipy.integrate


# Makes a plot of one array against another with a given title and
# given axis labels
def basicPlot(title, xList, yList, xLabel, yLabel):
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(9.5,6.5))
    plt.plot(xList, yList)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show(block=True)

# Analytically finds the position of the spring with given initial
# conditions at a given time. Assumes that k/m=1.
def xAtT(xi, vi, t):

    # spring energy is kinetic + potential; note that any nonzero value
    # for m (and thus also for k) cancels out with itself when
    # calculating the maximum position, so it is set to 1 for simplicity
    E = 0.5 * (vi)**2 + .5 * (xi)**2
    xMax = math.sqrt(2 * E) # maximum value for position
    
    # Calculates and returns the position at a given time.
    if vi == 0:
        return xMax * math.cos(t - \
                        math.acos(xi / xMax))
    else:
        return xMax * math.cos(np.sign(vi) * t - \
                               math.acos(xi / xMax))

# Analytically finds the velocity of the spring with given initial
# conditions at a given time. Assumes that k/m=1.
def vAtT(xi, vi, t):
    
    # spring energy is kinetic + potential; note that any nonzero value
    # for m (and thus also for k) cancels out with itself when
    # calculating the maximum velocity, so it is set to 1 for simplicity
    E = 0.5 * (vi)**2 + .5 * (xi)**2
    vMax = math.sqrt(2 * E) # maximum value for velocity
    
    # Calculates and returns the position at a given time.
    if xi == 0:
        return vMax * math.cos(t - \
                               math.acos(vi / vMax))
    else:
        return vMax * math.cos(np.sign(-xi) * t - \
                               math.acos(vi / vMax))


# Plots the position and velocity as a function of time of a mass on a
# spring as calculated using explicit Euler's method, with step size
# 'h', initial position 'xi', and initial velocity 'vi'. For simplicity,
# it is assumed that k/m=1 for the spring being modelled. Then plots the
# error in the values found using explicit Euler's method. Then plots
# the normalized total energy (x^2+v^2) of the system as a function of
# time.
def explicitEulerSpring(h, xi, vi):
    
    # approximately 10 cycles of oscillation will be plotted
    numCycles = 10
    
    # Calculates how many steps of size h are necessary to complete
    # the number of cycles -- rounds up the number of steps if that number
    # is not an integer
    numSteps = int(math.ceil(numCycles * 2 * np.pi / h))
    
    # numpy arrays that store the values of T, the approximated
    # positons and velocities at those times, and the normalized total
    # energy of the system at those times
    tOut = np.zeros(numSteps)
    xOut = np.zeros(numSteps)
    vOut = np.zeros(numSteps)
    eOut = np.zeros(numSteps)
    
    # numpy arrays that store the analytically assessed values
    xAnal = np.zeros(numSteps)
    vAnal = np.zeros(numSteps)
    
    # sets values at t=0
    xOut[0] = xi
    vOut[0] = vi
    eOut[0] = xi**2 + vi**2
    xAnal[0] = xi
    vAnal[0] = vi

    # calculates subsequent positions, velociites, energies, and times
    for i in range(numSteps - 1):
        xOut[i + 1] = xOut[i] + h * vOut[i]
        vOut[i + 1] = vOut[i] - h * xOut[i]
        tOut[i + 1] = tOut[i] + h
        eOut[i + 1] = xOut[i + 1]**2 + vOut[i + 1]**2
        xAnal[i + 1] = xAtT(xi, vi, tOut[i + 1])
        vAnal[i + 1] = vAtT(xi, vi, tOut[i + 1])

    # Plots the position and velocity from explicit Euler's method
    # against time.
    basicPlot("Position as a Function of Time with Step Size = " + str(h), tOut,  \
               xOut, "Time", "Position")
    basicPlot("Velocity as a Function of Time with Step Size = " + str(h), tOut,  \
               vOut, "Time", "Velocity")

    # Calculates and plots error from using explicit Euler's method as
    # compared to analytical solutions.
    xError = xOut - xAnal
    vError = vOut - vAnal
    basicPlot("Error in Position as a Function of Time with\nStep Size = " + str(h), \
              tOut, xError, "Time", "Error in Positon")
    basicPlot("Error in Velocity as a Function of Time with\nStep Size = " + str(h), \
              tOut, vError, "Time", "Error in Velocity")

    # Plots the normalized total energy as a function of time.
    basicPlot("Normalized Total Energy as a Function of Time\nwith Step Size = " + \
              str(h), tOut, eOut, "Time", "Energy")


# Plots the truncation error from using explicit Euler's method as a
# function of step size at a given time 't' with a given number of
# data points 'numPoints'.
def truncErr(t, numPoints):
    
    # numpy arrays to store values for error and step size
    xError = np.zeros(numPoints)
    stepSize = np.zeros(numPoints)
    
    # finds error at several step sizes
    for j in range(numPoints):
        
        # step size
        h = 0.05 / (j+1)
        
        # Calculates how many steps of size h are necessary to reach the
        # given time 't'.
        numSteps = int(math.ceil(t / h))
    
        # numpy arrays that store the positions of T, and of the
        # approximated positons, and velocities
        tOut = np.zeros(numSteps)
        xOut = np.zeros(numSteps)
        vOut = np.zeros(numSteps)
        
        # numpy arrays that store the analytically assessed position
        x = np.zeros(numSteps)
        
        # sets values at t=0 (1 and 0 are chosen arbitrarily)
        xOut[0] = 1
        vOut[0] = 0
        
        # calculates subsequent positions, velociites, and times
        for i in range(numSteps - 1):
            xOut[i + 1] = xOut[i] + h * vOut[i]
            vOut[i + 1] = vOut[i] - h * xOut[i]
            tOut[i + 1] = tOut[i] + h

        # records error and step size
        xError[j] = xOut[numSteps - 1] - xAtT(1, 0, tOut[numSteps - 1])
        stepSize[j] = h

    # plots error as a function of step size
    basicPlot("Error in Approximated Position as a Function of Step Size"\
             + " at t= " + str(t), stepSize, xError, "Step Size",\
             "Error in Approximated Positon")


# Plots the position and velocity as a function of time of a mass on a
# spring as calculated using the implicit Euler's method, with step size
# 'h', initial position 'xi', and initial velocity 'vi'. For simplicity,
# it is assumed that k/m=1 for the spring being modelled. Then plots the
# error in the values found using implicit Euler's method. Then plots
# the normalized total energy (x^2+v^2) of the system as a function of
# time.
def implicitEulerSpring(h, xi, vi):
    
    # approximately 10 cycles of oscillation will be plotted
    numCycles = 10
    
    # Calculates how many steps of size h are necessary to complete
    # the number of cycles -- rounds up the number of steps if that number
    # is not an integer
    numSteps = int(math.ceil(numCycles * 2 * np.pi / h))
    
    # numpy arrays that store the values of T, the approximated
    # positons and velocities at those times, and the normalized total
    # energy of the system at those times
    tOut = np.zeros(numSteps)
    xOut = np.zeros(numSteps)
    vOut = np.zeros(numSteps)
    eOut = np.zeros(numSteps)
    
    # numpy arrays that store the analytically assessed values
    xAnal = np.zeros(numSteps)
    vAnal = np.zeros(numSteps)
    
    # sets values at t=0
    xOut[0] = xi
    vOut[0] = vi
    eOut[0] = xi**2 + vi**2
    xAnal[0] = xi
    vAnal[0] = vi
    
    # calculates subsequent positions, velocites, energies, and times
    for i in range(numSteps - 1):
        xOut[i + 1] = (xOut[i] + h * vOut[i]) / (1 + h**2)
        vOut[i + 1] = (vOut[i] - h * xOut[i]) / (1 + h**2)
        tOut[i + 1] = tOut[i] + h
        eOut[i + 1] = xOut[i + 1]**2 + vOut[i + 1]**2
        xAnal[i + 1] = xAtT(xi, vi, tOut[i + 1])
        vAnal[i + 1] = vAtT(xi, vi, tOut[i + 1])

    # Plots the position and velocity from implicit Euler's method
    # against time.
    basicPlot("Position as a Function of Time with\n" +\
              "Step Size = " + str(h), tOut,  \
              xOut, "Time", "Position")
    basicPlot("Velocity as a Function of Time with\n" +\
              "Step Size = " + str(h), tOut,  \
              vOut, "Time", "Velocity")

    # Calculates and plots error from using implicit Euler's method as
    # compared to analytical solutions.
    xError = xOut - xAnal
    vError = vOut - vAnal
    basicPlot("Error in Position as a Function of Time with\n" +\
              "Step Size = " + str(h), tOut, xError, "Time", \
              "Error in Positon")
    basicPlot("Error in Velocity as a Function of Time with\n" +\
              "Step Size = " + str(h), tOut, vError, "Time", \
              "Error in Positon")
              
    # Plots the normalized total energy as a function of time.
    basicPlot("Normalized Total Energy as a Function of Time with\n"\
              + "Step Size = " + str(h), tOut, eOut, "Time", "Energy")



# Plots the phase-space geometry of the trajectories produced by the
# explicit and implicit Euler methods when used to model a simple harmonic
# oscillator with a given step-size 'h', given starting position 'xi', and
# given starting velocity 'vi', and for a given number of cycles
# 'numCycles'. Compares those plots to the plot that uses the analytical
# solutions.
def phaseSpaceGeoExpImpAna(xi, vi, h, numCycles):

    # Calculates how many steps of size h are necessary to complete
    # the number of cycles -- rounds up the number of steps if that number
    # is not an integer
    numSteps = int(math.ceil(numCycles * 2 * np.pi / h))

    # numpy arrays that store the values of the positon and velocity
    # as found with the explicit and implicit Euler methods, as well as
    # analystically.
    xExp = np.zeros(numSteps)
    vExp = np.zeros(numSteps)
    xImp = np.zeros(numSteps)
    vImp = np.zeros(numSteps)
    xAna = np.zeros(numSteps)
    vAna = np.zeros(numSteps)
    tOut = np.zeros(numSteps)


    # sets values at t=0
    xExp[0] = xi
    vExp[0] = vi
    xImp[0] = xi
    vImp[0] = vi
    xAna[0] = xi
    vAna[0] = vi

    # calculates subsequent positions and velociites
    for i in range(numSteps - 1):
        xExp[i + 1] = xExp[i] + h * vExp[i]
        vExp[i + 1] = vExp[i] - h * xExp[i]
        xImp[i + 1] = (xImp[i] + h * vImp[i]) / (1 + h**2)
        vImp[i + 1] = (vImp[i] - h * xImp[i]) / (1 + h**2)
        tOut[i + 1] = tOut[i] + h
        xAna[i + 1] = xAtT(xi, vi, tOut[i + 1])
        vAna[i + 1] = vAtT(xi, vi, tOut[i + 1])


    # Plots trajectories produced by the explicit and implicit Euler methods
    basicPlot("Phase-space Geometry of the Trajectory from \nExplicit Euler's "+\
              "Method with h = " + str(h) + ", Xi = " + str(xi) + ", Vi = " +\
              str(vi), xExp, vExp, "Position", "Velocity")
    basicPlot("Phase-space Geometry of the Trajectory from \nImplicit Euler's "+\
              "Method with h = " + str(h) + ", Xi = " + str(xi) + ", Vi = " +\
              str(vi), xImp, vImp, "Position", "Velocity")
    basicPlot("Phase-space Geometry of the Trajectory\nDetermined Analytically"+\
              " with h = " + str(h) + ", Xi = " + str(xi) + ", Vi = " +\
              str(vi), xAna, vAna, "Position", "Velocity")


# Plots the phase-space geometry of the trajectories produced by the
# symplectic Euler method when used to model a simple harmonic oscillator
# with a given step-size 'h', given starting position 'xi', and given
# starting velocity 'vi', and for a given number of cycles 'numCycles'.
def phaseSpaceGeoSym(xi, vi, h, numCycles):
    
    # Calculates how many steps of size h are necessary to complete
    # the number of cycles -- rounds up the number of steps if that number
    # is not an integer
    numSteps = int(math.ceil(numCycles * 2 * np.pi / h))
    
    # numpy arrays that store the values of the positon and velocity
    # as found with the explicit and implicit Euler methods
    xSym = np.zeros(numSteps)
    vSym = np.zeros(numSteps)
    
    # sets values at t=0
    xSym[0] = xi
    vSym[0] = vi
    
    # calculates subsequent positions and velociites
    for i in range(numSteps - 1):
        xSym[i + 1] = xSym[i] + h * vSym[i]
        vSym[i + 1] = vSym[i] * (1 - h**2) - h * xSym[i]
    
    # Plots trajectories produced by the symplectic Euler's method
    basicPlot("Phase-space Geometry of the Trajectory from \nSymplectic "+\
              "Euler's Method with h = " + str(h) + ", Xi = " + str(xi) + \
              ", Vi = " + str(vi), xSym, vSym, "Position", "Velocity")


# Plots the normalized energy of an oscillating system as a function of time
# when modelled with the symplectic Euler method with a given step-size 'h',
# given starting position 'xi', and given starting velocity 'vi', and for a
# given number of cycles 'numCycles'.
def symEnergy(xi, vi, h, numCycles):
    
    # Calculates how many steps of size h are necessary to complete
    # the number of cycles -- rounds up the number of steps if that number
    # is not an integer
    numSteps = int(math.ceil(numCycles * 2 * np.pi / h))
    
    # numpy arrays that store the values of the positon, velocity, and energy
    # as found with  symplectic Euler's method, as well as the times for
    # which those valeus are found
    xSym = np.zeros(numSteps)
    vSym = np.zeros(numSteps)
    eSym = np.zeros(numSteps)
    time = np.zeros(numSteps)
    
    # sets values at t=0
    xSym[0] = xi
    vSym[0] = vi
    eSym[0] = xi**2 + vi**2
    
    # calculates subsequent positions, velociites, energies, and times
    for i in range(numSteps - 1):
        xSym[i + 1] = xSym[i] + h * vSym[i]
        vSym[i + 1] = vSym[i] * (1 - h**2) - h * xSym[i]
        time[i + 1] = time[i] + h
        eSym[i + 1] = xSym[i + 1]**2 + vSym[i + 1]**2

    # Plots normalized energy of the system as modelled by the symplectic
    # Euler's method as a function of time
    basicPlot("Energy of the System Modelled with \nSymplectic "+\
              "Euler's Method with h = " + str(h) + ", Xi = " + str(xi) + \
              ", Vi = " + str(vi), time, eSym, "Time", "Energy")


# The calls to the above methods used to generate the figures in my report:

explicitEulerSpring(.1, -1, 1)
truncErr(10, 10)
implicitEulerSpring(.1, -1, 1)
phaseSpaceGeoExpImpAna(2, 0, .01, 10)
phaseSpaceGeoSym(2, 0, .01, 10)
phaseSpaceGeoSym(2, 0, .3, 2)
phaseSpaceGeoSym(2, 0, .7, 2)
symEnergy(2, 0, .01, 5)
