#!/usr/bin/env python
#import sys, os, math, glob
import matplotlib.pyplot as plt
import numpy as np
#import os
#import matplotlib.ticker as ticker
#from numpy import array
plt.close()
spit = np.loadtxt("spitzer_eta.txt")
calc = np.loadtxt("calc_eta.txt")
plt.close()
plt.ylabel('$\eta$')
plt.xlabel('Z')
plt.title('Spitzer vs calculated resistivity')
#ticks = np.arange(.01, .1, 2)
#ticklabels = [r"$10^{}$".format(tick) for tick in ticks]
#plt.yticks(ticks, ticklabels)
plt.plot(spit[:,0],(spit[:,1]),'.',label="Spitzer")
plt.plot(calc[:,0],(calc[:,1]),'*',label="Calculated")
#ax1.set_aspect(aspect=2)
xmin, xmax, ymin, ymax = plt.axis()
#xmin, xmax, ymin, ymax = plt.axis([xmin, xmax, .1, 1.5])
plt.grid()
plt.legend(ncol=1,loc="upper left", shadow=False, fancybox=True,title="$\eta$")
plt.savefig('spitzer_calc_eta_Z.png')
