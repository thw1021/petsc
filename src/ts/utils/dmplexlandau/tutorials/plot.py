#!/usr/bin/env python3
#
# ./plot.py out
#
import sys, os, math, glob
import matplotlib.pyplot as plt
import numpy as npy
import matplotlib.ticker as ticker
import random as rand
from numpy import array
import locale
import pandas as pd
locale.setlocale(locale.LC_ALL, '')
#plt.rcParams["font.size"] = 11
#plt.rcParams["font.weight"] = "bold"
#plt.rcParams["axes.labelweight"] = "bold"
#plt.rcParams.update({'font.size': 11})
#plt.rcParams.update({'font.weight': "bold"})
idx = 0
t_off = 0
Temps = npy.zeros([14410,5])
for filename in sys.argv[1:]: # just one
    parts = filename.split('-')
    #parts = base[0].split('-')
    iso_type = parts[1]
    shift_type = parts[2]
    normal_species_type = parts[3]
    elem_type = parts[4]
    ts_type = parts[5]
    ts_type_rtol = ts_type.split('_')
    ts_type_name = ts_type_rtol[0] + '-' + ts_type_rtol[1]
    if len(parts) > 6 : tag = '-' + parts[6]
    else : tag = ''
    #print (ts_type_rtol[1])
    #print (ts_type_rtol[1][0:3])
    if (len(ts_type_rtol) > 1):
        tol = int(ts_type_rtol[1][3])
        ts_type_name_full = ts_type_rtol[0] + r' time step (rtol=$10^{{{:d}}}$)'.format(-tol)
    else: ts_type_name_full = ts_type_rtol[0] + ' time step'
    #print(ts_type_name_full)
    #print(r'$10^{{{:d}}}$'.format(-tol))
    # if ts_type_rtol[0] == 'adaptive' and ts_type_rtol[1][0:4] == 'rtol': ts_type = '.' +  ts_type[1:]
    if normal_species_type == '0':
        normal_species = '$e$ normalized'
        normal_species_short = 'normalize-e'
        xlabel = r'Time ($\tau_e$)'
    else:
        normal_species = 'ion normalized'
        normal_species_short = 'normalize-i'
        xlabel = r'Time ($\tau_i$)'
    print(parts)
    #  0    1    2      3           4          5       6       7     8        9        10       11       12       13   14    15       16     17         18      19
    # step 80) time= 1.439800e+04 temperature (ev): electron: T= 3.0192e+02 T_par= 2.9447e+02 T_perp= 3.0564e+02 ;ion: T= 2.9807e+02 T_par= 2.9656e+02 T_perp= 2.9882e+02
    for text in open(filename,"r"): 
        words = text.split()
        n = len(words)
        if n > 2 and words[1] == 'FormLandau':
            ncells = words[4]
            nips = words[2]
        if n > 1 and words[0] == 'step':
            #print (words)
            Temps[idx,0] = float(words[3])
            if idx==0: Temps[idx,0] = Temps[idx,0] + 0.001
            Temps[idx,1] = float(words[10])
            Temps[idx,2] = float(words[12])
            Temps[idx,3] = float(words[17])
            Temps[idx,4] = float(words[19])
            idx = idx+1
    #print (Temps[:idx,1:])
    #print (Temps[:idx,0])
    series_name = [r'$T_{e,\parallel}$', '$T_{e,\perp}$', '$T_{i,\parallel}$', '$T_{i,\perp}$']
    #print (series_name)
    #marks = ['s','o', 'D', 'P']
    styles = ['bD-','bD:', 'go-', 'go:']
    #
    # plot
    #
    ylabel = 'Temperature (ev)'
    df = pd.DataFrame(data=Temps[:idx,1:], index=Temps[:idx,0]/230, columns=series_name)
    #df2.index.name = 'Nodes (8 GCDs/node)'
    #df2.columns.name = 'dof/'+device+':'  markersize=5,marker='.',
    plt.rcParams["figure.dpi"] = 320
    ax = df.plot(lw=1, colormap='jet', style=styles, markersize=2, logx=True,logy=False,  grid=True, legend=False)
    #ax = df.plot.line()
    #for i, line in enumerate(ax.get_lines()):
     #   line.set_marker(marks[i])
    title='Relaxation, ' + normal_species + ', ' + ts_type_name_full
    ax.set_title(title,pad=20) # , fontdict={'fontsize':16}
    patches, labels = ax.get_legend_handles_labels()
    ax.legend(patches, labels, loc='best') #, fontsize=14
    xmin, xmax, ymin, ymax = plt.axis()
    #ymax = max_thing[idx] 
    xmin, xmax, ymin, ymax = plt.axis([xmin, xmax, 175, ymax])
    ax.set_xlabel(xlabel) #, fontdict={'fontsize':16})
    ax.set_ylabel(ylabel) #, fontdict={'fontsize':16})
    plt.savefig('temperature-relaxation-' + normal_species_short + '-'+ elem_type + '-'+ ts_type_name + '-'+ shift_type + tag + '.png',bbox_inches='tight')
    #latex table
    #print(df2.to_latex(longtable=False,escape=False,float_format="{:0.2f}".format, caption = prob + '-- ' + type_name[idx], label='tab:' + '_' + lang + machine + '_' + type_tag[idx]))
    print('ncells = ',ncells,', nips = ',nips)
