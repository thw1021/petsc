
# from fileinput import filename
# from itertools import count
# from platform import node
# from turtle import color
# from matplotlib import projections
# from mpl_toolkits import mplot3d
# from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
#from matplotlib.colors import ListedColormap, BoundaryNorm
import time #for timing data
import colorsys
from matplotlib import colors as mcolors, rcParams
import sys

""" command line inputs
    use 3d or 2d or both as cmd line arg to select plot mode
    no arg/2d -> plot 2d only
    3d -> plot 3d only
    3d 2d -> plot both
"""
narg = len(sys.argv)
if   narg == 1:  pmode = ['-2d']
elif narg == 2:  pmode = [sys.argv[1]]
elif narg == 3:  pmode = [sys.argv[1], sys.argv[2]]

"""  To Make Full network we utilize a node structure to house all
    cordinate data and coloring data from the read in.
 """
class Node(object):
     def __init__(self, x, y, color, id):
        self.x = x
        self.y = y
        self.color = color
        self.id = id

"""  To Make Full network we utilize a edge structure to house all
    data of each edge, where coordinates are [[x1,y1],[x2,y2]] to fit
    line collection format
 """
class Edge(object):
     def __init__(self, cord, color):
        self.cord =  cord
        self.color = color


"""  Function for data read in of nodes.
    We use the node struct so we can create full network visualization.
 """
def readNodes(nodes,file,nv):
    line = file.readline() # Remove Buffer line
    for i in range(nv):
        line = file.readline()
        line = line.strip('\n')
        data = line.split(',')
        node = Node(float(data[0]),float(data[1]),float(data[2]),int(data[3]))
        nodes.append(node)

"""  Function for data read in edges. The data format is
    # of subedges>=1, this is so we can color in edges to reflect a DMDA that may exist there.
    The edges struct then houses all data from this file related to the edges.
 """
def readEdgeData(file,ne,edges):
    line = file.readline() # Remove Buffer line
    for i in range(ne):
        line = file.readline()
        line1 = line.strip('\n')
        data = line1.split(',')
        [n_sublines,x1,y1] = [int(data[0]),float(data[1]),float(data[2])]

        for j in range(n_sublines-1):
            line = file.readline()
            line1 = line.strip('\n')
            data = line1.split(',')
            [x2,y2,c] = [float(data[0]),float(data[1]),float(data[2])]
            cord = [[x1,y1],[x2,y2]]
            edge = Edge(cord,c)
            edges.append(edge)
            x1 = x2
            y1 = y2
"""
    Translates a list of Node objects into a list of x and y coordinates and the
    color we wish to apply to the nodes.
"""
def getNodeData(nodes,x,y,col,labels):
    for node in nodes:
        x.append(node.x)
        y.append(node.y)
        col.append(node.color)
        labels.append(str(node.id))

"""
    Translates a list of edge objects into a list of [[x1,y1],[x2,y2]] coordinates and the
    color we wish to apply to that edge. The coordinates are used in the LineCollection function
    to print all edges efficiently
"""
def getEdgeData(edges,cord,e_color):
    for edge in edges:
        cord.append(edge.cord)
        e_color.append(edge.color)

"""
    Tp print lines in 3d we are using the 2D [[x1,y1],[x2,y2]] coordinates and projecting
    them into 3D using the value of the edge. This is also used to color the system.
    Should there be a need for 3D network connections this can be adjusted easily.
"""
def getEdgeData3d(edges,cord,e_color):
    for edge in edges:
        [j,k] = [i for i in edge.cord]
        data = [[j[0],j[1],edge.color],[k[0],k[1],edge.color]]
        e_color.append(edge.color)
        cord.append(data)

def man_cmap(cmap, value=1.):
    colors = cmap(np.arange(cmap.N))
    hls = np.array([colorsys.rgb_to_hls(*c) for c in colors[:,:3]])  # reading node color
    hls[:,1] *= value
    rgb = np.clip(np.array([colorsys.hls_to_rgb(*c) for c in hls]), 0,1)
    return mcolors.LinearSegmentedColormap.from_list("", rgb)

"""
    Plots the network in 2D colors biased on edge value.
    data from nodes and edges are extracted here to reduce amount
    of inputs, thus e_color,cord,x,y, and col are calculated here.
"""
def plotNet2d(Title,nodes,edges):
    # Pre define list values to be obtained
    e_color,cord,x,y,col,labels = [],[],[],[],[],[]

    getEdgeData(edges,cord,e_color)
    getNodeData(nodes,x,y,col,labels)

    e_color = np.array(e_color) # rearrange the list into a np.array that is required.
    col = np.array(col)
    col2 = np.concatenate((col, e_color))
    col2 = np.array(col2)

    #Set a new figure to be made. I originally used subfigures, but having unequal rows
    #and columns skewed the appearance. Have each on their own figure prevents this.
    fig = plt.figure()
    axs = plt.axes()
    axs.set_title(Title)

    cmap = man_cmap(plt.cm.get_cmap('jet'), 1.25)

    norm = plt.Normalize(np.min(col2), np.max(col2)) # sets the norm to have max/min in line with data read in.
    #norm = plt.Normalize(e_color.min(), e_color.max()) # sets the norm to have max/min in line with data read in.
    lc = LineCollection(cord, cmap=cmap, norm=norm) # cmap='jet' sets color mapping.

    #Set the values used for colormapping
    lc.set_array(e_color) # decalres how we color each edge
    lc.set_linewidth(2)
    line = axs.add_collection(lc)
    fig.colorbar(line, ax=axs) # adds a legend for the coloring of the system

    axs.scatter(x, y,c=col, marker='o', cmap=cmap, zorder=3, s=rcParams['lines.markersize'] ** 3) # Plots Nodes biased on location and color.
    for i, lbl in enumerate(labels):
        axs.annotate(lbl, (x[i], y[i]), ha='center', va='center', color='white', zorder=4) # for vertex number

"""
    Plots the network in 3D colors biased on edge value.
    Nodes are not currently printed in 3D
    Edges are colored and ploted in z-dir by the same value
    Due to the number of lines to print and to use coloring, we are using the Line3DCollection
"""
def plotNet3d(Title,edges,xmin,xmax,ymin,ymax):
    e_color,cord = [],[] #declare the lists used for coloring

    getEdgeData3d(edges,cord,e_color)
    e_color = np.array(e_color)
    norm = plt.Normalize(e_color.min(), e_color.max())

    # Create the 3D-line collection object
    lc = Line3DCollection(cord, cmap='jet', norm=norm)
    lc.set_array(e_color)
    lc.set_linewidth(2)

    #plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_zlim(0, e_color.max())
    ax.axes.set_xlim3d(xmin,xmax)
    ax.axes.set_ylim3d(ymin,ymax)
    plt.title(Title)
    ax.add_collection3d(lc, zs=e_color, zdir='z')

""" 3d plot of Nodes
"""
def plotNet3dNew(Title,edges,xmin,xmax,ymin,ymax):
    e_color, cord = [],[]
    getEdgeData3d(edges,cord,e_color)
    e_color = np.array(e_color)


def main():
    nodes_full,edges_full = [],[] # declare lists to make Full network
    start = time.process_time()
    fname = "Net_proc0_snet.txt" # open initial file to get mpi_size
    f = open(fname,"r")
    print("\nBegin Reading from %s"% fname)
    line = f.readline()
    line1 = line.strip('\n')
    [xmin1,xmax1,ymin1,ymax1] = [float(i) for i in line1.split(',')]

    line = f.readline()
    line1 = line.strip('\n')
    [nv,ne,mpi_size] = [int(i) for i in line1.split(',')]

    [xmin,xmax,ymin,ymax] = [xmin1,xmax1,ymin1,ymax1]
    #fig, axs = plt.subplots(1,mpi_size)

    for i in range(mpi_size):
        nodes_sub = []
        edges_sub = []

        if i>0:
            start = time.process_time()
            fname = "Net_proc%d_snet.txt" % i
            print("\nBegin Reading from %s"% fname)
            f = open(fname,"r")
            line = f.readline()
            line1 = line.strip('\n')
            [xmin1,xmax1,ymin1,ymax1] = [float(i) for i in line1.split(',')]

            line = f.readline()
            line1 = line.strip('\n')
            [nv,ne,mpi_size] = [int(i) for i in line1.split(',')]

            if xmin1<xmin:
                xmin = xmin1
            if ymin1<ymin:
                ymin = ymin1
            if xmax1>xmax:
                xmax = xmax1
            if ymax1>ymax:
                ymax = ymax1

        readNodes(nodes_sub,f,nv)
        readEdgeData(f,ne,edges_sub)
        f.close()

        print("Finished Reading from %s in %d"% (fname, (time.process_time()-start)))
        start = time.process_time()
        if mpi_size>1:
            [nodes_full.append(x) for x in nodes_sub if x not in nodes_full] # Create list of all Nodes
            [edges_full.append(x) for x in edges_sub if x not in edges_full] # Create list of all Edges
        print("Connected %s to Full Network in %d sec"% (fname, (time.process_time()-start)))
        start = time.process_time()
        for mode in pmode:
            if mode == '-2d': plotNet2d("SubNetwork, Proc[%d]" % i,nodes_sub,edges_sub)
            if mode == '-3d': plotNet3d("Processor [%d] Network" % i,edges_sub,xmin1,xmax1,ymin1,ymax1)
        print("Setup %s to plot in %d sec"% (fname, (time.process_time()-start)))
        plotNet3dNew("Processor [%d] Network" % i,edges_sub,xmin1,xmax1,ymin1,ymax1)

    if mpi_size>1:
        start = time.process_time()
        for mode in pmode:
            if mode == '-2d': plotNet2d("Full Network",nodes_full,edges_full)
            if mode == '-3d': plotNet3d("3D Full Network",edges_full,xmin,xmax,ymin,ymax)
        print("Setup Full Net to plot in %d sec"% ((time.process_time()-start)))
    print("Ploting all Networks")
    plt.show()

main()
