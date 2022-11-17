#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import CircleCollection, LineCollection
from argparse import ArgumentParser


# Construct the argument parse and parse the program arguments
argparser = ArgumentParser(
	prog='dmnetwork_view.py',
	description="Displays a CSV file generated from a DMNetwork using matplotlib"
)
argparser.add_argument('filenames', nargs='+')
argparser.add_argument('-t', '--set-title', metavar='TITLE', action='store', help="Sets the title for the generated plot, overriding any title set in the source file")
argparser.add_argument('-nnl', '--no-node-labels', action='store_true', help="Disables labling nodes in the generated plot")
argparser.add_argument('-nel', '--no-edge-labels', action='store_true', help="Disables labling edges in the generated plot")
argparser.add_argument('-nc', '--set-node-color', metavar='COLOR', action='store', help="Sets the color for drawn nodes, overriding any per-node colors")
argparser.add_argument('-ec', '--set-edge-color', metavar='COLOR', action='store', help="Sets the color for drawn edges, overriding any per-edge colors")
args = argparser.parse_args()


# Parses a color string into an RGBA tuple to use with matplotlib
def parseColor(color, defaultValue = (0, 0, 0, 1)):
	# We only accept string values for parsing
	if not isinstance(color, str):
		return defaultValue
	else:
		# Currently only HTML format colors are accepted, ie. #RRGGBB
		if color[0] == '#':
			rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
			return (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, 1)
		else:
			return defaultValue

# Parses an ID value to a string *consistently*
def parseID(idval):
	# If float has no fractional part format it as an integer instead of leaving trailing zeros
	if isinstance(idval, float):
		if idval % 1 == 0:
			return str(int(idval))
	return str(idval)

# Parse any set node or edge colors
nodeColor = None
edgeColor = None

if 'set_node_color' in args:
	nodeColor = parseColor(args.set_node_color)
if 'set_edge_color' in args:
	edgeColor = parseColor(args.set_edge_color)

# The sets of nodes and edges we read from the CSV file
nodes = {}
edges = {}

# Class for holding the properties of a node
class Node:
	def __init__(self, row):
		# Set our ID
		self.id = parseID(row['ID'])
		
		# Set our position
		x = row['X']
		if np.isnan(x):
			x = 0
		y = row['Y']
		if np.isnan(y):
			y = 0
		z = row['Z']
		if np.isnan(z):
			z = 0
		self.position = (x,y,z)

		# Set name and color, defaulting to a None name if not specified
		self.name = row['Name']
		if not isinstance(self.name, str):
			if np.isnan(self.name):
				self.name = None
			else:
				self.name = str(self.name)

		global nodeColor
		self.color = nodeColor if nodeColor is not None else parseColor(row['Color'])

# Class for holding the properties of an edge
class Edge:
	def __init__(self, row):
		# Set our ID
		self.id = parseID(row['ID'])

		# Determine our starting and ending nodes from the X and Y properties
		global nodes
		self.startNode = nodes[parseID(row['X'])]
		self.endNode = nodes[parseID(row['Y'])]

		# Set name and color, defaulting to a None name if not specified
		self.name = row['Name']
		if not isinstance(self.name, str):
			if np.isnan(self.name):
				self.name = None
			else:
				self.name = str(self.name)

		global edgeColor
		self.color = edgeColor if edgeColor is not None else parseColor(row['Color'])

# Global variable storing a title to use or None
title = None

for filename in args.filenames:
	# Read the data from the supplied CSV file
	data = pd.read_csv(filename, skipinitialspace=True)
	# Iterate each row of data in the file
	for i,row in data.iterrows():
		# Switch based on the type of the entry
		type = row['Type']
		if type == 'Title':
			# Set the title based on name and color
			titleColor = parseColor(row['Color'])
			title = (row['Name'], titleColor)
		elif type == 'Node':
			# Register a new node
			node = Node(row)
			nodes[node.id] = node
		elif type == 'Edge':
			# Register a new edge
			edge = Edge(row)
			edges[edge.id] = edge

# Create Numpy arrays for node and edge positions and colors
nodePositions = np.zeros((len(nodes), 2))
nodeColors = np.zeros((len(nodes), 4))
edgeSegments = np.zeros((len(edges), 2, 2))
edgeColors = np.zeros((len(edges), 4))

# Copy node positions and colors to the arrays
i = 0
for node in nodes.values():
	nodePositions[i] = node.position[0], node.position[1]
	nodeColors[i] = node.color
	i += 1

# Copy edge positions and colors to the arrays
i = 0
for edge in edges.values():
	start = edge.startNode.position
	end = edge.endNode.position
	edgeSegments[i] = [
		(start[0], start[1]),
		(end[0], end[1])
	]
	edgeColors[i] = edge.color
	i += 1

# Get axis for the plot
axis = plt.axes()

# Set the title of the plot if specified
if 'set_title' in args:
	title = (args.set_title, (0, 0, 0, 1))
if not title is None:
	axis.set_title(title[0], color=title[1])

# Add a line collection to the axis for the edges
axis.add_collection(LineCollection(
	segments=edgeSegments,
	colors=edgeColors,
	linewidths=2
))
# Add a circle collection to the axis for the nodes
axis.add_collection(CircleCollection(
	sizes=np.ones(len(nodes)) * (20 ** 2),
	offsets=nodePositions,
	transOffset=axis.transData,
	facecolors=nodeColors,
	# Place above the lines
	zorder=3
))

if not args.no_node_labels:
	# For each node, plot its name at the center of its point
	for node in nodes.values():
		if node.name is not None:
			axis.text(
				x=node.position[0], y=node.position[1],
				s=node.name,
				# Center text vertically and horizontally
				va='center', ha='center',
				# Make sure the text is clipped within the plot area
				clip_on=True
			)

if not args.no_edge_labels:
	# For each edge, plot its name at the center of the line segment
	for edge in edges.values():
		if edge.name is not None:
			axis.text(
				x=(edge.startNode.position[0]+edge.endNode.position[0])/2,
				y=(edge.startNode.position[1]+edge.endNode.position[1])/2,
				s=edge.name,
				va='center', ha='center',
				clip_on=True
			)

# Scale the plot to the content
axis.autoscale()
# Show the plot
plt.show()
