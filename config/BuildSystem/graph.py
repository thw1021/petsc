from functools import reduce

class DirectedGraph(object):
  '''This class is for directed graphs with vertices of arbitrary type'''
  def __init__(self):
    '''Create an empty graph'''
    self.vertices = []
    self.inEdges  = {}
    self.outEdges = {}
    return

  def __len__(self):
    return len(self.vertices)

  def __str__(self):
    return 'DirectedGraph with '+str(len(self.vertices))+' vertices and '+str(reduce(lambda k,l: k+l, [len(edgeList) for edgeList in self.inEdges.values()], 0))+' edges'

  def addVertex(self, vertex):
    '''Add a vertex if it does not already exist in the vertex list'''
    if vertex is None: return
    if not vertex in self.vertices:
      self.vertices.append(vertex)
      self.clearEdges(vertex)
    return

  def addEdges(self, vertex, inputs = [], outputs = []):
    '''Define the in and out edges for a vertex by listing the other vertices defining the edges
       - If any vertex does not exist in the graph, it is created'''
    self.addVertex(vertex)
    for input in inputs:
      self.addVertex(input)
      if not vertex is None and not input is None:
        if not input  in self.inEdges[vertex]: self.inEdges[vertex].append(input)
        if not vertex in self.outEdges[input]: self.outEdges[input].append(vertex)
    for output in outputs:
      self.addVertex(output)
      if not vertex is None and not output is None:
        if not vertex in self.inEdges[output]:  self.inEdges[output].append(vertex)
        if not output in self.outEdges[vertex]: self.outEdges[vertex].append(output)
    return

  def getEdges(self, vertex):
    return (self.inEdges[vertex], self.outEdges[vertex])

  def clearEdges(self, vertex, inOnly = 0, outOnly = 0):
    if inOnly and outOnly:
      raise RuntimeError('Inconsistent arguments')
    if not outOnly:
      self.inEdges[vertex]  = []
    if not inOnly:
      self.outEdges[vertex] = []
    return

  def removeVertex(self, vertex):
    '''Remove a vertex if it already exists in the vertex list
       - Also removes all associated edges'''
    if vertex is None: return
    if vertex in self.vertices:
      self.vertices.remove(vertex)
      del self.inEdges[vertex]
      del self.outEdges[vertex]
      for v in self.vertices:
        if vertex in self.inEdges[v]:  self.inEdges[v].remove(vertex)
        if vertex in self.outEdges[v]: self.outEdges[v].remove(vertex)
    return

  def replaceVertex(self, vertex, newVertex):
    '''Replace a vertex with newVertex if it already exists in the vertex list
       - Also transfers all associated edges'''
    if vertex is None or newVertex is None: return
    self.addEdges(newVertex, self.inEdges[vertex], self.outEdges[vertex])
    self.removeVertex(vertex)
    return

  def getRoots(graph):
    '''Return all the sources in the graph (nodes without entering edges)'''
    return [v for v in graph.vertices if not len(graph.getEdges(v)[0])]
  getRoots = staticmethod(getRoots)

  def depthFirstVisit(graph, vertex, seen = None, returnFinished = 0, outEdges = 1):
    '''This is a generator returning vertices in a depth-first traversal only for the subtree rooted at vertex
       - If returnFinished is True, return a vertex when it finishes
       - Otherwise, return a vertex when it is first seen
       - If outEdges is True, proceed along these, otherwise use inEdges'''
    if seen is None: seen = []
    seen.append(vertex)
    if not returnFinished:
      yield vertex
    # Cute trick since outEdges is index 1, and inEdges is index 0
    for v in graph.getEdges(vertex)[outEdges]:
      if not v in seen:
        try:
          for v2 in DirectedGraph.depthFirstVisit(graph, v, seen, returnFinished, outEdges):
            yield v2
        except StopIteration:
          pass
    if returnFinished:
      yield vertex
    return
  depthFirstVisit = staticmethod(depthFirstVisit)

  def depthFirstSearch(graph, returnFinished = 0, outEdges = 1):
    '''This is a generator returning vertices in a depth-first traversal
       - If returnFinished is True, return a vertex when it finishes
       - Otherwise, return a vertex when it is first seen
       - If outEdges is True, proceed along these, otherwise use inEdges'''
    seen = []
    for vertex in graph.vertices:
      if not vertex in seen:
        try:
          for v in DirectedGraph.depthFirstVisit(graph, vertex, seen, returnFinished, outEdges):
            yield v
        except StopIteration:
          pass
    return
  depthFirstSearch = staticmethod(depthFirstSearch)

  def topologicalSort(graph, start = None, outEdges = 1):
    '''Reorder the vertices using topological sort'''
    if start is None:
      vertices = [vertex for vertex in DirectedGraph.depthFirstSearch(graph, returnFinished = 1, outEdges = outEdges)]
    else:
      vertices = [vertex for vertex in DirectedGraph.depthFirstVisit(graph, start, returnFinished = 1, outEdges = outEdges)]
    vertices.reverse()
    for vertex in vertices:
      yield vertex
    return
  topologicalSort = staticmethod(topologicalSort)
