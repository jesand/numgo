package graph

import (
	"github.com/jesand/numgo/container"
	"github.com/jesand/numgo/matrix"
	"math"
)

// Create a graph using a dense adjacency matrix
func NewDenseAdjacencyGraph(directed bool, maxNodes int) Graph {
	return &adjacencyGraph{
		directed: directed,
		edges:    matrix.Dense(maxNodes, maxNodes).M(),
	}
}

// Create a graph using a sparse adjacency matrix
func NewSparseAdjacencyGraph(directed bool, maxNodes int) Graph {
	return &adjacencyGraph{
		directed: directed,
		edges:    matrix.SparseCoo(maxNodes, maxNodes),
	}
}

// A graph represented as an adjacency matrix
type adjacencyGraph struct {

	// Whether the graph is directed
	directed bool

	// The node information
	nodes []Node

	// The adjacency matrix
	edges matrix.Matrix
}

// Add an edge to the graph. If the graph is not directed, the edge is
// bidirectional. If the graph is weighted, the edge is assigned weight 1.
func (graph *adjacencyGraph) AddEdge(from, to NodeID) {
	graph.AddEdgeWithWeight(from, to, 1)
}

// Add an edge with a weight to the graph.
func (graph *adjacencyGraph) AddEdgeWithWeight(from, to NodeID, weight float64) {
	if int(from) >= len(graph.nodes) || int(to) >= len(graph.nodes) {
		panic(ErrInvalidNode{})
	}
	graph.edges.ItemSet(weight, int(from), int(to))
	graph.nodes[from].OutDegree++
	graph.nodes[to].InDegree++
	if !graph.directed {
		graph.edges.ItemSet(weight, int(to), int(from))
		graph.nodes[to].OutDegree++
		graph.nodes[from].InDegree++
	}
}

// Add a vertex to the graph, with optional name. If the graph's internal
// storage runs out of capacity, this will panic with ErrGraphCapacity.
func (graph *adjacencyGraph) AddNode(name string) NodeID {
	if len(graph.nodes) >= graph.edges.Rows() {
		panic(ErrGraphCapacity{})
	}
	id := NodeID(len(graph.nodes))
	graph.nodes = append(graph.nodes, Node{
		ID:        id,
		Name:      name,
		InDegree:  0,
		OutDegree: 0,
	})
	return id
}

// Returns all children of a given node
func (graph adjacencyGraph) Children(of NodeID) (children []Node) {
	if int(of) >= len(graph.nodes) {
		panic(ErrInvalidNode{})
	}
	for child := 0; child < len(graph.nodes); child++ {
		if graph.edges.Item(int(of), int(child)) != 0 {
			children = append(children, graph.nodes[child])
		}
	}
	return
}

// Returns all children of a given node and their corresponding edge weights
func (graph adjacencyGraph) ChildrenWithWeights(of NodeID) (children []Node, weights []float64) {
	if int(of) >= len(graph.nodes) {
		panic(ErrInvalidNode{})
	}
	for child := 0; child < len(graph.nodes); child++ {
		weight := graph.edges.Item(int(of), int(child))
		if weight != 0 {
			children = append(children, graph.nodes[child])
			weights = append(weights, weight)
		}
	}
	return
}

// Make a copy of the graph
func (graph adjacencyGraph) Copy() Graph {
	result := &adjacencyGraph{
		directed: graph.directed,
		nodes:    make([]Node, len(graph.nodes)),
		edges:    graph.edges.Copy().M(),
	}
	copy(result.nodes[:], graph.nodes[:])
	return result
}

// Ask whether the graph contains any edges
func (graph adjacencyGraph) HasEdges() bool {
	return graph.edges.CountNonzero() > 0
}

// Ask whether the graph contains cycles
func (graph adjacencyGraph) HasCycles() bool {
	_, err := graph.TopologicalSort()
	return err != nil
}

// Ask whether the graph contains any nodes
func (graph adjacencyGraph) HasNodes() bool {
	return len(graph.nodes) > 0
}

// Ask whether a path exists between two nodes
func (graph adjacencyGraph) HasPath(from, to NodeID) bool {
	if int(from) >= len(graph.nodes) || int(to) >= len(graph.nodes) {
		panic(ErrInvalidNode{})
	}
	p, _ := graph.ShortestPath(from, to)
	return len(p) > 0
}

// Ask whether the graph is a directed acyclic graph
func (graph adjacencyGraph) IsDag() bool {
	return graph.IsDirected() && !graph.HasCycles()
}

// Ask whether the graph is directed
func (graph adjacencyGraph) IsDirected() bool {
	return graph.directed
}

// Ask whether the graph is a tree
func (graph adjacencyGraph) IsTree() bool {
	if !graph.IsDag() {
		return false
	}
	for _, node := range graph.nodes {
		if node.InDegree > 1 {
			return false
		}
	}
	return true
}

// Returns all nodes with out-degree zero
func (graph adjacencyGraph) Leaves() (leaves []Node) {
	for _, node := range graph.nodes {
		if node.OutDegree == 0 {
			leaves = append(leaves, node)
		}
	}
	return
}

// Get the node with the given ID
func (graph adjacencyGraph) Node(id NodeID) Node {
	if int(id) >= len(graph.nodes) {
		panic(ErrInvalidNode{})
	}
	return graph.nodes[id]
}

// Returns all parents of a given node
func (graph adjacencyGraph) Parents(of NodeID) (parents []Node) {
	if int(of) >= len(graph.nodes) {
		panic(ErrInvalidNode{})
	}
	for parent := 0; parent < len(graph.nodes); parent++ {
		if graph.edges.Item(int(parent), int(of)) != 0 {
			parents = append(parents, graph.nodes[parent])
		}
	}
	return
}

// Returns all parents of a given node and their corresponding edge weights
func (graph adjacencyGraph) ParentsWithWeights(of NodeID) (parents []Node, weights []float64) {
	if int(of) >= len(graph.nodes) {
		panic(ErrInvalidNode{})
	}
	for parent := 0; parent < len(graph.nodes); parent++ {
		weight := graph.edges.Item(int(parent), int(of))
		if weight != 0 {
			parents = append(parents, graph.nodes[parent])
			weights = append(weights, weight)
		}
	}
	return
}

// Remove an edge from the graph
func (graph *adjacencyGraph) RemoveEdge(from, to NodeID) {
	if int(from) >= len(graph.nodes) || int(to) >= len(graph.nodes) {
		panic(ErrInvalidNode{})
	} else if graph.edges.Item(int(from), int(to)) == 0 {
		return
	}
	graph.edges.ItemSet(0.0, int(from), int(to))
	graph.nodes[from].OutDegree--
	graph.nodes[to].InDegree--
	if !graph.directed {
		graph.edges.ItemSet(0.0, int(to), int(from))
		graph.nodes[to].OutDegree--
		graph.nodes[from].InDegree--
	}
}

// Returns all nodes with in-degree zero
func (graph adjacencyGraph) Roots() (roots []Node) {
	for _, node := range graph.nodes {
		if node.InDegree == 0 {
			roots = append(roots, node)
		}
	}
	return
}

// Returns the shortest path between two nodes and the edge weights along
// the path
func (graph adjacencyGraph) ShortestPath(from, to NodeID) (p []Node, w []float64) {
	if int(from) >= len(graph.nodes) || int(to) >= len(graph.nodes) {
		panic(ErrInvalidNode{})
	}
	if from == to {
		return []Node{graph.nodes[from]}, []float64{}
	}
	var isNode = func(node Node, path []Node, weights []float64) bool {
		if node.ID == to {
			p = path
			w = weights
			return true
		}
		return false
	}
	graph.VisitBFS(from, isNode)
	return
}

// Returns the total weight along the shortest path between two nodes, or
// 0 if there is no such path.
func (graph adjacencyGraph) ShortestPathWeight(from, to NodeID) (weight float64) {
	if int(from) >= len(graph.nodes) || int(to) >= len(graph.nodes) {
		panic(ErrInvalidNode{})
	}
	path, weights := graph.ShortestPath(from, to)
	if len(path) == 0 {
		return math.Inf(1)
	}
	weight = 0
	for _, w := range weights {
		weight += w
	}
	return
}

// Returns the weights along all pairwise shortest paths in the graph.
// In the returned array, weights[from][to] gives the minimum path weight
// from the node with ID=from to the node with ID=to. The weight will be
// positive infinity if there is no such path.
func (graph adjacencyGraph) ShortestPathWeights() (weights [][]float64) {
	// This is an implementation of the Floyd-Warshall algorithm.

	// Initialize all path weights
	var numNodes = len(graph.nodes)
	for i := 0; i < numNodes; i++ {
		weights = append(weights, make([]float64, numNodes))
		for j := 0; j < numNodes; j++ {
			if i != j {
				weight := graph.edges.Item(i, j)
				if weight == 0 {
					weight = math.Inf(1)
				}
				weights[i][j] = weight
			}
		}
	}

	// Update based on shortest paths
	for k := 0; k < numNodes; k++ {
		for i := 0; i < numNodes; i++ {
			for j := 0; j < numNodes; j++ {
				path := weights[i][k] + weights[k][j]
				if path < weights[i][j] {
					weights[i][j] = path
				}
			}
		}
	}

	return
}

// Returns the number of nodes and edges in the graph
func (graph adjacencyGraph) Size() (nodes, edges int) {
	return len(graph.nodes), graph.edges.CountNonzero()
}

// Returns a topological sort of the graph, if possible. All nodes will
// follow their ancestors in the resulting list. If there is no path between
// a given pair of nodes, their ordering is chosen arbitrarily.
// If the graph is not acyclic, fails with ErrGraphIsCyclic.
func (graph adjacencyGraph) TopologicalSort() (order []Node, err error) {
	var (
		g     = graph.Copy()
		queue = g.Roots()
	)
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		order = append(order, graph.Node(node.ID))
		for _, child := range g.Children(node.ID) {
			if child.InDegree == 1 {
				queue = append(queue, child)
			}
			g.RemoveEdge(node.ID, child.ID)
		}
	}
	if g.HasEdges() {
		return nil, ErrGraphIsCyclic{}
	}
	return
}

// Create the transitive closure of the graph. Adds edges so that all nodes
// reachable from a given node have an edge between them. Any added edges
// will be assigned a weight equal to the shortest path weight between the
// nodes in the original graph.
func (graph adjacencyGraph) TransitiveClosure() Graph {
	var (
		numNodes = len(graph.nodes)
		g        = graph.Copy()
		weights  = g.ShortestPathWeights()
	)
	for i := 0; i < numNodes; i++ {
		for j := 0; j < numNodes; j++ {
			if i != j && graph.edges.Item(i, j) == 0 && !math.IsInf(weights[i][j], 1) {
				g.AddEdgeWithWeight(NodeID(i), NodeID(j), weights[i][j])
			}
		}
	}
	return g
}

// Create the transitive reduction of the graph. Keeps only edges necessary
// to preserve all paths in the graph. The behavior of this algorithm is
// undefined unless the graph is a DAG.
func (graph adjacencyGraph) TransitiveReduction() Graph {
	var (
		numNodes = len(graph.nodes)
		g        = graph.Copy().(*adjacencyGraph)
	)
	weights := g.ShortestPathWeights()
	for i := 0; i < numNodes; i++ {
		for j := 0; j < numNodes; j++ {
			if weights[i][j] != 0 && !math.IsInf(weights[i][j], 1) {
				for k := 0; k < numNodes; k++ {
					if weights[j][k] != 0 && !math.IsInf(weights[j][k], 1) {
						g.RemoveEdge(NodeID(i), NodeID(k))
					}
				}
			}
		}
	}
	return g
}

// Information for visiting nodes
type visitInfo struct {
	Node    Node
	Path    []Node
	Weights []float64
}

// Generic graph visitor method
func visit(graph Graph, frontier container.List, fn NodeVisitor) bool {
	visited := make(map[NodeID]bool)
	for frontier.Size() > 0 {
		next := frontier.Pop().(*visitInfo)
		visited[next.Node.ID] = true
		if fn(next.Node, next.Path, next.Weights) {
			return true
		}
		children, weights := graph.ChildrenWithWeights(next.Node.ID)
		for i := 0; i < len(children); i++ {
			if !visited[children[i].ID] {
				frontier.Push(&visitInfo{
					Node:    children[i],
					Path:    append(next.Path, children[i]),
					Weights: append(next.Weights, weights[i]),
				})
			}
		}
	}
	return false
}

// Visit all descendants using breadth-first search. Returns the result of
// the final call to fn.
func (graph *adjacencyGraph) VisitBFS(from NodeID, fn NodeVisitor) bool {
	if int(from) >= len(graph.nodes) {
		panic(ErrInvalidNode{})
	}
	frontier := &container.Queue{}
	frontier.Push(&visitInfo{
		Node:    graph.nodes[from],
		Path:    []Node{graph.nodes[from]},
		Weights: []float64{},
	})
	return visit(graph, frontier, fn)
}

// Visit all descendants using depth-first search. Returns the result of
// the final call to fn.
func (graph *adjacencyGraph) VisitDFS(from NodeID, fn NodeVisitor) bool {
	if int(from) >= len(graph.nodes) {
		panic(ErrInvalidNode{})
	}
	frontier := &container.Stack{}
	frontier.Push(&visitInfo{
		Node:    graph.nodes[from],
		Path:    []Node{graph.nodes[from]},
		Weights: []float64{},
	})
	return visit(graph, frontier, fn)
}
