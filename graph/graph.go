package graph

// Identifies a vertex in a graph
type NodeID int

// The data stored for a node
type Node struct {
	ID        NodeID
	Name      string
	InDegree  int
	OutDegree int
}

// A function to invoke when visiting nodes in the graph. Gives the node, the
// path along which the node was found, and the edge weights along that path.
// Should return true if the search should stop.
type NodeVisitor func(node Node, path []Node, weights []float64) bool

// The error raised when too many nodes are added
type ErrGraphCapacity struct{}

func (err ErrGraphCapacity) Error() string {
	return "The graph cannot store any additional nodes"
}

// The error raised when an invalid NodeID is passed to a function
type ErrInvalidNode struct{}

func (err ErrInvalidNode) Error() string {
	return "Invalid graph node ID"
}

// The error returned when a topological sort fails
type ErrGraphIsCyclic struct{}

func (err ErrGraphIsCyclic) Error() string {
	return "Graph contains cycles"
}

// A graph, with various graph manipulation routines.  Note that any method
// which accepts a NodeID will panic with ErrInvalidNode unless the ID has
// previously been returned by a call to AddNode().
type Graph interface {

	// Add an edge to the graph. If the graph is not directed, the edge is
	// bidirectional. If the graph is weighted, the edge is assigned weight 1.
	AddEdge(from, to NodeID)

	// Add an edge with a weight to the graph.
	AddEdgeWithWeight(from, to NodeID, weight float64)

	// Add a vertex to the graph, with optional name. If the graph's internal
	// storage runs out of capacity, this will panic with ErrGraphCapacity.
	AddNode(name string) NodeID

	// Returns all children of a given node
	Children(of NodeID) []Node

	// Returns all children of a given node and their corresponding edge weights
	ChildrenWithWeights(of NodeID) ([]Node, []float64)

	// Make a copy of the graph
	Copy() Graph

	// Ask whether a given edge exists in the graph
	HasEdge(from, to NodeID) bool

	// Ask whether the graph contains any edges
	HasEdges() bool

	// Ask whether the graph contains cycles
	HasCycles() bool

	// Ask whether the graph contains any nodes
	HasNodes() bool

	// Ask whether a path exists between two nodes
	HasPath(from, to NodeID) bool

	// Ask whether the graph is a directed acyclic graph
	IsDag() bool

	// Ask whether the graph is directed
	IsDirected() bool

	// Ask whether the graph is a tree
	IsTree() bool

	// Returns all nodes with out-degree zero
	Leaves() []Node

	// Get the node with the given ID
	Node(id NodeID) Node

	// Returns all parents of a given node
	Parents(of NodeID) []Node

	// Returns all parents of a given node and their corresponding edge weights
	ParentsWithWeights(of NodeID) ([]Node, []float64)

	// Remove an edge from the graph
	RemoveEdge(from, to NodeID)

	// Returns all nodes with in-degree zero
	Roots() []Node

	// Returns the shortest path between two nodes and the edge weights along
	// the path
	ShortestPath(from, to NodeID) ([]Node, []float64)

	// Returns the total weight along the shortest path between two nodes, or
	// 0 if there is no such path.
	ShortestPathWeight(from, to NodeID) float64

	// Returns the weights along all pairwise shortest paths in the graph.
	// In the returned array, weights[from][to] gives the minimum path weight
	// from the node with ID=from to the node with ID=to. The weight will be
	// positive infinity if there is no such path.
	ShortestPathWeights() (weights [][]float64)

	// Returns the number of nodes and edges in the graph
	Size() (nodes, edges int)

	// Returns a string representation of the graph
	String() string

	// Returns a topological sort of the graph, if possible. All nodes will
	// follow their ancestors in the resulting list. If there is no path between
	// a given pair of nodes, their ordering is chosen arbitrarily.
	// If the graph is not acyclic, fails with ErrGraphIsCyclic.
	TopologicalSort() ([]Node, error)

	// Create the transitive closure of the graph. Adds edges so that all nodes
	// reachable from a given node have an edge between them. Any added edges
	// will be assigned a weight equal to the shortest path weight between the
	// nodes in the original graph.
	TransitiveClosure() Graph

	// Create the transitive reduction of the graph. Keeps only edges necessary
	// to preserve all paths in the graph. The behavior of this algorithm is
	// undefined unless the graph is a DAG.
	TransitiveReduction() Graph

	// Visit all descendants using breadth-first search. Returns the result of
	// the final call to fn.
	VisitBFS(from NodeID, fn NodeVisitor) bool

	// Visit all descendants using depth-first search. Returns the result of
	// the final call to fn.
	VisitDFS(from NodeID, fn NodeVisitor) bool
}
