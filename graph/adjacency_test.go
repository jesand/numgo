package graph

import (
	. "github.com/smartystreets/goconvey/convey"
	"math"
	"testing"
)

var graphFactories = []func(directed bool, maxNodes int) Graph{
	NewDenseAdjacencyGraph,
	NewSparseAdjacencyGraph,
}

func makeTree(factory func(directed bool, maxNodes int) Graph) Graph {
	graph := factory(true, 10)
	a := graph.AddNode("a")
	b := graph.AddNode("b")
	c := graph.AddNode("c")
	d := graph.AddNode("d")
	e := graph.AddNode("e")
	f := graph.AddNode("f")
	graph.AddEdgeWithWeight(a, b, 0.1)
	graph.AddEdgeWithWeight(a, c, 0.2)
	graph.AddEdgeWithWeight(b, d, 0.3)
	graph.AddEdgeWithWeight(b, e, 0.4)
	graph.AddEdgeWithWeight(e, f, 0.5)
	return graph
}

func makeDag(factory func(directed bool, maxNodes int) Graph) Graph {
	graph := factory(true, 10)
	a := graph.AddNode("a")
	b := graph.AddNode("b")
	c := graph.AddNode("c")
	d := graph.AddNode("d")
	e := graph.AddNode("e")
	f := graph.AddNode("f")
	graph.AddEdgeWithWeight(a, b, 0.1)
	graph.AddEdgeWithWeight(a, c, 0.2)
	graph.AddEdgeWithWeight(b, d, 0.3)
	graph.AddEdgeWithWeight(b, e, 0.4)
	graph.AddEdgeWithWeight(e, f, 0.5)
	graph.AddEdgeWithWeight(c, f, 0.6)
	return graph
}

func makeCyclic(factory func(directed bool, maxNodes int) Graph) Graph {
	graph := makeTree(factory)
	graph.AddEdgeWithWeight(2, 0, 0.6)
	return graph
}

func TestAddEdge(t *testing.T) {
	for _, factory := range graphFactories {
		Convey("Given an empty graph", t, func() {
			g := factory(true, 10)

			Convey("AddEdge panics", func() {
				So(func() { g.AddEdge(0, 0) }, ShouldPanic)
			})
		})

		Convey("Given a directed graph with multiple nodes", t, func() {
			g := factory(true, 10)
			u := g.AddNode("u")
			v := g.AddNode("v")

			Convey("AddEdge(u, u) succeeds", func() {
				g.AddEdge(u, u)

				nodes, weights := g.ChildrenWithWeights(u)
				So(nodes, ShouldResemble, []Node{{
					ID:        u,
					Name:      "u",
					InDegree:  1,
					OutDegree: 1,
				}})
				So(weights, ShouldResemble, []float64{1})

				nodes, weights = g.ParentsWithWeights(u)
				So(nodes, ShouldResemble, []Node{{
					ID:        u,
					Name:      "u",
					InDegree:  1,
					OutDegree: 1,
				}})
				So(weights, ShouldResemble, []float64{1})
			})

			Convey("AddEdge(u, v) succeeds with valid nodes", func() {
				g.AddEdge(u, v)
				nodes, weights := g.ChildrenWithWeights(u)
				So(nodes, ShouldResemble, []Node{{
					ID:        v,
					Name:      "v",
					InDegree:  1,
					OutDegree: 0,
				}})
				So(weights, ShouldResemble, []float64{1})

				nodes, weights = g.ParentsWithWeights(v)
				So(nodes, ShouldResemble, []Node{{
					ID:        u,
					Name:      "u",
					InDegree:  0,
					OutDegree: 1,
				}})
				So(weights, ShouldResemble, []float64{1})
			})

			Convey("AddEdge(u, v) panics when u is invalid", func() {
				So(func() { g.AddEdge(u+10, v) }, ShouldPanic)
			})

			Convey("AddEdge(u, v) panics when v is invalid", func() {
				So(func() { g.AddEdge(u, u+10) }, ShouldPanic)
			})
		})
		Convey("Given an undirected graph with multiple nodes", t, func() {
			g := factory(false, 10)
			u := g.AddNode("u")
			v := g.AddNode("v")

			Convey("AddEdge(u,v) works, and is traversable in both directions", func() {
				g.AddEdge(u, v)
				So(g.Children(u), ShouldResemble, []Node{{
					ID:        1,
					Name:      "v",
					InDegree:  1,
					OutDegree: 1,
				}})
				So(g.Children(v), ShouldResemble, []Node{{
					ID:        0,
					Name:      "u",
					InDegree:  1,
					OutDegree: 1,
				}})
				So(g.Parents(u), ShouldResemble, []Node{{
					ID:        1,
					Name:      "v",
					InDegree:  1,
					OutDegree: 1,
				}})
				So(g.Parents(v), ShouldResemble, []Node{{
					ID:        0,
					Name:      "u",
					InDegree:  1,
					OutDegree: 1,
				}})
			})
		})
	}
}

func TestAddEdgeWithWeight(t *testing.T) {
	for _, factory := range graphFactories {
		Convey("Given an empty graph", t, func() {
			g := factory(true, 10)

			Convey("AddEdgeWithWeight panics", func() {
				So(func() { g.AddEdgeWithWeight(0, 0, 2.0) }, ShouldPanic)
			})
		})

		Convey("Given a graph with multiple nodes", t, func() {
			g := factory(true, 10)
			u := g.AddNode("u")
			v := g.AddNode("v")

			Convey("AddEdgeWithWeight(u, u) succeeds", func() {
				g.AddEdgeWithWeight(u, u, 2.0)

				nodes, weights := g.ChildrenWithWeights(u)
				So(nodes, ShouldResemble, []Node{{
					ID:        u,
					Name:      "u",
					InDegree:  1,
					OutDegree: 1,
				}})
				So(weights, ShouldResemble, []float64{2})

				nodes, weights = g.ParentsWithWeights(u)
				So(nodes, ShouldResemble, []Node{{
					ID:        u,
					Name:      "u",
					InDegree:  1,
					OutDegree: 1,
				}})
				So(weights, ShouldResemble, []float64{2})
			})

			Convey("AddEdgeWithWeight(u, v) succeeds with valid nodes", func() {
				g.AddEdgeWithWeight(u, v, 2.0)
				nodes, weights := g.ChildrenWithWeights(u)
				So(nodes, ShouldResemble, []Node{{
					ID:        v,
					Name:      "v",
					InDegree:  1,
					OutDegree: 0,
				}})
				So(weights, ShouldResemble, []float64{2})

				nodes, weights = g.ParentsWithWeights(v)
				So(nodes, ShouldResemble, []Node{{
					ID:        u,
					Name:      "u",
					InDegree:  0,
					OutDegree: 1,
				}})
				So(weights, ShouldResemble, []float64{2})
			})

			Convey("AddEdgeWithWeight(u, v) panics when u is invalid", func() {
				So(func() { g.AddEdgeWithWeight(u+10, v, 2.0) }, ShouldPanic)
			})

			Convey("AddEdgeWithWeight(u, v) panics when v is invalid", func() {
				So(func() { g.AddEdgeWithWeight(u, u+10, 2.0) }, ShouldPanic)
			})
		})
	}
}

func TestAddNode(t *testing.T) {
	for _, factory := range graphFactories {
		Convey("Given an empty graph", t, func() {
			g := factory(true, 10)

			Convey("AddNode succeeds", func() {
				So(g.HasNodes(), ShouldBeFalse)
				g.AddNode("u")
				So(g.HasNodes(), ShouldBeTrue)
			})

			Convey("AddNode permits duplicate names", func() {
				g.AddNode("u")
				g.AddNode("u")
				nodes, _ := g.Size()
				So(nodes, ShouldEqual, 2)
			})
		})
	}
}

func TestChildren(t *testing.T) {
	for _, factory := range graphFactories {
		Convey("Given a populated graph", t, func() {
			g := makeTree(factory)
			Convey("Children works when there are children", func() {
				So(g.Children(0), ShouldResemble, []Node{{
					ID:        1,
					Name:      "b",
					InDegree:  1,
					OutDegree: 2,
				}, {
					ID:        2,
					Name:      "c",
					InDegree:  1,
					OutDegree: 0,
				}})
			})

			Convey("Children returns an empty list for a leaf", func() {
				So(g.Children(2), ShouldBeEmpty)
			})
		})
	}
}

func TestChildrenWithWeights(t *testing.T) {
	for _, factory := range graphFactories {
		Convey("Given a populated graph", t, func() {
			g := makeTree(factory)

			Convey("ChildrenWithWeights works when there are children", func() {
				children, weights := g.ChildrenWithWeights(0)
				So(children, ShouldResemble, []Node{{
					ID:        1,
					Name:      "b",
					InDegree:  1,
					OutDegree: 2,
				}, {
					ID:        2,
					Name:      "c",
					InDegree:  1,
					OutDegree: 0,
				}})
				So(weights, ShouldResemble, []float64{0.1, 0.2})
			})

			Convey("ChildrenWithWeights returns an empty list for a leaf", func() {
				children, weights := g.ChildrenWithWeights(2)
				So(children, ShouldBeEmpty)
				So(weights, ShouldBeEmpty)
			})
		})
	}
}

func TestCopy(t *testing.T) {
	for _, factory := range graphFactories {
		Convey("Given an empty graph", t, func() {
			g := factory(true, 10)

			Convey("Copy works", func() {
				h := g.Copy()
				So(h.HasNodes(), ShouldBeFalse)
				So(h.HasEdges(), ShouldBeFalse)
			})
		})

		Convey("Given a populated graph", t, func() {
			g := makeTree(factory)

			Convey("Copy works", func() {
				h := g.Copy()
				nodes, edges := h.Size()
				So(nodes, ShouldEqual, 6)
				So(edges, ShouldEqual, 5)
				So(h.ShortestPathWeights(), ShouldResemble, g.ShortestPathWeights())
			})
		})
	}
}

func TestHasEdges(t *testing.T) {
	for _, factory := range graphFactories {
		Convey("Given an empty graph", t, func() {
			g := factory(true, 10)
			Convey("HasEdges is false", func() {
				So(g.HasEdges(), ShouldBeFalse)
			})
		})

		Convey("Given a graph with nodes but no edges", t, func() {
			g := factory(true, 10)
			g.AddNode("u")
			g.AddNode("v")
			g.AddNode("w")
			Convey("HasEdges is false", func() {
				So(g.HasEdges(), ShouldBeFalse)
			})
		})

		Convey("Given a graph with nodes and edges", t, func() {
			g := makeTree(factory)
			Convey("HasEdges is true", func() {
				So(g.HasEdges(), ShouldBeTrue)
			})
		})
	}
}

func TestHasCycles(t *testing.T) {
	for _, factory := range graphFactories {
		Convey("Given an empty graph", t, func() {
			g := factory(true, 10)
			Convey("HasCycles is false", func() {
				So(g.HasCycles(), ShouldBeFalse)
			})
		})

		Convey("Given a graph with nodes but no edges", t, func() {
			g := factory(true, 10)
			g.AddNode("u")
			g.AddNode("v")
			g.AddNode("w")
			Convey("HasCycles is false", func() {
				So(g.HasCycles(), ShouldBeFalse)
			})
		})

		Convey("Given an acyclic graph with nodes and edges", t, func() {
			g := makeTree(factory)
			Convey("HasCycles is false", func() {
				So(g.HasCycles(), ShouldBeFalse)
			})
		})

		Convey("Given a cyclic graph with nodes and edges", t, func() {
			g := makeCyclic(factory)
			Convey("HasCycles is true", func() {
				So(g.HasCycles(), ShouldBeTrue)
			})
		})
	}
}

func TestHasNodes(t *testing.T) {
	for _, factory := range graphFactories {
		Convey("Given an empty graph", t, func() {
			g := factory(true, 10)
			Convey("HasNodes is false", func() {
				So(g.HasNodes(), ShouldBeFalse)
			})
		})

		Convey("Given a graph with nodes but no edges", t, func() {
			g := factory(true, 10)
			g.AddNode("u")
			g.AddNode("v")
			g.AddNode("w")

			Convey("HasNodes is true", func() {
				So(g.HasNodes(), ShouldBeTrue)
			})
		})
	}
}

func TestHasPath(t *testing.T) {
	for _, factory := range graphFactories {
		Convey("Given a graph with edges (u, v) and (v, w)", t, func() {
			g := makeTree(factory)

			Convey("HasPath(u, v) is true", func() {
				So(g.HasPath(0, 1), ShouldBeTrue)
			})

			Convey("HasPath(u, w) is true", func() {
				So(g.HasPath(0, 3), ShouldBeTrue)
			})

			Convey("HasPath(v, u) is false", func() {
				So(g.HasPath(1, 0), ShouldBeFalse)
			})

			Convey("HasPath(w, u) is false", func() {
				So(g.HasPath(3, 0), ShouldBeFalse)
			})

			Convey("HasPath(w, v) is false", func() {
				So(g.HasPath(3, 1), ShouldBeFalse)
			})
		})
	}
}

func TestIsDag(t *testing.T) {
	for _, factory := range graphFactories {
		Convey("Given an empty graph", t, func() {
			g := factory(true, 10)
			Convey("IsDag is true", func() {
				So(g.IsDag(), ShouldBeTrue)
			})
		})

		Convey("Given a graph with nodes but no edges", t, func() {
			g := factory(true, 10)
			g.AddNode("u")
			g.AddNode("v")
			g.AddNode("w")

			Convey("IsDag is true", func() {
				So(g.IsDag(), ShouldBeTrue)
			})
		})

		Convey("Given an acyclic graph with nodes and edges and some multi-parent nodes", t, func() {
			g := makeDag(factory)
			Convey("IsDag is true", func() {
				So(g.IsDag(), ShouldBeTrue)
			})
		})

		Convey("Given a tree with nodes and edges", t, func() {
			g := makeTree(factory)
			Convey("IsDag is true", func() {
				So(g.IsDag(), ShouldBeTrue)
			})
		})

		Convey("Given a cyclic graph", t, func() {
			g := makeCyclic(factory)
			Convey("IsDag is false", func() {
				So(g.IsDag(), ShouldBeFalse)
			})
		})

		Convey("Given an undirected acyclic graph", t, func() {
			g := factory(false, 10)
			g.AddNode("u")
			Convey("IsDag is false", func() {
				So(g.IsDag(), ShouldBeFalse)
			})
		})
	}
}

func TestIsDirected(t *testing.T) {
	for _, factory := range graphFactories {
		Convey("Given an empty directed graph", t, func() {
			g := factory(true, 10)
			Convey("IsDirected is true", func() {
				So(g.IsDirected(), ShouldBeTrue)
			})
		})

		Convey("Given a directed acyclic graph", t, func() {
			g := makeDag(factory)
			Convey("IsDirected is true", func() {
				So(g.IsDirected(), ShouldBeTrue)
			})
		})

		Convey("Given a directed cyclic graph", t, func() {
			g := makeCyclic(factory)
			Convey("IsDirected is true", func() {
				So(g.IsDirected(), ShouldBeTrue)
			})
		})

		Convey("Given an empty undirected graph", t, func() {
			g := factory(false, 10)
			Convey("IsDirected is false", func() {
				So(g.IsDirected(), ShouldBeFalse)
			})
		})

		Convey("Given an undirected acyclic graph", t, func() {
			g := factory(false, 10)
			u := g.AddNode("u")
			v := g.AddNode("v")
			g.AddEdge(u, v)
			Convey("IsDirected is false", func() {
				So(g.IsDirected(), ShouldBeFalse)
			})
		})
	}
}

func TestIsTree(t *testing.T) {
	for _, factory := range graphFactories {
		Convey("Given an empty graph", t, func() {
			g := factory(true, 10)
			Convey("IsTree is true", func() {
				So(g.IsTree(), ShouldBeTrue)
			})
		})

		Convey("Given a graph with nodes but no edges", t, func() {
			g := factory(true, 10)
			g.AddNode("u")
			g.AddNode("v")
			g.AddNode("w")
			Convey("IsTree is true", func() {
				So(g.IsTree(), ShouldBeTrue)
			})
		})

		Convey("Given an acyclic graph with nodes and edges and some multi-parent nodes", t, func() {
			g := makeDag(factory)
			Convey("IsTree is false", func() {
				So(g.IsTree(), ShouldBeFalse)
			})
		})

		Convey("Given a tree with nodes and edges", t, func() {
			g := makeTree(factory)
			Convey("IsTree is true", func() {
				So(g.IsTree(), ShouldBeTrue)
			})
		})

		Convey("Given a cyclic graph", t, func() {
			g := makeCyclic(factory)
			Convey("IsTree is false", func() {
				So(g.IsTree(), ShouldBeFalse)
			})
		})

		Convey("Given an undirected acyclic graph", t, func() {
			g := factory(false, 10)
			u := g.AddNode("u")
			v := g.AddNode("v")
			g.AddEdge(u, v)
			Convey("IsTree is false", func() {
				So(g.IsTree(), ShouldBeFalse)
			})
		})
	}
}

func TestLeaves(t *testing.T) {
	for _, factory := range graphFactories {
		Convey("Given an empty graph", t, func() {
			g := factory(true, 10)
			Convey("Leaves returns an empty list", func() {
				So(g.Leaves(), ShouldBeEmpty)
			})
		})

		Convey("Given a populated graph", t, func() {
			g := makeTree(factory)
			Convey("Leaves returns the leaves", func() {
				So(g.Leaves(), ShouldResemble, []Node{{
					ID:        2,
					Name:      "c",
					InDegree:  1,
					OutDegree: 0,
				}, {
					ID:        3,
					Name:      "d",
					InDegree:  1,
					OutDegree: 0,
				}, {
					ID:        5,
					Name:      "f",
					InDegree:  1,
					OutDegree: 0,
				}})
			})
		})
	}
}

func TestParents(t *testing.T) {
	for _, factory := range graphFactories {
		Convey("Given a populated graph", t, func() {
			g := makeTree(factory)

			Convey("Parents works when there are parents", func() {
				So(g.Parents(1), ShouldResemble, []Node{{
					ID:        0,
					Name:      "a",
					InDegree:  0,
					OutDegree: 2,
				}})
			})

			Convey("Parents returns an empty list for the root node", func() {
				So(g.Parents(0), ShouldBeEmpty)
			})
		})
	}
}

func TestParentsWithWeights(t *testing.T) {
	for _, factory := range graphFactories {
		Convey("Given a populated graph", t, func() {
			g := makeTree(factory)

			Convey("ParentsWithWeights works when there are parents", func() {
				parents, weights := g.ParentsWithWeights(1)
				So(parents, ShouldResemble, []Node{{
					ID:        0,
					Name:      "a",
					InDegree:  0,
					OutDegree: 2,
				}})
				So(weights, ShouldResemble, []float64{0.1})
			})

			Convey("ParentsWithWeights returns an empty list for the root node", func() {
				parents, weights := g.ParentsWithWeights(0)
				So(parents, ShouldBeEmpty)
				So(weights, ShouldBeEmpty)
			})
		})
	}
}

func TestRemoveEdge(t *testing.T) {
	for _, factory := range graphFactories {
		Convey("Given an empty graph", t, func() {
			g := factory(true, 10)
			Convey("RemoveEdge panics", func() {
				So(func() { g.RemoveEdge(0, 1) }, ShouldPanic)
			})
		})

		Convey("Given a populated graph", t, func() {
			g := makeTree(factory)

			Convey("RemoveEdge(u, v) succeeds when edge (u, v) exists", func() {
				g.RemoveEdge(0, 1)
				So(g.Parents(1), ShouldBeEmpty)
				So(g.Children(0), ShouldResemble, []Node{{
					ID:        2,
					Name:      "c",
					InDegree:  1,
					OutDegree: 0,
				}})
			})

			Convey("RemoveEdge(u, v) succeeds when edge (u, v) does not exist", func() {
				g.RemoveEdge(2, 1)
			})

			Convey("RemoveEdge(u, v) panics when u is invalid", func() {
				So(func() { g.RemoveEdge(11, 0) }, ShouldPanic)
			})

			Convey("RemoveEdge(u, v) panics when v is invalid", func() {
				So(func() { g.RemoveEdge(0, 11) }, ShouldPanic)
			})
		})
	}
}

func TestRoots(t *testing.T) {
	for _, factory := range graphFactories {
		Convey("Given an empty graph", t, func() {
			g := factory(true, 10)
			Convey("Roots returns an empty list", func() {
				So(g.Roots(), ShouldBeEmpty)
			})
		})

		Convey("Given a populated graph", t, func() {
			g := makeTree(factory)
			g.AddNode("g")
			Convey("Roots returns the roots", func() {
				So(g.Roots(), ShouldResemble, []Node{{
					ID:        0,
					Name:      "a",
					InDegree:  0,
					OutDegree: 2,
				}, {
					ID:        6,
					Name:      "g",
					InDegree:  0,
					OutDegree: 0,
				}})
			})
		})
	}
}

func TestShortestPath(t *testing.T) {
	for _, factory := range graphFactories {
		Convey("Given a populated graph", t, func() {
			g := makeDag(factory)

			Convey("ShortestPath(u, u) returns u", func() {
				path, weights := g.ShortestPath(0, 0)
				So(path, ShouldResemble, []Node{{
					ID:        0,
					Name:      "a",
					InDegree:  0,
					OutDegree: 2,
				}})
				So(weights, ShouldResemble, []float64{})
			})

			Convey("ShortestPath(u, v) with edge (u, v) returns u, v", func() {
				path, weights := g.ShortestPath(0, 1)
				So(path, ShouldResemble, []Node{{
					ID:        0,
					Name:      "a",
					InDegree:  0,
					OutDegree: 2,
				}, {
					ID:        1,
					Name:      "b",
					InDegree:  1,
					OutDegree: 2,
				}})
				So(weights, ShouldResemble, []float64{0.1})
			})

			Convey("ShortestPath finds the shortest multi-edge path", func() {
				path, weights := g.ShortestPath(0, 5)
				So(path, ShouldResemble, []Node{{
					ID:        0,
					Name:      "a",
					InDegree:  0,
					OutDegree: 2,
				}, {
					ID:        2,
					Name:      "c",
					InDegree:  1,
					OutDegree: 1,
				}, {
					ID:        5,
					Name:      "f",
					InDegree:  2,
					OutDegree: 0,
				}})
				So(weights, ShouldResemble, []float64{0.2, 0.6})
			})

			Convey("ShortestPath returns nil when there is no path", func() {
				path, weights := g.ShortestPath(5, 0)
				So(path, ShouldBeEmpty)
				So(weights, ShouldBeEmpty)
			})
		})
	}
}

func TestShortestPathWeight(t *testing.T) {
	for _, factory := range graphFactories {
		Convey("Given a populated graph", t, func() {
			g := makeDag(factory)

			Convey("ShortestPathWeight(u, u) returns 0", func() {
				So(g.ShortestPathWeight(0, 0), ShouldEqual, 0)
			})

			Convey("ShortestPathWeight(u, v) with edge (u, v) returns the weight of u, v", func() {
				So(g.ShortestPathWeight(0, 1), ShouldEqual, 0.1)
			})

			Convey("ShortestPathWeight finds the shortest multi-edge path", func() {
				So(g.ShortestPathWeight(0, 5), ShouldEqual, 0.8)
			})

			Convey("ShortestPathWeight returns infinity when there is no path", func() {
				So(g.ShortestPathWeight(5, 0), ShouldEqual, math.Inf(1))
			})
		})
	}
}

func TestShortestPathWeights(t *testing.T) {
	var inf = math.Inf(1)
	for _, factory := range graphFactories {
		Convey("Given an empty graph", t, func() {
			g := factory(true, 10)
			Convey("ShortestPathWeights returns nothing", func() {
				So(g.ShortestPathWeights(), ShouldBeEmpty)
			})
		})

		Convey("Given a disconnected graph", t, func() {
			g := factory(true, 10)
			g.AddNode("u")
			g.AddNode("v")
			g.AddNode("w")
			Convey("ShortestPathWeights returns the correct weights", func() {
				So(g.ShortestPathWeights(), ShouldResemble, [][]float64{{
					0, inf, inf,
				}, {
					inf, 0, inf,
				}, {
					inf, inf, 0,
				}})
			})
		})

		Convey("Given a populated graph", t, func() {
			g := makeDag(factory)
			Convey("ShortestPathWeights returns the correct weights", func() {
				So(g.ShortestPathWeights(), ShouldResemble, [][]float64{{
					0, 0.1, 0.2, 0.4, 0.5, 0.8,
				}, {
					inf, 0, inf, 0.3, 0.4, 0.9,
				}, {
					inf, inf, 0, inf, inf, 0.6,
				}, {
					inf, inf, inf, 0, inf, inf,
				}, {
					inf, inf, inf, inf, 0, 0.5,
				}, {
					inf, inf, inf, inf, inf, 0,
				}})
			})
		})
	}
}

func TestSize(t *testing.T) {
	for _, factory := range graphFactories {
		Convey("Given an empty graph", t, func() {
			g := factory(true, 10)
			Convey("Size() returns 0, 0", func() {
				nodes, edges := g.Size()
				So(nodes, ShouldEqual, 0)
				So(edges, ShouldEqual, 0)
			})
		})

		Convey("Given a populated graph", t, func() {
			g := makeTree(factory)
			Convey("Size() returns the correct values", func() {
				nodes, edges := g.Size()
				So(nodes, ShouldEqual, 6)
				So(edges, ShouldEqual, 5)
			})
		})
	}
}

func TestTopologicalSort(t *testing.T) {
	for _, factory := range graphFactories {
		Convey("Given an empty graph", t, func() {
			g := factory(true, 10)

			Convey("TopologicalSort returns an empty ordering", func() {
				order, err := g.TopologicalSort()
				So(err, ShouldBeNil)
				So(order, ShouldBeEmpty)
			})
		})

		Convey("Given a DAG", t, func() {
			g := makeDag(factory)
			Convey("TopologicalSort returns the correct ordering", func() {
				order, err := g.TopologicalSort()
				So(err, ShouldBeNil)
				So(order, ShouldResemble, []Node{
					{ID: 0, Name: "a", InDegree: 0, OutDegree: 2},
					{ID: 1, Name: "b", InDegree: 1, OutDegree: 2},
					{ID: 2, Name: "c", InDegree: 1, OutDegree: 1},
					{ID: 3, Name: "d", InDegree: 1, OutDegree: 0},
					{ID: 4, Name: "e", InDegree: 1, OutDegree: 1},
					{ID: 5, Name: "f", InDegree: 2, OutDegree: 0},
				})
			})
		})

		Convey("Given a cyclic graph", t, func() {
			g := makeCyclic(factory)
			Convey("TopologicalSort fails", func() {
				order, err := g.TopologicalSort()
				So(order, ShouldBeEmpty)
				So(err, ShouldResemble, ErrGraphIsCyclic{})
			})
		})
	}
}

func TestTransitiveClosure(t *testing.T) {
	for _, factory := range graphFactories {
		Convey("Given an empty graph", t, func() {
			g := factory(true, 10)
			Convey("TransitiveClosure returns an empty graph", func() {
				h := g.TransitiveClosure()
				So(h.HasNodes(), ShouldBeFalse)
				So(h.HasEdges(), ShouldBeFalse)
			})
		})

		Convey("Given a DAG", t, func() {
			g := makeDag(factory)
			Convey("TransitiveClosure returns the correct transitive closure", func() {
				h := g.TransitiveClosure()
				nodes, edges := h.Size()
				So(nodes, ShouldEqual, 6)
				So(edges, ShouldEqual, 10)
				So(h.Children(0), ShouldResemble, []Node{
					{ID: 1, Name: "b", InDegree: 1, OutDegree: 3},
					{ID: 2, Name: "c", InDegree: 1, OutDegree: 1},
					{ID: 3, Name: "d", InDegree: 2, OutDegree: 0},
					{ID: 4, Name: "e", InDegree: 2, OutDegree: 1},
					{ID: 5, Name: "f", InDegree: 4, OutDegree: 0},
				})
				So(h.Children(1), ShouldResemble, []Node{
					{ID: 3, Name: "d", InDegree: 2, OutDegree: 0},
					{ID: 4, Name: "e", InDegree: 2, OutDegree: 1},
					{ID: 5, Name: "f", InDegree: 4, OutDegree: 0},
				})
				So(h.Children(2), ShouldResemble, []Node{
					{ID: 5, Name: "f", InDegree: 4, OutDegree: 0},
				})
				So(h.Children(3), ShouldBeEmpty)
				So(h.Children(4), ShouldResemble, []Node{
					{ID: 5, Name: "f", InDegree: 4, OutDegree: 0},
				})
				So(h.Children(5), ShouldBeEmpty)
			})
		})

		Convey("Given a cyclic graph", t, func() {
			g := makeCyclic(factory)
			Convey("TransitiveClosure returns the correct transitive closure", func() {
				h := g.TransitiveClosure()
				nodes, edges := h.Size()
				So(nodes, ShouldEqual, 6)
				So(edges, ShouldEqual, 14)
				So(h.Children(0), ShouldResemble, []Node{
					{ID: 1, Name: "b", InDegree: 2, OutDegree: 3},
					{ID: 2, Name: "c", InDegree: 1, OutDegree: 5},
					{ID: 3, Name: "d", InDegree: 3, OutDegree: 0},
					{ID: 4, Name: "e", InDegree: 3, OutDegree: 1},
					{ID: 5, Name: "f", InDegree: 4, OutDegree: 0},
				})
				So(h.Children(1), ShouldResemble, []Node{
					{ID: 3, Name: "d", InDegree: 3, OutDegree: 0},
					{ID: 4, Name: "e", InDegree: 3, OutDegree: 1},
					{ID: 5, Name: "f", InDegree: 4, OutDegree: 0},
				})
				So(h.Children(2), ShouldResemble, []Node{
					{ID: 0, Name: "a", InDegree: 1, OutDegree: 5},
					{ID: 1, Name: "b", InDegree: 2, OutDegree: 3},
					{ID: 3, Name: "d", InDegree: 3, OutDegree: 0},
					{ID: 4, Name: "e", InDegree: 3, OutDegree: 1},
					{ID: 5, Name: "f", InDegree: 4, OutDegree: 0},
				})
				So(h.Children(3), ShouldBeEmpty)
				So(h.Children(4), ShouldResemble, []Node{
					{ID: 5, Name: "f", InDegree: 4, OutDegree: 0},
				})
				So(h.Children(5), ShouldBeEmpty)
			})
		})
	}
}

func TestTransitiveReduction(t *testing.T) {
	for _, factory := range graphFactories {
		Convey("Given an empty graph", t, func() {
			g := factory(true, 10)
			Convey("TransitiveReduction returns an empty graph", func() {
				h := g.TransitiveReduction()
				nodes, edges := h.Size()
				So(nodes, ShouldEqual, 0)
				So(edges, ShouldEqual, 0)
			})
		})

		Convey("Given a DAG", t, func() {
			g := makeDag(factory).TransitiveClosure()
			Convey("TransitiveReduction returns the correct transitive reduction", func() {
				h := g.TransitiveReduction()
				nodes, edges := h.Size()
				So(nodes, ShouldEqual, 6)
				So(edges, ShouldEqual, 6)
				So(h.Children(0), ShouldResemble, []Node{
					{ID: 1, Name: "b", InDegree: 1, OutDegree: 2},
					{ID: 2, Name: "c", InDegree: 1, OutDegree: 1},
				})
				So(h.Children(1), ShouldResemble, []Node{
					{ID: 3, Name: "d", InDegree: 1, OutDegree: 0},
					{ID: 4, Name: "e", InDegree: 1, OutDegree: 1},
				})
				So(h.Children(2), ShouldResemble, []Node{
					{ID: 5, Name: "f", InDegree: 2, OutDegree: 0},
				})
				So(h.Children(3), ShouldBeEmpty)
				So(h.Children(4), ShouldResemble, []Node{
					{ID: 5, Name: "f", InDegree: 2, OutDegree: 0},
				})
				So(h.Children(5), ShouldBeEmpty)
			})
		})

		SkipConvey("Given a cyclic graph", t, func() {
			g := makeCyclic(factory)
			Convey("TransitiveReduction returns a correct transitive reduction", func() {
				h := g.TransitiveReduction()
				nodes, edges := h.Size()
				So(nodes, ShouldEqual, 6)
				So(edges, ShouldEqual, 6)
				So(h.Children(0), ShouldResemble, []Node{
					{ID: 1, Name: "b", InDegree: 1, OutDegree: 2},
					{ID: 2, Name: "c", InDegree: 1, OutDegree: 1},
				})
				So(h.Children(1), ShouldResemble, []Node{
					{ID: 3, Name: "d", InDegree: 1, OutDegree: 0},
					{ID: 4, Name: "e", InDegree: 1, OutDegree: 1},
				})
				So(h.Children(2), ShouldResemble, []Node{
					{ID: 0, Name: "a", InDegree: 1, OutDegree: 2},
				})
				So(h.Children(3), ShouldBeEmpty)
				So(h.Children(4), ShouldResemble, []Node{
					{ID: 5, Name: "f", InDegree: 1, OutDegree: 0},
				})
				So(h.Children(5), ShouldBeEmpty)
			})
		})
	}
}

func TestVisitBFS(t *testing.T) {
	var (
		visitedNodes   []Node
		visitedPaths   [][]Node
		visitedWeights [][]float64
		stopAt         NodeID = 20
		visitor               = func(node Node, path []Node, weights []float64) bool {
			visitedNodes = append(visitedNodes, node)
			visitedPaths = append(visitedPaths, path)
			visitedWeights = append(visitedWeights, weights)
			return node.ID == stopAt
		}
	)

	for _, factory := range graphFactories {
		Convey("Given an empty graph", t, func() {
			g := factory(true, 10)
			Convey("VisitBFS panics", func() {
				So(func() { g.VisitBFS(0, visitor) }, ShouldPanic)
			})
		})

		Convey("Given a multi-root graph", t, func() {
			g := makeCyclic(factory)
			g.AddNode("extra")
			visitedNodes = nil
			visitedPaths = nil
			visitedWeights = nil

			Convey("VisitBFS visits descendants of the specified root", func() {
				stopAt = NodeID(-1)
				So(g.VisitBFS(0, visitor), ShouldBeFalse)
				So(visitedNodes, ShouldResemble, []Node{
					{ID: 0, Name: "a", InDegree: 1, OutDegree: 2},
					{ID: 1, Name: "b", InDegree: 1, OutDegree: 2},
					{ID: 2, Name: "c", InDegree: 1, OutDegree: 1},
					{ID: 3, Name: "d", InDegree: 1, OutDegree: 0},
					{ID: 4, Name: "e", InDegree: 1, OutDegree: 1},
					{ID: 5, Name: "f", InDegree: 1, OutDegree: 0},
				})
				So(visitedPaths, ShouldResemble, [][]Node{
					[]Node{
						{ID: 0, Name: "a", InDegree: 1, OutDegree: 2},
					},
					[]Node{
						{ID: 0, Name: "a", InDegree: 1, OutDegree: 2},
						{ID: 1, Name: "b", InDegree: 1, OutDegree: 2},
					},
					[]Node{
						{ID: 0, Name: "a", InDegree: 1, OutDegree: 2},
						{ID: 2, Name: "c", InDegree: 1, OutDegree: 1},
					},
					[]Node{
						{ID: 0, Name: "a", InDegree: 1, OutDegree: 2},
						{ID: 1, Name: "b", InDegree: 1, OutDegree: 2},
						{ID: 3, Name: "d", InDegree: 1, OutDegree: 0},
					},
					[]Node{
						{ID: 0, Name: "a", InDegree: 1, OutDegree: 2},
						{ID: 1, Name: "b", InDegree: 1, OutDegree: 2},
						{ID: 4, Name: "e", InDegree: 1, OutDegree: 1},
					},
					[]Node{
						{ID: 0, Name: "a", InDegree: 1, OutDegree: 2},
						{ID: 1, Name: "b", InDegree: 1, OutDegree: 2},
						{ID: 4, Name: "e", InDegree: 1, OutDegree: 1},
						{ID: 5, Name: "f", InDegree: 1, OutDegree: 0},
					},
				})
				So(visitedWeights, ShouldResemble, [][]float64{
					[]float64{},
					[]float64{0.1},
					[]float64{0.2},
					[]float64{0.1, 0.3},
					[]float64{0.1, 0.4},
					[]float64{0.1, 0.4, 0.5},
				})
			})

			Convey("VisitBFS stops when so instructed", func() {
				stopAt = NodeID(2)
				So(g.VisitBFS(0, visitor), ShouldBeTrue)
				So(visitedNodes, ShouldResemble, []Node{
					{ID: 0, Name: "a", InDegree: 1, OutDegree: 2},
					{ID: 1, Name: "b", InDegree: 1, OutDegree: 2},
					{ID: 2, Name: "c", InDegree: 1, OutDegree: 1},
				})
				So(visitedPaths, ShouldResemble, [][]Node{
					[]Node{
						{ID: 0, Name: "a", InDegree: 1, OutDegree: 2},
					},
					[]Node{
						{ID: 0, Name: "a", InDegree: 1, OutDegree: 2},
						{ID: 1, Name: "b", InDegree: 1, OutDegree: 2},
					},
					[]Node{
						{ID: 0, Name: "a", InDegree: 1, OutDegree: 2},
						{ID: 2, Name: "c", InDegree: 1, OutDegree: 1},
					},
				})
				So(visitedWeights, ShouldResemble, [][]float64{
					[]float64{},
					[]float64{0.1},
					[]float64{0.2},
				})
			})
		})
	}
}

func TestVisitDFS(t *testing.T) {
	var (
		visitedNodes   []Node
		visitedPaths   [][]Node
		visitedWeights [][]float64
		stopAt         NodeID = 20
		visitor               = func(node Node, path []Node, weights []float64) bool {
			visitedNodes = append(visitedNodes, node)
			visitedPaths = append(visitedPaths, path)
			visitedWeights = append(visitedWeights, weights)
			return node.ID == stopAt
		}
	)

	for _, factory := range graphFactories {
		Convey("Given an empty graph", t, func() {
			g := factory(true, 10)
			Convey("VisitDFS panics", func() {
				So(func() { g.VisitDFS(0, visitor) }, ShouldPanic)
			})
		})

		Convey("Given a multi-root graph", t, func() {
			g := makeCyclic(factory)
			g.AddNode("extra")
			visitedNodes = nil
			visitedPaths = nil
			visitedWeights = nil

			Convey("VisitDFS visits descendants of the specified root", func() {
				stopAt = NodeID(-1)
				So(g.VisitDFS(0, visitor), ShouldBeFalse)
				So(visitedNodes, ShouldResemble, []Node{
					{ID: 0, Name: "a", InDegree: 1, OutDegree: 2},
					{ID: 2, Name: "c", InDegree: 1, OutDegree: 1},
					{ID: 1, Name: "b", InDegree: 1, OutDegree: 2},
					{ID: 4, Name: "e", InDegree: 1, OutDegree: 1},
					{ID: 5, Name: "f", InDegree: 1, OutDegree: 0},
					{ID: 3, Name: "d", InDegree: 1, OutDegree: 0},
				})
				So(visitedPaths, ShouldResemble, [][]Node{
					[]Node{
						{ID: 0, Name: "a", InDegree: 1, OutDegree: 2},
					},
					[]Node{
						{ID: 0, Name: "a", InDegree: 1, OutDegree: 2},
						{ID: 2, Name: "c", InDegree: 1, OutDegree: 1},
					},
					[]Node{
						{ID: 0, Name: "a", InDegree: 1, OutDegree: 2},
						{ID: 1, Name: "b", InDegree: 1, OutDegree: 2},
					},
					[]Node{
						{ID: 0, Name: "a", InDegree: 1, OutDegree: 2},
						{ID: 1, Name: "b", InDegree: 1, OutDegree: 2},
						{ID: 4, Name: "e", InDegree: 1, OutDegree: 1},
					},
					[]Node{
						{ID: 0, Name: "a", InDegree: 1, OutDegree: 2},
						{ID: 1, Name: "b", InDegree: 1, OutDegree: 2},
						{ID: 4, Name: "e", InDegree: 1, OutDegree: 1},
						{ID: 5, Name: "f", InDegree: 1, OutDegree: 0},
					},
					[]Node{
						{ID: 0, Name: "a", InDegree: 1, OutDegree: 2},
						{ID: 1, Name: "b", InDegree: 1, OutDegree: 2},
						{ID: 3, Name: "d", InDegree: 1, OutDegree: 0},
					},
				})
				So(visitedWeights, ShouldResemble, [][]float64{
					[]float64{},
					[]float64{0.2},
					[]float64{0.1},
					[]float64{0.1, 0.4},
					[]float64{0.1, 0.4, 0.5},
					[]float64{0.1, 0.3},
				})
			})

			Convey("VisitDFS stops when so instructed", func() {
				stopAt = NodeID(1)
				So(g.VisitDFS(0, visitor), ShouldBeTrue)
				So(visitedNodes, ShouldResemble, []Node{
					{ID: 0, Name: "a", InDegree: 1, OutDegree: 2},
					{ID: 2, Name: "c", InDegree: 1, OutDegree: 1},
					{ID: 1, Name: "b", InDegree: 1, OutDegree: 2},
				})
				So(visitedPaths, ShouldResemble, [][]Node{
					[]Node{
						{ID: 0, Name: "a", InDegree: 1, OutDegree: 2},
					},
					[]Node{
						{ID: 0, Name: "a", InDegree: 1, OutDegree: 2},
						{ID: 2, Name: "c", InDegree: 1, OutDegree: 1},
					},
					[]Node{
						{ID: 0, Name: "a", InDegree: 1, OutDegree: 2},
						{ID: 1, Name: "b", InDegree: 1, OutDegree: 2},
					},
				})
				So(visitedWeights, ShouldResemble, [][]float64{
					[]float64{},
					[]float64{0.2},
					[]float64{0.1},
				})
			})
		})
	}
}
