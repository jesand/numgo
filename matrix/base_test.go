package matrix

import (
	. "github.com/smartystreets/goconvey/convey"
	"math"
	"testing"
)

func TestIndexing(t *testing.T) {
	Convey("Given an array shape (5,3,2)", t, func() {
		shape := []int{5, 3, 2}

		Convey("ndToFlat panics on invalid input", func() {
			So(func() { ndToFlat(shape, []int{}) }, ShouldPanic)
			So(func() { ndToFlat(shape, []int{0, 0}) }, ShouldPanic)
			So(func() { ndToFlat(shape, []int{0, 0, 0, 0}) }, ShouldPanic)
			So(func() { ndToFlat(shape, []int{5, 0, 0}) }, ShouldPanic)
			So(func() { ndToFlat(shape, []int{0, 3, 0}) }, ShouldPanic)
			So(func() { ndToFlat(shape, []int{0, 0, 2}) }, ShouldPanic)
			So(func() { ndToFlat(shape, []int{-6, 0, 0}) }, ShouldPanic)
			So(func() { ndToFlat(shape, []int{0, -4, 0}) }, ShouldPanic)
			So(func() { ndToFlat(shape, []int{0, 0, -3}) }, ShouldPanic)
		})

		Convey("flatToNd panics on invalid input", func() {
			So(func() { flatToNd(shape, 30) }, ShouldPanic)
			So(func() { flatToNd(shape, -31) }, ShouldPanic)
		})

		Convey("ndToFlat works with positive indices", func() {
			next := 0
			for i0 := 0; i0 < 5; i0++ {
				for i1 := 0; i1 < 3; i1++ {
					for i2 := 0; i2 < 2; i2++ {
						So(ndToFlat(shape, []int{i0, i1, i2}), ShouldEqual, next)
						next++
					}
				}
			}
		})

		Convey("flatToNd works with positive indices", func() {
			next := 0
			for i0 := 0; i0 < 5; i0++ {
				for i1 := 0; i1 < 3; i1++ {
					for i2 := 0; i2 < 2; i2++ {
						So(flatToNd(shape, next), ShouldResemble, []int{i0, i1, i2})
						next++
					}
				}
			}
		})

		Convey("ndToFlat works with negative indices", func() {
			next := 0
			for i0 := -5; i0 < 0; i0++ {
				for i1 := -3; i1 < 0; i1++ {
					for i2 := -2; i2 < 0; i2++ {
						So(ndToFlat(shape, []int{i0, i1, i2}), ShouldEqual, next)
						next++
					}
				}
			}
		})

		Convey("flatToNd works with negative indices", func() {
			next := -30
			for i0 := 0; i0 < 5; i0++ {
				for i1 := 0; i1 < 3; i1++ {
					for i2 := 0; i2 < 2; i2++ {
						So(flatToNd(shape, next), ShouldResemble, []int{i0, i1, i2})
						next++
					}
				}
			}
		})
	})
}

func TestAdd(t *testing.T) {
	Convey("Add panics when given arrays of conflicting shapes", t, func() {
		So(func() { Add(Rand(5), Rand(6)) }, ShouldPanic)
		So(func() { Add(Rand(5), Rand(5, 2)) }, ShouldPanic)
	})

	Convey("Given two each dense, sparse coo, and sparse diag arrays", t, func() {
		d1 := A([]int{3, 3},
			1, 2, 3,
			4, 5, 6,
			7, 8, 9)
		d2 := d1.ItemProd(10)
		c1 := SparseCoo(3, 3)
		c1.ItemSet(100, 0, 0)
		c1.ItemSet(102, 0, 2)
		c1.ItemSet(101, 1, 0)
		c2 := SparseCoo(3, 3)
		c2.ItemSet(200, 0, 1)
		c2.ItemSet(202, 0, 2)
		c2.ItemSet(201, 1, 0)
		g1 := Diag(1, 2, 3)
		g2 := Diag(4, 5, 6)

		Convey("Add(dense, dense) gives the correct dense array", func() {
			a := Add(d1, d2)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, DenseArray)
			So(a.Array(), ShouldResemble, []float64{
				11, 22, 33,
				44, 55, 66,
				77, 88, 99,
			})
		})

		Convey("Add(dense, coo) gives the correct dense array", func() {
			a := Add(d1, c1)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, DenseArray)
			So(a.Array(), ShouldResemble, []float64{
				101, 2, 105,
				105, 5, 6,
				7, 8, 9,
			})
		})

		Convey("Add(coo, dense) gives the correct dense array", func() {
			a := Add(c1, d1)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, DenseArray)
			So(a.Array(), ShouldResemble, []float64{
				101, 2, 105,
				105, 5, 6,
				7, 8, 9,
			})
		})

		Convey("Add(dense, diag) gives the correct dense array", func() {
			a := Add(d1, g1)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, DenseArray)
			So(a.Array(), ShouldResemble, []float64{
				2, 2, 3,
				4, 7, 6,
				7, 8, 12,
			})
		})

		Convey("Add(diag, dense) gives the correct dense array", func() {
			a := Add(g1, d1)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, DenseArray)
			So(a.Array(), ShouldResemble, []float64{
				2, 2, 3,
				4, 7, 6,
				7, 8, 12,
			})
		})

		Convey("Add(coo, coo) gives the correct coo array", func() {
			a := Add(c1, c2)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, SparseCooMatrix)
			So(a.Array(), ShouldResemble, []float64{
				100, 200, 304,
				302, 0, 0,
				0, 0, 0,
			})
		})

		Convey("Add(coo, diag) gives the correct coo array", func() {
			a := Add(c1, g1)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, SparseCooMatrix)
			So(a.Array(), ShouldResemble, []float64{
				101, 0, 102,
				101, 2, 0,
				0, 0, 3,
			})
		})

		Convey("Add(diag, coo) gives the correct coo array", func() {
			a := Add(g1, c1)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, SparseCooMatrix)
			So(a.Array(), ShouldResemble, []float64{
				101, 0, 102,
				101, 2, 0,
				0, 0, 3,
			})
		})

		Convey("Add(diag, diag) gives the correct diag array", func() {
			a := Add(g1, g2)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, SparseDiagMatrix)
			So(a.Array(), ShouldResemble, []float64{
				5, 0, 0,
				0, 7, 0,
				0, 0, 9,
			})
		})
	})
}

func TestAll(t *testing.T) {
	Convey("All returns true when all items are set", t, func() {
		So(All(Rand(3)), ShouldBeTrue)
		So(All(A1(1, 2, 3)), ShouldBeTrue)
	})
	Convey("All returns false when all items are not set", t, func() {
		So(All(SparseRand(4, 2, 0.25)), ShouldBeFalse)
		So(All(A1(1, 0, 3)), ShouldBeFalse)
		So(All(Diag(1, 2, 3)), ShouldBeFalse)
	})
	Convey("All returns false when no item is set", t, func() {
		So(All(Zeros(4, 2)), ShouldBeFalse)
	})
}

func TestAllF(t *testing.T) {
	f := func(v float64) bool { return v > 0 }
	Convey("AllF returns true when all items pass the test", t, func() {
		So(AllF(Rand(3), f), ShouldBeTrue)
		So(AllF(A1(1, 2, 3), f), ShouldBeTrue)
	})
	Convey("AllF returns false when any item fails to pass", t, func() {
		So(AllF(SparseRand(4, 2, 0.25), f), ShouldBeFalse)
		So(AllF(Diag(1, -2, 3), f), ShouldBeFalse)
		So(AllF(A1(1, 0, 3), f), ShouldBeFalse)
	})
	Convey("AllF returns false when no item passes the test", t, func() {
		So(AllF(Zeros(4, 2), f), ShouldBeFalse)
	})
}

func TestAllF2(t *testing.T) {
	f := func(v1, v2 float64) bool { return v1 > v2 }
	Convey("AllF2 panics on arrays of different shape", t, func() {
		So(func() { AllF2(Rand(2), f, Rand(3)) }, ShouldPanic)
		So(func() { AllF2(Rand(2), f, Rand(2, 1)) }, ShouldPanic)
	})
	Convey("AllF2 returns true when all items pass the test", t, func() {
		So(AllF2(Rand(3), f, Zeros(3)), ShouldBeTrue)
		So(AllF2(A1(1, 2, 3), f, Zeros(3)), ShouldBeTrue)
	})
	Convey("AllF2 returns false when any item fails to pass", t, func() {
		So(AllF2(SparseRand(4, 2, 0.25), f, Zeros(4, 2)), ShouldBeFalse)
		So(AllF2(Diag(1, -2, 3), f, Zeros(3, 3)), ShouldBeFalse)
		So(AllF2(A1(1, 1, 3), f, A1(1, 0, 3)), ShouldBeFalse)
	})
	Convey("AllF2 returns false when no item passes the test", t, func() {
		So(AllF2(Zeros(4, 2), f, Ones(4, 2)), ShouldBeFalse)
	})
}

func TestAny(t *testing.T) {
	Convey("Any returns true when all items are set", t, func() {
		So(Any(Rand(3)), ShouldBeTrue)
		So(Any(A1(1, 2, 3)), ShouldBeTrue)
	})
	Convey("Any returns true when some items are set", t, func() {
		So(Any(SparseRand(4, 2, 0.25)), ShouldBeTrue)
		So(Any(A1(1, 0, 3)), ShouldBeTrue)
	})
	Convey("Any returns false when no item is set", t, func() {
		So(Any(Zeros(4, 2)), ShouldBeFalse)
	})
}

func TestAnyF(t *testing.T) {
	f := func(v float64) bool { return v > 0 }
	Convey("AnyF returns true when all items passes the test", t, func() {
		So(AnyF(Rand(3), f), ShouldBeTrue)
		So(AnyF(A1(1, 2, 3), f), ShouldBeTrue)
	})
	Convey("AnyF returns true when some items fail to pass", t, func() {
		So(AnyF(SparseRand(4, 2, 0.25), f), ShouldBeTrue)
		So(AnyF(Diag(1, -2, 3), f), ShouldBeTrue)
		So(AnyF(A1(1, 0, 3), f), ShouldBeTrue)
	})
	Convey("AnyF returns false when no item passes the test", t, func() {
		So(AnyF(Zeros(4, 2), f), ShouldBeFalse)
	})
}

func TestAnyF2(t *testing.T) {
	f := func(v1, v2 float64) bool { return v1 > v2 }
	Convey("AnyF2 panics on arrays of different shape", t, func() {
		So(func() { AnyF2(Rand(2), f, Rand(3)) }, ShouldPanic)
		So(func() { AnyF2(Rand(2), f, Rand(2, 1)) }, ShouldPanic)
	})
	Convey("AnyF2 returns true when all items pass the test", t, func() {
		So(AnyF2(Rand(3), f, Zeros(3)), ShouldBeTrue)
		So(AnyF2(A1(1, 2, 3), f, Zeros(3)), ShouldBeTrue)
	})
	Convey("AnyF2 returns true when any item passes", t, func() {
		So(AnyF2(SparseRand(4, 2, 0.25), f, Zeros(4, 2)), ShouldBeTrue)
		So(AnyF2(Diag(1, -2, 3), f, Zeros(3, 3)), ShouldBeTrue)
		So(AnyF2(A1(1, 1, 3), f, A1(1, 0, 3)), ShouldBeTrue)
	})
	Convey("AnyF2 returns false when no item passes the test", t, func() {
		So(AnyF2(Zeros(4, 2), f, Ones(4, 2)), ShouldBeFalse)
	})
}

func TestApply(t *testing.T) {
	Convey("Apply works", t, func() {
		a := A([]int{3, 5},
			1, 2, 3, 4, 5,
			6, 7, 8, 9, 10,
			11, 12, 13, 14, 15,
		)
		a2 := Apply(a, func(v float64) float64 { return 2 * v })
		So(a2.Shape(), ShouldResemble, []int{3, 5})
		So(a2.Array(), ShouldResemble, []float64{
			2, 4, 6, 8, 10,
			12, 14, 16, 18, 20,
			22, 24, 26, 28, 30,
		})
	})
}

func TestConcat(t *testing.T) {
	Convey("Concat() panics with mismatched array sizes", t, func() {
		So(func() { Concat(1, Rand(3), Rand(4)) }, ShouldPanic)
		So(func() { Concat(1, Rand(3), Rand(3, 1)) }, ShouldPanic)
	})

	Convey("Given two 3x3 arrays", t, func() {
		a1 := WithValue(1, 3, 3)
		a2 := WithValue(2, 3, 3)

		Convey("Concat() works with just one array", func() {
			c := Concat(0, a1)
			So(c.Shape(), ShouldResemble, []int{3, 3})
			So(c.Array(), ShouldResemble, []float64{
				1, 1, 1,
				1, 1, 1,
				1, 1, 1,
			})
		})

		Convey("Concat() works on axis 0", func() {
			c := Concat(0, a1, a2)
			So(c.Shape(), ShouldResemble, []int{6, 3})
			So(c.Array(), ShouldResemble, []float64{
				1, 1, 1,
				1, 1, 1,
				1, 1, 1,
				2, 2, 2,
				2, 2, 2,
				2, 2, 2,
			})
		})

		Convey("Concat() works on axis 1", func() {
			c := Concat(1, a1, a2)
			So(c.Shape(), ShouldResemble, []int{3, 6})
			So(c.Array(), ShouldResemble, []float64{
				1, 1, 1, 2, 2, 2,
				1, 1, 1, 2, 2, 2,
				1, 1, 1, 2, 2, 2,
			})
		})

		Convey("Concat() works on axis 2", func() {
			c := Concat(2, a1, a2)
			So(c.Shape(), ShouldResemble, []int{3, 3, 2})
			So(c.Array(), ShouldResemble, []float64{
				1, 2, 1, 2, 1, 2,
				1, 2, 1, 2, 1, 2,
				1, 2, 1, 2, 1, 2,
			})
		})

		Convey("Concat() panics on axis 3", func() {
			So(func() { Concat(3, a1, a2) }, ShouldPanic)
		})
	})
}

func TestDist(t *testing.T) {
	Convey("Given a matrix", t, func() {
		m := A([]int{3, 2},
			1, 2,
			3, 2,
			-1, 4,
		).M()

		Convey("Invalid distance types panic", func() {
			So(func() { m.Dist(DistType(-1)) }, ShouldPanic)
		})

		Convey("Euclidean distance works", func() {
			d := m.Dist(EuclideanDist)
			So(d.Array(), ShouldResemble, []float64{
				0, 2, math.Sqrt(8),
				2, 0, math.Sqrt(20),
				math.Sqrt(8), math.Sqrt(20), 0,
			})
		})
	})
}

func TestDiv(t *testing.T) {
	var inf = math.Inf(+1)

	Convey("Div panics when given arrays of conflicting shapes", t, func() {
		So(func() { Div(Rand(5), Rand(6)) }, ShouldPanic)
		So(func() { Div(Rand(5), Rand(5, 2)) }, ShouldPanic)
	})

	Convey("Given two each dense, sparse coo, and sparse diag arrays", t, func() {
		d1 := A([]int{3, 3},
			1, 2, 3,
			4, 5, 6,
			7, 8, 9)
		d2 := d1.ItemProd(10)
		c1 := SparseCoo(3, 3)
		c1.ItemSet(100, 0, 0)
		c1.ItemSet(102, 0, 2)
		c1.ItemSet(101, 1, 0)
		c2 := SparseCoo(3, 3)
		c2.ItemSet(200, 0, 1)
		c2.ItemSet(202, 0, 2)
		c2.ItemSet(201, 1, 0)
		g1 := Diag(1, 2, 3)
		g2 := Diag(4, 5, 6)

		Convey("Div(dense, dense) gives the correct dense array", func() {
			a := Div(d1, d2)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, DenseArray)
			So(a.Array(), ShouldResemble, []float64{
				.1, .1, .1,
				.1, .1, .1,
				.1, .1, .1,
			})
		})

		Convey("Div(dense, coo) gives the correct dense array", func() {
			a := Div(d1, c1)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, DenseArray)
			So(a.Array(), ShouldResemble, []float64{
				1. / 100, inf, 3. / 102,
				4. / 101, inf, inf,
				inf, inf, inf,
			})
		})

		Convey("Div(coo, dense) gives the correct coo array", func() {
			a := Div(c1, d1)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, SparseCooMatrix)
			So(a.Array(), ShouldResemble, []float64{
				100, 0, 102. / 3,
				101. / 4, 0, 0,
				0, 0, 0,
			})
		})

		Convey("Div(dense, diag) gives the correct dense array", func() {
			a := Div(d1, g1)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, DenseArray)
			So(a.Array(), ShouldResemble, []float64{
				1, inf, inf,
				inf, 2.5, inf,
				inf, inf, 3,
			})
		})

		Convey("Div(diag, dense) gives the correct diag array", func() {
			a := Div(g1, d1)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, SparseDiagMatrix)
			So(a.Array(), ShouldResemble, []float64{
				1, 0, 0,
				0, .4, 0,
				0, 0, 1. / 3,
			})
		})

		Convey("Div(coo, coo) gives the correct coo array", func() {
			a := Div(c1, c2)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, SparseCooMatrix)
			So(a.Array(), ShouldResemble, []float64{
				inf, 0, 102. / 202,
				101. / 201, 0, 0,
				0, 0, 0,
			})
		})

		Convey("Div(coo, diag) gives the correct coo array", func() {
			a := Div(c1, g1)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, SparseCooMatrix)
			So(a.Array(), ShouldResemble, []float64{
				100, 0, inf,
				inf, 0, 0,
				0, 0, 0,
			})
		})

		Convey("Div(diag, coo) gives the correct diag array", func() {
			a := Div(g1, c1)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, SparseDiagMatrix)
			So(a.Array(), ShouldResemble, []float64{
				.01, 0, 0,
				0, inf, 0,
				0, 0, inf,
			})
		})

		Convey("Div(diag, diag) gives the correct diag array", func() {
			a := Div(g1, g2)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, SparseDiagMatrix)
			So(a.Array(), ShouldResemble, []float64{
				.25, 0, 0,
				0, .4, 0,
				0, 0, .5,
			})
		})
	})
}

func TestEqual(t *testing.T) {
	Convey("Equal() returns false when given arrays of inconsistent dimension", t, func() {
		So(Equal(Rand(3), Rand(4)), ShouldBeFalse)
		So(Equal(Rand(3), Rand(3, 1)), ShouldBeFalse)
	})

	Convey("Two random arrays are not equal", t, func() {
		a1 := Rand(3)
		a2 := Rand(3)
		So(Equal(a1, a2), ShouldBeFalse)
	})

	Convey("An array equals itself", t, func() {
		a1 := Rand(3)
		So(Equal(a1, a1), ShouldBeTrue)
	})

	Convey("An array equals its duplicate", t, func() {
		a1 := Rand(3)
		So(Equal(a1, a1.Copy()), ShouldBeTrue)
	})
}

func TestFill(t *testing.T) {
	Convey("Fill() panics when given a sparse array", t, func() {
		So(func() { Fill(SparseRand(3, 1, .25), 1) }, ShouldPanic)
		So(func() { Fill(Eye(3), 1) }, ShouldPanic)
	})

	Convey("Fill() works on a dense array", t, func() {
		a := Dense(2, 3, 4)
		a.Fill(1)

		So(a.Shape(), ShouldResemble, []int{2, 3, 4})
		So(a.Array(), ShouldResemble, []float64{
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		})
	})
}

func TestItemAdd(t *testing.T) {
	Convey("Given a dense array", t, func() {
		a := A([]int{3, 4},
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12)

		Convey("ItemAdd(0) returns the same array", func() {
			s := ItemAdd(a, 0)
			So(s.Shape(), ShouldResemble, []int{3, 4})
			So(s.Sparsity(), ShouldEqual, DenseArray)
			So(s.Array(), ShouldResemble, []float64{
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 10, 11, 12,
			})
		})

		Convey("ItemAdd(1) returns the correct dense array", func() {
			s := ItemAdd(a, 1)
			So(s.Shape(), ShouldResemble, []int{3, 4})
			So(s.Sparsity(), ShouldEqual, DenseArray)
			So(s.Array(), ShouldResemble, []float64{
				2, 3, 4, 5,
				6, 7, 8, 9,
				10, 11, 12, 13,
			})
		})
	})

	Convey("Given a sparse coo array", t, func() {
		a := SparseCoo(3, 4)
		a.ItemSet(1, 0, 0)
		a.ItemSet(2, 0, 1)
		a.ItemSet(3, 1, 0)

		Convey("ItemAdd() returns the correct dense array", func() {
			s := ItemAdd(a, 1)
			So(s.Shape(), ShouldResemble, []int{3, 4})
			So(s.Sparsity(), ShouldEqual, DenseArray)
			So(s.Array(), ShouldResemble, []float64{
				2, 3, 1, 1,
				4, 1, 1, 1,
				1, 1, 1, 1,
			})
		})
	})

	Convey("Given a diag array", t, func() {
		a := Diag(1, 2, 3, 4)

		Convey("ItemAdd() returns the correct dense array", func() {
			s := ItemAdd(a, 1)
			So(s.Shape(), ShouldResemble, []int{4, 4})
			So(s.Sparsity(), ShouldEqual, DenseArray)
			So(s.Array(), ShouldResemble, []float64{
				2, 1, 1, 1,
				1, 3, 1, 1,
				1, 1, 4, 1,
				1, 1, 1, 5,
			})
		})
	})
}

func TestItemDiv(t *testing.T) {
	Convey("Given a dense array", t, func() {
		a := A([]int{3, 4},
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12)

		Convey("ItemDiv(1) returns the same array", func() {
			s := ItemDiv(a, 1)
			So(s.Shape(), ShouldResemble, []int{3, 4})
			So(s.Sparsity(), ShouldEqual, DenseArray)
			So(s.Array(), ShouldResemble, []float64{
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 10, 11, 12,
			})
		})

		Convey("ItemDiv() returns the correct dense array", func() {
			s := ItemDiv(a, 2)
			So(s.Shape(), ShouldResemble, []int{3, 4})
			So(s.Sparsity(), ShouldEqual, DenseArray)
			So(s.Array(), ShouldResemble, []float64{
				.5, 1, 1.5, 2,
				2.5, 3, 3.5, 4,
				4.5, 5, 5.5, 6,
			})
		})
	})

	Convey("Given a sparse coo array", t, func() {
		a := SparseCoo(3, 4)
		a.ItemSet(1, 0, 0)
		a.ItemSet(2, 0, 1)
		a.ItemSet(3, 1, 0)

		Convey("ItemDiv() returns the correct coo array", func() {
			s := ItemDiv(a, 2)
			So(s.Shape(), ShouldResemble, []int{3, 4})
			So(s.Sparsity(), ShouldEqual, SparseCooMatrix)
			So(s.Array(), ShouldResemble, []float64{
				.5, 1, 0, 0,
				1.5, 0, 0, 0,
				0, 0, 0, 0,
			})
		})
	})

	Convey("Given a diag array", t, func() {
		a := Diag(1, 2, 3, 4)

		Convey("ItemDiv() returns the correct diag array", func() {
			s := ItemDiv(a, 2)
			So(s.Shape(), ShouldResemble, []int{4, 4})
			So(s.Sparsity(), ShouldEqual, SparseDiagMatrix)
			So(s.Array(), ShouldResemble, []float64{
				.5, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1.5, 0,
				0, 0, 0, 2,
			})
		})
	})
}

func TestItemProd(t *testing.T) {
	Convey("Given a dense array", t, func() {
		a := A([]int{3, 4},
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12)

		Convey("ItemProd(1) returns the same array", func() {
			s := ItemProd(a, 1)
			So(s.Shape(), ShouldResemble, []int{3, 4})
			So(s.Sparsity(), ShouldEqual, DenseArray)
			So(s.Array(), ShouldResemble, []float64{
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 10, 11, 12,
			})
		})

		Convey("ItemProd() returns the correct dense array", func() {
			s := ItemProd(a, 2)
			So(s.Shape(), ShouldResemble, []int{3, 4})
			So(s.Sparsity(), ShouldEqual, DenseArray)
			So(s.Array(), ShouldResemble, []float64{
				2, 4, 6, 8,
				10, 12, 14, 16,
				18, 20, 22, 24,
			})
		})
	})

	Convey("Given a sparse coo array", t, func() {
		a := SparseCoo(3, 4)
		a.ItemSet(1, 0, 0)
		a.ItemSet(2, 0, 1)
		a.ItemSet(3, 1, 0)

		Convey("ItemProd() returns the correct coo array", func() {
			s := ItemProd(a, 2)
			So(s.Shape(), ShouldResemble, []int{3, 4})
			So(s.Sparsity(), ShouldEqual, SparseCooMatrix)
			So(s.Array(), ShouldResemble, []float64{
				2, 4, 0, 0,
				6, 0, 0, 0,
				0, 0, 0, 0,
			})
		})
	})

	Convey("Given a diag array", t, func() {
		a := Diag(1, 2, 3, 4)

		Convey("ItemProd() returns the correct diag array", func() {
			s := ItemProd(a, 2)
			So(s.Shape(), ShouldResemble, []int{4, 4})
			So(s.Sparsity(), ShouldEqual, SparseDiagMatrix)
			So(s.Array(), ShouldResemble, []float64{
				2, 0, 0, 0,
				0, 4, 0, 0,
				0, 0, 6, 0,
				0, 0, 0, 8,
			})
		})
	})
}

func TestItemSub(t *testing.T) {
	Convey("Given a dense array", t, func() {
		a := A([]int{3, 4},
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12)

		Convey("ItemSub(0) returns the same array", func() {
			s := ItemSub(a, 0)
			So(s.Shape(), ShouldResemble, []int{3, 4})
			So(s.Sparsity(), ShouldEqual, DenseArray)
			So(s.Array(), ShouldResemble, []float64{
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 10, 11, 12,
			})
		})

		Convey("ItemSub() returns the correct dense array", func() {
			s := ItemSub(a, 1)
			So(s.Shape(), ShouldResemble, []int{3, 4})
			So(s.Sparsity(), ShouldEqual, DenseArray)
			So(s.Array(), ShouldResemble, []float64{
				0, 1, 2, 3,
				4, 5, 6, 7,
				8, 9, 10, 11,
			})
		})
	})

	Convey("Given a sparse coo array", t, func() {
		a := SparseCoo(3, 4)
		a.ItemSet(1, 0, 0)
		a.ItemSet(2, 0, 1)
		a.ItemSet(3, 1, 0)

		Convey("ItemSub() returns the correct dense array", func() {
			s := ItemSub(a, 1)
			So(s.Shape(), ShouldResemble, []int{3, 4})
			So(s.Sparsity(), ShouldEqual, DenseArray)
			So(s.Array(), ShouldResemble, []float64{
				0, 1, -1, -1,
				2, -1, -1, -1,
				-1, -1, -1, -1,
			})
		})
	})

	Convey("Given a diag array", t, func() {
		a := Diag(1, 2, 3, 4)

		Convey("ItemSub() returns the correct dense array", func() {
			s := ItemSub(a, 1)
			So(s.Shape(), ShouldResemble, []int{4, 4})
			So(s.Sparsity(), ShouldEqual, DenseArray)
			So(s.Array(), ShouldResemble, []float64{
				0, -1, -1, -1,
				-1, 1, -1, -1,
				-1, -1, 2, -1,
				-1, -1, -1, 3,
			})
		})
	})
}

func TestMProd(t *testing.T) {
	Convey("Given dense 3x5 and 5x2 matrixes", t, func() {
		m1 := A([]int{3, 5},
			1, 2, 3, 4, 5,
			6, 7, 8, 9, 10,
			11, 12, 13, 14, 15).M()
		m2 := A([]int{5, 2},
			10, 9,
			8, 7,
			6, 5,
			4, 3,
			2, 1).M()

		Convey("MProd in the wrong order panics", func() {
			So(func() { MProd(m2, m1) }, ShouldPanic)
		})

		Convey("MProd with just one array returns a copy", func() {
			p := MProd(m1)
			So(p.Shape(), ShouldResemble, []int{3, 5})
			So(p.Array(), ShouldResemble, []float64{
				1, 2, 3, 4, 5,
				6, 7, 8, 9, 10,
				11, 12, 13, 14, 15,
			})
		})

		Convey("MProd is correct", func() {
			p := MProd(m1, m2)
			So(p.Shape(), ShouldResemble, []int{3, 2})
			So(p.Array(), ShouldResemble, []float64{
				70, 55,
				220, 180,
				370, 305,
			})
		})
	})

	Convey("Given sparse coo 3x5 and 5x2 matrixes", t, func() {
		m1 := SparseCoo(3, 5)
		for row := 0; row < 3; row++ {
			for col := 0; col < 5; col++ {
				m1.ItemSet(float64(row*5+col), row, col)
			}
		}
		m2 := SparseCoo(5, 2)
		for row := 0; row < 5; row++ {
			for col := 0; col < 2; col++ {
				m2.ItemSet(float64(row*2+col), row, col)
			}
		}
		Convey("MProd is correct", func() {
			p := MProd(m1, m2)
			So(p.Shape(), ShouldResemble, []int{3, 2})
			So(p.Array(), ShouldResemble, []float64{
				60, 70,
				160, 195,
				260, 320,
			})
		})
	})

	Convey("Given two diag 4x4 matrixes", t, func() {
		m1 := Diag(1, 2, 3, 4)
		m2 := Diag(5, 6, 7, 8)
		Convey("MProd is correct", func() {
			p := MProd(m1, m2)
			So(p.Shape(), ShouldResemble, []int{4, 4})
			So(p.Array(), ShouldResemble, []float64{
				5, 0, 0, 0,
				0, 12, 0, 0,
				0, 0, 21, 0,
				0, 0, 0, 32,
			})
		})
	})

	Convey("Given dense 3x5 and diag 5x5 matrixes", t, func() {
		m1 := A([]int{3, 5},
			1, 2, 3, 4, 5,
			6, 7, 8, 9, 10,
			11, 12, 13, 14, 15).M()
		m2 := Diag(2, 4, 6, 8, 10)
		Convey("MProd is correct", func() {
			p := MProd(m1, m2)
			So(p.Shape(), ShouldResemble, []int{3, 5})
			So(p.Array(), ShouldResemble, []float64{
				2, 8, 18, 32, 50,
				12, 28, 48, 72, 100,
				22, 48, 78, 112, 150,
			})
		})
	})

	Convey("Given diag 3x3 and dense 3x5 matrixes", t, func() {
		m1 := Diag(3, 6, 9)
		m2 := A([]int{3, 5},
			1, 2, 3, 4, 5,
			6, 7, 8, 9, 10,
			11, 12, 13, 14, 15).M()
		Convey("MProd is correct", func() {
			p := MProd(m1, m2)
			So(p.Shape(), ShouldResemble, []int{3, 5})
			So(p.Array(), ShouldResemble, []float64{
				3, 6, 9, 12, 15,
				36, 42, 48, 54, 60,
				99, 108, 117, 126, 135,
			})
		})
	})

	Convey("Given sparse coo 3x5 and diag 5x5 matrixes", t, func() {
		m1 := SparseCoo(3, 5)
		for row := 0; row < 3; row++ {
			for col := 0; col < 5; col++ {
				m1.ItemSet(float64(row*5+col), row, col)
			}
		}
		m2 := Diag(2, 4, 6, 8, 10)
		Convey("MProd is correct", func() {
			p := MProd(m1, m2)
			So(p.Shape(), ShouldResemble, []int{3, 5})
			So(p.Array(), ShouldResemble, []float64{
				0, 4, 12, 24, 40,
				10, 24, 42, 64, 90,
				20, 44, 72, 104, 140,
			})
		})
	})

	Convey("Given diag 3x3 and sparse coo 3x5 matrixes", t, func() {
		m1 := Diag(3, 6, 9)
		m2 := SparseCoo(3, 5)
		for row := 0; row < 3; row++ {
			for col := 0; col < 5; col++ {
				m2.ItemSet(float64(row*5+col), row, col)
			}
		}
		Convey("MProd is correct", func() {
			p := MProd(m1, m2)
			So(p.Shape(), ShouldResemble, []int{3, 5})
			So(p.Array(), ShouldResemble, []float64{
				0, 3, 6, 9, 12,
				30, 36, 42, 48, 54,
				90, 99, 108, 117, 126,
			})
		})
	})
}

func TestProd(t *testing.T) {
	Convey("Prod panics when given arrays of conflicting shapes", t, func() {
		So(func() { Prod(Rand(5), Rand(6)) }, ShouldPanic)
		So(func() { Prod(Rand(5), Rand(5, 2)) }, ShouldPanic)
	})

	Convey("Given two each dense, sparse coo, and sparse diag arrays", t, func() {
		d1 := A([]int{3, 3},
			1, 2, 3,
			4, 5, 6,
			7, 8, 9)
		d2 := d1.ItemProd(10)
		c1 := SparseCoo(3, 3)
		c1.ItemSet(100, 0, 0)
		c1.ItemSet(102, 0, 2)
		c1.ItemSet(101, 1, 0)
		c2 := SparseCoo(3, 3)
		c2.ItemSet(200, 0, 1)
		c2.ItemSet(202, 0, 2)
		c2.ItemSet(201, 1, 0)
		g1 := Diag(1, 2, 3)
		g2 := Diag(4, 5, 6)

		Convey("Prod(dense, dense) gives the correct dense array", func() {
			a := Prod(d1, d2)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, DenseArray)
			So(a.Array(), ShouldResemble, []float64{
				10, 40, 90,
				160, 250, 360,
				490, 640, 810,
			})
		})

		Convey("Prod(dense, coo) gives the correct dense array", func() {
			a := Prod(d1, c1)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, DenseArray)
			So(a.Array(), ShouldResemble, []float64{
				100, 0, 306,
				404, 0, 0,
				0, 0, 0,
			})
		})

		Convey("Prod(coo, dense) gives the correct coo array", func() {
			a := Prod(c1, d1)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, SparseCooMatrix)
			So(a.Array(), ShouldResemble, []float64{
				100, 0, 306,
				404, 0, 0,
				0, 0, 0,
			})
		})

		Convey("Prod(dense, diag) gives the correct dense array", func() {
			a := Prod(d1, g1)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, DenseArray)
			So(a.Array(), ShouldResemble, []float64{
				1, 0, 0,
				0, 10, 0,
				0, 0, 27,
			})
		})

		Convey("Prod(diag, dense) gives the correct diag array", func() {
			a := Prod(g1, d1)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, SparseDiagMatrix)
			So(a.Array(), ShouldResemble, []float64{
				1, 0, 0,
				0, 10, 0,
				0, 0, 27,
			})
		})

		Convey("Prod(coo, coo) gives the correct coo array", func() {
			a := Prod(c1, c2)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, SparseCooMatrix)
			So(a.Array(), ShouldResemble, []float64{
				0, 0, 102 * 202,
				101 * 201, 0, 0,
				0, 0, 0,
			})
		})

		Convey("Prod(coo, diag) gives the correct coo array", func() {
			a := Prod(c1, g1)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, SparseCooMatrix)
			So(a.Array(), ShouldResemble, []float64{
				100, 0, 0,
				0, 0, 0,
				0, 0, 0,
			})
		})

		Convey("Prod(diag, coo) gives the correct diag array", func() {
			a := Prod(g1, c1)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, SparseDiagMatrix)
			So(a.Array(), ShouldResemble, []float64{
				100, 0, 0,
				0, 0, 0,
				0, 0, 0,
			})
		})

		Convey("Prod(diag, diag) gives the correct diag array", func() {
			a := Prod(g1, g2)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, SparseDiagMatrix)
			So(a.Array(), ShouldResemble, []float64{
				4, 0, 0,
				0, 10, 0,
				0, 0, 18,
			})
		})
	})
}

func TestMaxMin(t *testing.T) {
	Convey("Given a dense array", t, func() {
		a := A([]int{3, 4},
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12)

		Convey("Max finds the largest element", func() {
			So(Max(a), ShouldEqual, 12)
		})

		Convey("Min finds the smallest element", func() {
			So(Min(a), ShouldEqual, 1)
		})
	})

	Convey("Given a sparse coo array with mixed sign values", t, func() {
		a := SparseCoo(3, 4)
		a.ItemSet(-1.5, 0, 0)
		a.ItemSet(2.5, 2, 0)
		a.ItemSet(3.5, 2, 3)

		Convey("Max finds the largest element", func() {
			So(Max(a), ShouldEqual, 3.5)
		})

		Convey("Min finds the smallest element", func() {
			So(Min(a), ShouldEqual, -1.5)
		})
	})

	Convey("Given a sparse coo array with all positive values", t, func() {
		a := SparseCoo(3, 4)
		a.ItemSet(1.5, 0, 0)
		a.ItemSet(2.5, 2, 0)
		a.ItemSet(3.5, 2, 3)

		Convey("Max finds the largest element", func() {
			So(Max(a), ShouldEqual, 3.5)
		})

		Convey("Min finds the smallest element", func() {
			So(Min(a), ShouldEqual, 0)
		})
	})

	Convey("Given a sparse coo array with all negative values", t, func() {
		a := SparseCoo(3, 4)
		a.ItemSet(-1.5, 0, 0)
		a.ItemSet(-2.5, 2, 0)
		a.ItemSet(-3.5, 2, 3)

		Convey("Max finds the largest element", func() {
			So(Max(a), ShouldEqual, 0)
		})

		Convey("Min finds the smallest element", func() {
			So(Min(a), ShouldEqual, -3.5)
		})
	})

	Convey("Given a sparse diag array with mixed sign values", t, func() {
		a := Diag(-1.5, 2.5, 3.5)

		Convey("Max finds the largest element", func() {
			So(Max(a), ShouldEqual, 3.5)
		})

		Convey("Min finds the smallest element", func() {
			So(Min(a), ShouldEqual, -1.5)
		})
	})

	Convey("Given a sparse diag array with all positive values", t, func() {
		a := Diag(1.5, 2.5, 3.5)

		Convey("Max finds the largest element", func() {
			So(Max(a), ShouldEqual, 3.5)
		})

		Convey("Min finds the smallest element", func() {
			So(Min(a), ShouldEqual, 0)
		})
	})

	Convey("Given a sparse diag array with all negative values", t, func() {
		a := Diag(-1.5, -2.5, -3.5)

		Convey("Max finds the largest element", func() {
			So(Max(a), ShouldEqual, 0)
		})

		Convey("Min finds the smallest element", func() {
			So(Min(a), ShouldEqual, -3.5)
		})
	})
}

func TestRavel(t *testing.T) {
	Convey("Given a dense array", t, func() {
		a := A([]int{3, 4},
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12)

		Convey("Ravel is correct", func() {
			b := Ravel(a)
			So(b.Shape(), ShouldResemble, []int{12})
			So(b.Array(), ShouldResemble, []float64{
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 10, 11, 12})
		})
	})

	Convey("Given a sparse coo array", t, func() {
		a := SparseCoo(3, 4)
		a.ItemSet(-1.5, 0, 0)
		a.ItemSet(-2.5, 2, 0)
		a.ItemSet(-3.5, 2, 3)

		Convey("Ravel is correct", func() {
			b := Ravel(a)
			So(b.Shape(), ShouldResemble, []int{12})
			So(b.Array(), ShouldResemble, []float64{
				-1.5, 0, 0, 0,
				0, 0, 0, 0,
				-2.5, 0, 0, -3.5})
		})
	})

	Convey("Given a sparse diag array", t, func() {
		a := Diag(1, 2, 3, 4)

		Convey("Ravel is correct", func() {
			b := Ravel(a)
			So(b.Shape(), ShouldResemble, []int{16})
			So(b.Array(), ShouldResemble, []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
				0, 0, 0, 4})
		})
	})
}

func TestSub(t *testing.T) {
	Convey("Sub panics when given arrays of conflicting shapes", t, func() {
		So(func() { Sub(Rand(5), Rand(6)) }, ShouldPanic)
		So(func() { Sub(Rand(5), Rand(5, 2)) }, ShouldPanic)
	})

	Convey("Given two each dense, sparse coo, and sparse diag arrays", t, func() {
		d1 := A([]int{3, 3},
			1, 2, 3,
			4, 5, 6,
			7, 8, 9)
		d2 := d1.ItemProd(10)
		c1 := SparseCoo(3, 3)
		c1.ItemSet(100, 0, 0)
		c1.ItemSet(102, 0, 2)
		c1.ItemSet(101, 1, 0)
		c2 := SparseCoo(3, 3)
		c2.ItemSet(200, 0, 1)
		c2.ItemSet(202, 0, 2)
		c2.ItemSet(201, 1, 0)
		g1 := Diag(1, 2, 3)
		g2 := Diag(4, 5, 6)

		Convey("Sub(dense, dense) gives the correct dense array", func() {
			a := Sub(d1, d2)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, DenseArray)
			So(a.Array(), ShouldResemble, []float64{
				-9, -18, -27,
				-36, -45, -54,
				-63, -72, -81,
			})
		})

		Convey("Sub(dense, coo) gives the correct dense array", func() {
			a := Sub(d1, c1)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, DenseArray)
			So(a.Array(), ShouldResemble, []float64{
				-99, 2, -99,
				-97, 5, 6,
				7, 8, 9,
			})
		})

		Convey("Sub(coo, dense) gives the correct dense array", func() {
			a := Sub(c1, d1)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, DenseArray)
			So(a.Array(), ShouldResemble, []float64{
				99, -2, 99,
				97, -5, -6,
				-7, -8, -9,
			})
		})

		Convey("Sub(dense, diag) gives the correct dense array", func() {
			a := Sub(d1, g1)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, DenseArray)
			So(a.Array(), ShouldResemble, []float64{
				0, 2, 3,
				4, 3, 6,
				7, 8, 6,
			})
		})

		Convey("Sub(diag, dense) gives the correct dense array", func() {
			a := Sub(g1, d1)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, DenseArray)
			So(a.Array(), ShouldResemble, []float64{
				0, -2, -3,
				-4, -3, -6,
				-7, -8, -6,
			})
		})

		Convey("Sub(coo, coo) gives the correct coo array", func() {
			a := Sub(c1, c2)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, SparseCooMatrix)
			So(a.Array(), ShouldResemble, []float64{
				100, -200, -100,
				-100, 0, 0,
				0, 0, 0,
			})
		})

		Convey("Sub(coo, diag) gives the correct coo array", func() {
			a := Sub(c1, g1)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, SparseCooMatrix)
			So(a.Array(), ShouldResemble, []float64{
				99, 0, 102,
				101, -2, 0,
				0, 0, -3,
			})
		})

		Convey("Sub(diag, coo) gives the correct coo array", func() {
			a := Sub(g1, c1)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, SparseCooMatrix)
			So(a.Array(), ShouldResemble, []float64{
				-99, 0, -102,
				-101, 2, 0,
				0, 0, 3,
			})
		})

		Convey("Sub(diag, diag) gives the correct diag array", func() {
			a := Sub(g1, g2)
			So(a.Shape(), ShouldResemble, []int{3, 3})
			So(a.Sparsity(), ShouldEqual, SparseDiagMatrix)
			So(a.Array(), ShouldResemble, []float64{
				-3, 0, 0,
				0, -3, 0,
				0, 0, -3,
			})
		})
	})
}

func BenchmarkMProdDenseDense(b *testing.B) {
	l := Rand(5, 5).M()
	r := Rand(5, 5).M()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l.MProd(r)
	}
}

func BenchmarkMProdDenseCoo(b *testing.B) {
	l := Rand(5, 5).M()
	r := SparseRand(5, 5, 0.5)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l.MProd(r)
	}
}

func BenchmarkMProdDenseDiag(b *testing.B) {
	l := Rand(5, 5).M()
	r := Diag(1, 2, 3, 4, 5)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l.MProd(r)
	}
}

func BenchmarkMProdCooDense(b *testing.B) {
	l := SparseRand(5, 5, 0.5)
	r := Rand(5, 5).M()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l.MProd(r)
	}
}

func BenchmarkMProdCooCoo(b *testing.B) {
	l := SparseRand(5, 5, 0.5)
	r := SparseRand(5, 5, 0.5)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l.MProd(r)
	}
}

func BenchmarkMProdCooDiag(b *testing.B) {
	l := SparseRand(5, 5, 0.5)
	r := Diag(1, 2, 3, 4, 5)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l.MProd(r)
	}
}

func BenchmarkMProdDiagDense(b *testing.B) {
	l := Diag(1, 2, 3, 4, 5)
	r := Rand(5, 5).M()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l.MProd(r)
	}
}

func BenchmarkMProdDiagCoo(b *testing.B) {
	l := Diag(1, 2, 3, 4, 5)
	r := SparseRand(5, 5, 0.5)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l.MProd(r)
	}
}

func BenchmarkMProdDiagDiag(b *testing.B) {
	l := Diag(1, 2, 3, 4, 5)
	r := Diag(5, 4, 3, 2, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l.MProd(r)
	}
}
