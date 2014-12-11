package matrix

import (
	. "github.com/smartystreets/goconvey/convey"
	"math"
	"testing"
)

func init() {
	InitDefaultBlas()
}

func TestConversion(t *testing.T) {
	Convey("Given a Matrix", t, func() {
		m := Rand(3, 5).M()
		Convey("Converting it toMat64 works", func() {
			m2 := toMat64(m)
			r, c := m2.Dims()
			So(r, ShouldEqual, 3)
			So(c, ShouldEqual, 5)
			for i0 := 0; i0 < 3; i0++ {
				for i1 := 0; i1 < 5; i1++ {
					So(m2.At(i0, i1), ShouldEqual, m.Item(i0, i1))
				}
			}

			Convey("Converting it back to a matrix works", func() {
				m3 := toMatrix(m2)
				So(m3.Shape(), ShouldResemble, []int{3, 5})
				for i0 := 0; i0 < 3; i0++ {
					for i1 := 0; i1 < 5; i1++ {
						So(m3.Item(i0, i1), ShouldEqual, m.Item(i0, i1))
					}
				}
			})
		})
	})
}

func TestDiag(t *testing.T) {
	Convey("Given a diagonal array with 3 elements", t, func() {
		array := Diag(1, 2, 3)

		Convey("Shape() is (3, 3)", func() {
			So(array.Shape(), ShouldResemble, []int{3, 3})
		})

		Convey("Size() is 9", func() {
			So(array.Size(), ShouldEqual, 9)
		})

		Convey("Only diagonal values are set; others are zero", func() {
			So(array.Array(), ShouldResemble, []float64{
				1, 0, 0,
				0, 2, 0,
				0, 0, 3,
			})
		})
	})
}

func TestEye(t *testing.T) {
	Convey("Given an identity array with shape 3, 3", t, func() {
		array := Eye(3)

		Convey("Shape() is (3, 3)", func() {
			So(array.Shape(), ShouldResemble, []int{3, 3})
		})

		Convey("Size() is 9", func() {
			So(array.Size(), ShouldEqual, 9)
		})

		Convey("Only diagonal values are one; others are zero", func() {
			for i0 := 0; i0 < 3; i0++ {
				for i1 := 0; i1 < 3; i1++ {
					if i0 == i1 {
						So(array.Item(i0, i1), ShouldEqual, 1)
					} else {
						So(array.Item(i0, i1), ShouldEqual, 0)
					}
				}
			}
		})
	})
}

func TestM(t *testing.T) {
	Convey("M() panics when data doesn't match dimensions", t, func() {
		So(func() { M(2, 3, 1, 2, 3, 4, 5) }, ShouldPanic)
		So(func() { M(2, 3, 1, 2, 3, 4, 5, 6, 7) }, ShouldPanic)
	})

	Convey("Given a vector created with M", t, func() {
		m := M(5, 1, 1, 2, 3, 4, 5)
		Convey("Shape() is 5,1", func() {
			So(m.Shape(), ShouldResemble, []int{5, 1})
		})
		Convey("Size() is 5", func() {
			So(m.Size(), ShouldResemble, 5)
		})
		Convey("The data is correct", func() {
			So(m.Array(), ShouldResemble, []float64{1, 2, 3, 4, 5})
		})
	})

	Convey("Given a 2D matrix created with M", t, func() {
		m := M(2, 3, 1, 2, 3, 4, 5, 6)
		Convey("Shape() is 2, 3", func() {
			So(m.Shape(), ShouldResemble, []int{2, 3})
		})
		Convey("Size() is 6", func() {
			So(m.Size(), ShouldResemble, 6)
		})
		Convey("The data is correct", func() {
			So(m.Array(), ShouldResemble, []float64{
				1, 2, 3,
				4, 5, 6,
			})
		})
	})
}

func TestSparseCoo(t *testing.T) {
	Convey("Given a 2x3 SparseCoo matrix", t, func() {
		m := SparseCoo(2, 3)

		Convey("Shape() is 2, 3", func() {
			So(m.Shape(), ShouldResemble, []int{2, 3})
		})

		Convey("Size() is 6", func() {
			So(m.Size(), ShouldResemble, 6)
		})

		Convey("The data is correct", func() {
			So(m.Array(), ShouldResemble, []float64{
				0, 0, 0,
				0, 0, 0,
			})
		})
	})
}

func TestSparseDiag(t *testing.T) {
	Convey("SparseDiag panics when given too many diagonal elements", t, func() {
		So(func() { SparseDiag(2, 3, 1, 2, 3, 4, 5, 6, 7) }, ShouldPanic)
	})

	Convey("Given a 2x3 SparseDiag matrix with diagonal elements set", t, func() {
		m := SparseDiag(2, 3, 1, 2)

		Convey("Shape() is 2, 3", func() {
			So(m.Shape(), ShouldResemble, []int{2, 3})
		})

		Convey("Size() is 6", func() {
			So(m.Size(), ShouldResemble, 6)
		})

		Convey("The data is correct", func() {
			So(m.Array(), ShouldResemble, []float64{
				1, 0, 0,
				0, 2, 0,
			})
		})
	})

	Convey("Given a 3x2 SparseDiag matrix with diagonal elements set", t, func() {
		m := SparseDiag(3, 2, 1, 2)

		Convey("Shape() is 3, 2", func() {
			So(m.Shape(), ShouldResemble, []int{3, 2})
		})

		Convey("Size() is 6", func() {
			So(m.Size(), ShouldResemble, 6)
		})

		Convey("The data is correct", func() {
			So(m.Array(), ShouldResemble, []float64{
				1, 0,
				0, 2,
				0, 0,
			})
		})
	})

	Convey("Given a 3x3 SparseDiag matrix with no diagonal elements set", t, func() {
		m := SparseDiag(3, 3)

		Convey("Shape() is 3, 3", func() {
			So(m.Shape(), ShouldResemble, []int{3, 3})
		})

		Convey("Size() is 9", func() {
			So(m.Size(), ShouldResemble, 9)
		})

		Convey("The data is correct", func() {
			So(m.Array(), ShouldResemble, []float64{
				0, 0, 0,
				0, 0, 0,
				0, 0, 0,
			})
		})
	})

	Convey("Given a 3x3 SparseDiag matrix with some diagonal elements set", t, func() {
		m := SparseDiag(3, 3, 1, 2)

		Convey("Shape() is 3, 3", func() {
			So(m.Shape(), ShouldResemble, []int{3, 3})
		})

		Convey("Size() is 9", func() {
			So(m.Size(), ShouldResemble, 9)
		})

		Convey("The data is correct", func() {
			So(m.Array(), ShouldResemble, []float64{
				1, 0, 0,
				0, 2, 0,
				0, 0, 0,
			})
		})
	})

	Convey("Given a 3x3 SparseDiag matrix with all diagonal elements set", t, func() {
		m := SparseDiag(3, 3, 1, 2, 3)

		Convey("Shape() is 3, 3", func() {
			So(m.Shape(), ShouldResemble, []int{3, 3})
		})

		Convey("Size() is 9", func() {
			So(m.Size(), ShouldResemble, 9)
		})

		Convey("The data is correct", func() {
			So(m.Array(), ShouldResemble, []float64{
				1, 0, 0,
				0, 2, 0,
				0, 0, 3,
			})
		})
	})
}

func TestSparseRand(t *testing.T) {
	Convey("SparseRand panics with an invalid density", t, func() {
		So(func() { SparseRand(2, 3, -1) }, ShouldPanic)
		So(func() { SparseRand(2, 3, 1) }, ShouldPanic)
	})

	Convey("Given a sparse random array with shape 2, 3 and density 0.5", t, func() {
		array := SparseRand(2, 3, 0.5)

		Convey("Shape() is (2, 3)", func() {
			So(array.Shape(), ShouldResemble, []int{2, 3})
		})

		Convey("Size() is 6", func() {
			So(array.Size(), ShouldEqual, 6)
		})

		Convey("Half the values are filled", func() {
			So(array.CountNonzero(), ShouldEqual, 3)
		})
	})
}

func TestSparseRandN(t *testing.T) {
	Convey("SparseRandN panics with an invalid density", t, func() {
		So(func() { SparseRandN(2, 3, -1) }, ShouldPanic)
		So(func() { SparseRandN(2, 3, 1) }, ShouldPanic)
	})

	Convey("Given a sparse random array with shape 2, 3 and density 0.5", t, func() {
		array := SparseRandN(2, 3, 0.5)

		Convey("Shape() is (2, 3)", func() {
			So(array.Shape(), ShouldResemble, []int{2, 3})
		})

		Convey("Size() is 6", func() {
			So(array.Size(), ShouldEqual, 6)
		})

		Convey("Half the values are filled", func() {
			So(array.CountNonzero(), ShouldEqual, 3)
		})
	})
}

func TestInverse(t *testing.T) {
	Convey("Given an invertible square matrix", t, func() {
		m := A2(
			[]float64{4, 7},
			[]float64{2, 6},
		).M()

		Convey("When I take the inverse", func() {
			mi, err := Inverse(m)
			So(err, ShouldBeNil)

			Convey("The inverse is correct", func() {
				So(mi.Shape(), ShouldResemble, []int{2, 2})
				So(mi.Item(0, 0), ShouldBeBetween, 0.6-Eps, 0.6+Eps)
				So(mi.Item(0, 1), ShouldBeBetween, -0.7-Eps, 0.7+Eps)
				So(mi.Item(1, 0), ShouldBeBetween, -0.2-Eps, -0.2+Eps)
				So(mi.Item(1, 1), ShouldBeBetween, 0.4-Eps, 0.4+Eps)
			})

			Convey("The inverse gives us back I", func() {
				i := m.MProd(mi)
				So(i.Shape(), ShouldResemble, []int{2, 2})
				So(i.Item(0, 0), ShouldBeBetween, 1-Eps, 1+Eps)
				So(i.Item(0, 1), ShouldEqual, 0)
				So(i.Item(1, 0), ShouldEqual, 0)
				So(i.Item(1, 1), ShouldBeBetween, 1-Eps, 1+Eps)
			})
		})
	})

	Convey("Given a non-invertible matrix", t, func() {
		m := A([]int{2, 2},
			0, 0,
			1, 1).M()

		Convey("Inverse returns an error", func() {
			mi, err := Inverse(m)
			So(mi, ShouldBeNil)
			So(err, ShouldNotBeNil)
		})
	})
}

func TestLDivide(t *testing.T) {
	Convey("Given a simple division problem", t, func() {
		a := M(3, 3,
			8, 1, 6,
			3, 5, 7,
			4, 9, 2)
		b := M(3, 1,
			15,
			15,
			15)

		Convey("When I solve the system", func() {
			x := LDivide(a, b)

			Convey("I get the correct solution", func() {
				So(x.Shape(), ShouldResemble, []int{3, 1})
				So(x.Item(0, 0), ShouldBeBetween, 1-Eps, 1+Eps)
				So(x.Item(1, 0), ShouldBeBetween, 1-Eps, 1+Eps)
				So(x.Item(2, 0), ShouldBeBetween, 1-Eps, 1+Eps)
			})

			Convey("The product ax = b is true", func() {
				b2 := a.MProd(x)
				So(b2.Shape(), ShouldResemble, []int{3, 1})
				So(b2.Item(0, 0), ShouldBeBetween, 15-Eps, 15+Eps)
				So(b2.Item(1, 0), ShouldBeBetween, 15-Eps, 15+Eps)
				So(b2.Item(2, 0), ShouldBeBetween, 15-Eps, 15+Eps)
			})
		})
	})

	Convey("Given a singular division problem", t, func() {
		a := M(3, 3,
			0, 0, 0,
			3, 5, 7,
			4, 9, 2)
		b := M(3, 1,
			15,
			15,
			15)

		Convey("When I solve the system", func() {
			x := LDivide(a, b)

			Convey("I get NaN", func() {
				So(x.Shape(), ShouldResemble, []int{3, 1})
				So(x.AllF(math.IsNaN), ShouldBeTrue)
			})
		})
	})
}

func TestNorm(t *testing.T) {
	Convey("Given a 3x3 matrix", t, func() {
		m := M(3, 3,
			1, 2, 3,
			4, 5, 6,
			7, 8, 9)

		Convey("The 1-norm is correct", func() {
			So(Norm(m, 1), ShouldEqual, 18)
		})

		Convey("The 2-norm is correct", func() {
			So(Norm(m, 2), ShouldEqual, 16.84810335261421)
		})

		Convey("The inf-norm is correct", func() {
			So(Norm(m, math.Inf(1)), ShouldEqual, 24)
		})
	})
}

func TestSolve(t *testing.T) {
	Convey("Given a simple division problem", t, func() {
		a := M(3, 3,
			8, 1, 6,
			3, 5, 7,
			4, 9, 2)
		b := M(3, 1,
			15,
			15,
			15)

		Convey("When I solve the system", func() {
			x := Solve(a, b)

			Convey("I get the correct solution", func() {
				So(x.Shape(), ShouldResemble, []int{3, 1})
				So(x.Item(0, 0), ShouldBeBetween, 1-Eps, 1+Eps)
				So(x.Item(1, 0), ShouldBeBetween, 1-Eps, 1+Eps)
				So(x.Item(2, 0), ShouldBeBetween, 1-Eps, 1+Eps)
			})

			Convey("The product ax = b is true", func() {
				b2 := a.MProd(x)
				So(b2.Shape(), ShouldResemble, []int{3, 1})
				So(b2.Item(0, 0), ShouldBeBetween, 15-Eps, 15+Eps)
				So(b2.Item(1, 0), ShouldBeBetween, 15-Eps, 15+Eps)
				So(b2.Item(2, 0), ShouldBeBetween, 15-Eps, 15+Eps)
			})
		})
	})

	Convey("Given a singular division problem", t, func() {
		a := M(3, 3,
			0, 0, 0,
			3, 5, 7,
			4, 9, 2)
		b := M(3, 1,
			15,
			15,
			15)

		Convey("When I solve the system", func() {
			x := Solve(a, b)

			Convey("I get NaN", func() {
				So(x.Shape(), ShouldResemble, []int{3, 1})
				So(x.AllF(math.IsNaN), ShouldBeTrue)
			})
		})
	})
}
