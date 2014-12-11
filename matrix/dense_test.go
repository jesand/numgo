package matrix

import (
	. "github.com/smartystreets/goconvey/convey"
	"math"
	"testing"
)

// The smallest floating point difference permissible for near-equality
const Eps float64 = 1e-9

func TestDenseAddDivMulSub(t *testing.T) {
	Convey("Given two dense matrices", t, func() {
		d1 := A([]int{3, 4},
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12)
		d2 := A([]int{3, 4},
			11, 12, 13, 14,
			15, 16, 17, 18,
			19, 20, 21, 22)

		Convey("Add works", func() {
			a := d1.Add(d2)
			So(a.Shape(), ShouldResemble, []int{3, 4})
			So(a.Array(), ShouldResemble, []float64{
				12, 14, 16, 18,
				20, 22, 24, 26,
				28, 30, 32, 34,
			})
		})

		Convey("Div works", func() {
			a := d2.Div(d1)
			So(a.Shape(), ShouldResemble, []int{3, 4})
			So(a.Array(), ShouldResemble, []float64{
				11. / 1, 12. / 2, 13. / 3, 14. / 4,
				15. / 5, 16. / 6, 17. / 7, 18. / 8,
				19. / 9, 20. / 10, 21. / 11, 22. / 12,
			})
		})

		Convey("Prod works", func() {
			a := d2.Prod(d1)
			So(a.Shape(), ShouldResemble, []int{3, 4})
			So(a.Array(), ShouldResemble, []float64{
				11 * 1, 12 * 2, 13 * 3, 14 * 4,
				15 * 5, 16 * 6, 17 * 7, 18 * 8,
				19 * 9, 20 * 10, 21 * 11, 22 * 12,
			})
		})

		Convey("Sub works", func() {
			a := d2.Sub(d1)
			So(a.Shape(), ShouldResemble, []int{3, 4})
			So(a.Array(), ShouldResemble, []float64{
				10, 10, 10, 10,
				10, 10, 10, 10,
				10, 10, 10, 10,
			})
		})
	})
}

func TestDenseAllAny(t *testing.T) {
	Convey("Given full, half-full, and empty arrays", t, func() {
		full := Rand(3, 4)
		half := SparseRand(3, 4, .5).Dense()
		empty := Zeros(3, 4)

		Convey("All() is correct", func() {
			So(full.All(), ShouldBeTrue)
			So(half.All(), ShouldBeFalse)
			So(empty.All(), ShouldBeFalse)
		})

		Convey("Any() is correct", func() {
			So(full.Any(), ShouldBeTrue)
			So(half.Any(), ShouldBeTrue)
			So(empty.Any(), ShouldBeFalse)
		})
	})

	Convey("Given pos, neg, and mixed arrays", t, func() {
		pos := Ones(3, 4)
		neg := WithValue(-1, 3, 4)
		mixed := A([]int{3, 4},
			1, 1, 1, 1,
			-1, -1, -1, -1,
			1, 1, 1, 1)
		f := func(v float64) bool { return v > 0 }
		f2 := func(v1, v2 float64) bool { return v1 > v2 }

		Convey("AllF() is correct", func() {
			So(pos.AllF(f), ShouldBeTrue)
			So(mixed.AllF(f), ShouldBeFalse)
			So(neg.AllF(f), ShouldBeFalse)
		})

		Convey("AnyF() is correct", func() {
			So(pos.AnyF(f), ShouldBeTrue)
			So(mixed.AnyF(f), ShouldBeTrue)
			So(neg.AnyF(f), ShouldBeFalse)
		})

		Convey("AllF2() is correct", func() {
			So(pos.AllF2(f2, neg), ShouldBeTrue)
			So(mixed.AllF2(f2, neg), ShouldBeFalse)
			So(neg.AllF2(f2, neg), ShouldBeFalse)
		})

		Convey("AnyF2() is correct", func() {
			So(pos.AnyF2(f2, neg), ShouldBeTrue)
			So(mixed.AnyF2(f2, neg), ShouldBeTrue)
			So(neg.AnyF2(f2, neg), ShouldBeFalse)
		})
	})
}

func TestDenseApply(t *testing.T) {
	Convey("Apply works", t, func() {
		a := A([]int{3, 5},
			1, 2, 3, 4, 5,
			6, 7, 8, 9, 10,
			11, 12, 13, 14, 15,
		)
		a2 := a.Apply(func(v float64) float64 { return 2 * v })
		So(a2.Shape(), ShouldResemble, []int{3, 5})
		So(a2.Array(), ShouldResemble, []float64{
			2, 4, 6, 8, 10,
			12, 14, 16, 18, 20,
			22, 24, 26, 28, 30,
		})
	})
}

func TestDenseConversion(t *testing.T) {
	Convey("Given an array", t, func() {
		a := M(3, 4,
			1, 2, 0, 0,
			0, 3, 4, 0,
			0, 0, 5, 6)

		Convey("Conversion to Dense works", func() {
			b := a.Dense()
			So(b.Dense().Shape(), ShouldResemble, []int{3, 4})
			So(b.Dense().Array(), ShouldResemble, []float64{
				1, 2, 0, 0,
				0, 3, 4, 0,
				0, 0, 5, 6,
			})
		})

		Convey("Conversion of transpose to Dense works", func() {
			b := a.T().Dense()
			So(b.Dense().Shape(), ShouldResemble, []int{4, 3})
			So(b.Dense().Array(), ShouldResemble, []float64{
				1, 0, 0,
				2, 3, 0,
				0, 4, 5,
				0, 0, 6,
			})
		})

		Convey("Conversion to sparse coo works", func() {
			b := a.SparseCoo()
			So(b.Dense().Shape(), ShouldResemble, []int{3, 4})
			So(b.Dense().Array(), ShouldResemble, []float64{
				1, 2, 0, 0,
				0, 3, 4, 0,
				0, 0, 5, 6,
			})
		})

		Convey("Conversion of transpose to sparse coo works", func() {
			b := a.T().SparseCoo()
			So(b.Dense().Shape(), ShouldResemble, []int{4, 3})
			So(b.Dense().Array(), ShouldResemble, []float64{
				1, 0, 0,
				2, 3, 0,
				0, 4, 5,
				0, 0, 6,
			})
		})

		Convey("Conversion to diag panics if matrix is not diagonal", func() {
			So(func() { a.SparseDiag() }, ShouldPanic)
		})

		Convey("Conversion to diag works", func() {
			a.ItemSet(0, 0, 1)
			a.ItemSet(0, 1, 2)
			a.ItemSet(0, 2, 3)
			b := a.SparseDiag()
			So(b.Dense().Shape(), ShouldResemble, []int{3, 4})
			So(b.Dense().Array(), ShouldResemble, []float64{
				1, 0, 0, 0,
				0, 3, 0, 0,
				0, 0, 5, 0,
			})
		})

		Convey("Conversion of transpose to diag works", func() {
			a.ItemSet(0, 0, 1)
			a.ItemSet(0, 1, 2)
			a.ItemSet(0, 2, 3)
			b := a.T().SparseDiag()
			So(b.Dense().Shape(), ShouldResemble, []int{4, 3})
			So(b.Dense().Array(), ShouldResemble, []float64{
				1, 0, 0,
				0, 3, 0,
				0, 0, 5,
				0, 0, 0,
			})
		})
	})
}

func TestDenseColColSetCols(t *testing.T) {
	Convey("Given a dense array", t, func() {
		a := A([]int{3, 4},
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
		).M()

		Convey("Col panics with invalid input", func() {
			So(func() { a.Col(-1) }, ShouldPanic)
			So(func() { a.Col(4) }, ShouldPanic)
		})

		Convey("Col works", func() {
			So(a.Col(0), ShouldResemble, []float64{1, 5, 9})
			So(a.Col(1), ShouldResemble, []float64{2, 6, 10})
			So(a.Col(2), ShouldResemble, []float64{3, 7, 11})
			So(a.Col(3), ShouldResemble, []float64{4, 8, 12})
		})

		Convey("ColSet panics with invalid input", func() {
			So(func() { a.ColSet(-1, []float64{0, 1, 2}) }, ShouldPanic)
			So(func() { a.ColSet(4, []float64{0, 1, 2}) }, ShouldPanic)
			So(func() { a.ColSet(0, []float64{0, 1}) }, ShouldPanic)
			So(func() { a.ColSet(0, []float64{0, 1, 2, 3}) }, ShouldPanic)
		})

		Convey("ColSet works", func() {
			a.ColSet(0, []float64{0, 1, 2})
			So(a.Array(), ShouldResemble, []float64{
				0, 2, 3, 4,
				1, 6, 7, 8,
				2, 10, 11, 12,
			})
			a.ColSet(1, []float64{0, 1, 2})
			So(a.Array(), ShouldResemble, []float64{
				0, 0, 3, 4,
				1, 1, 7, 8,
				2, 2, 11, 12,
			})
			a.ColSet(2, []float64{0, 1, 2})
			So(a.Array(), ShouldResemble, []float64{
				0, 0, 0, 4,
				1, 1, 1, 8,
				2, 2, 2, 12,
			})
			a.ColSet(3, []float64{0, 1, 2})
			So(a.Array(), ShouldResemble, []float64{
				0, 0, 0, 0,
				1, 1, 1, 1,
				2, 2, 2, 2,
			})
		})
	})
}

func TestDenseDiag(t *testing.T) {
	Convey("Given a dense array", t, func() {
		a := M(4, 3,
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
			10, 11, 12,
		)

		Convey("Diag() works", func() {
			d := a.Diag()
			So(d.Shape(), ShouldResemble, []int{3, 1})
			So(d.Array(), ShouldResemble, []float64{1, 5, 9})
		})

		Convey("Transpose Diag() works", func() {
			d := a.T().M().Diag()
			So(d.Shape(), ShouldResemble, []int{3, 1})
			So(d.Array(), ShouldResemble, []float64{1, 5, 9})
		})
	})
}

func TestDenseEqual(t *testing.T) {
	Convey("A dense array equals itself", t, func() {
		a := Rand(3, 4)
		So(a.Equal(a), ShouldBeTrue)
	})
	Convey("Different arrays are not equal", t, func() {
		a := Rand(3, 4)
		a2 := Rand(3, 4)
		So(a.Equal(a2), ShouldBeFalse)
	})
}

func TestDenseInverseNormLDivide(t *testing.T) {
	Convey("Given an invertible square matrix", t, func() {
		m := M(2, 2,
			4, 7,
			2, 6)

		Convey("When I take the inverse", func() {
			mi, err := m.Inverse()
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

	Convey("Given a 3x3 matrix", t, func() {
		m := M(3, 3,
			1, 2, 3,
			4, 5, 6,
			7, 8, 9)

		Convey("The 1-norm is correct", func() {
			So(m.Norm(1), ShouldEqual, 18)
		})

		Convey("The 2-norm is correct", func() {
			So(m.Norm(2), ShouldEqual, 16.84810335261421)
		})

		Convey("The inf-norm is correct", func() {
			So(m.Norm(math.Inf(1)), ShouldEqual, 24)
		})
	})

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
			x := a.LDivide(b)

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
}

func TestDenseItemMath(t *testing.T) {
	Convey("Given a dense array", t, func() {
		a := M(4, 3,
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
			10, 11, 12,
		)

		Convey("When I call ItemAdd", func() {
			a2 := a.ItemAdd(0.5)
			Convey("The result is correct", func() {
				So(a2.Array(), ShouldResemble, []float64{
					1.5, 2.5, 3.5,
					4.5, 5.5, 6.5,
					7.5, 8.5, 9.5,
					10.5, 11.5, 12.5,
				})
			})
		})

		Convey("When I call ItemDiv", func() {
			a2 := a.ItemDiv(2)
			Convey("The result is correct", func() {
				So(a2.Array(), ShouldResemble, []float64{
					.5, 1, 1.5,
					2, 2.5, 3,
					3.5, 4, 4.5,
					5, 5.5, 6,
				})
			})
		})

		Convey("When I call ItemProd", func() {
			a2 := a.ItemProd(2)
			Convey("The result is correct", func() {
				So(a2.Array(), ShouldResemble, []float64{
					2, 4, 6,
					8, 10, 12,
					14, 16, 18,
					20, 22, 24,
				})
			})
		})

		Convey("When I call ItemSub", func() {
			a2 := a.ItemSub(1)
			Convey("The result is correct", func() {
				So(a2.Array(), ShouldResemble, []float64{
					0, 1, 2,
					3, 4, 5,
					6, 7, 8,
					9, 10, 11,
				})
			})
		})
	})
}

func TestDenseRowRowSetRows(t *testing.T) {
	Convey("Given a dense array", t, func() {
		a := M(4, 3,
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
			10, 11, 12,
		)

		Convey("Row panics with invalid input", func() {
			So(func() { a.Row(-1) }, ShouldPanic)
			So(func() { a.Row(4) }, ShouldPanic)
		})

		Convey("Row works", func() {
			So(a.Row(0), ShouldResemble, []float64{1, 2, 3})
			So(a.Row(1), ShouldResemble, []float64{4, 5, 6})
			So(a.Row(2), ShouldResemble, []float64{7, 8, 9})
			So(a.Row(3), ShouldResemble, []float64{10, 11, 12})
		})

		Convey("RowSet panics with invalid input", func() {
			So(func() { a.RowSet(-1, []float64{0, 1, 2}) }, ShouldPanic)
			So(func() { a.RowSet(4, []float64{0, 1, 2}) }, ShouldPanic)
			So(func() { a.RowSet(0, []float64{0, 1}) }, ShouldPanic)
			So(func() { a.RowSet(0, []float64{0, 1, 2, 3}) }, ShouldPanic)
		})

		Convey("RowSet works", func() {
			a.RowSet(0, []float64{0, 1, 2})
			So(a.Array(), ShouldResemble, []float64{
				0, 1, 2,
				4, 5, 6,
				7, 8, 9,
				10, 11, 12,
			})
			a.RowSet(1, []float64{0, 1, 2})
			So(a.Array(), ShouldResemble, []float64{
				0, 1, 2,
				0, 1, 2,
				7, 8, 9,
				10, 11, 12,
			})
			a.RowSet(2, []float64{0, 1, 2})
			So(a.Array(), ShouldResemble, []float64{
				0, 1, 2,
				0, 1, 2,
				0, 1, 2,
				10, 11, 12,
			})
			a.RowSet(3, []float64{0, 1, 2})
			So(a.Array(), ShouldResemble, []float64{
				0, 1, 2,
				0, 1, 2,
				0, 1, 2,
				0, 1, 2,
			})
		})
	})
}

func TestDenseTranspose(t *testing.T) {
	Convey("Given a transposed matrix", t, func() {
		a := M(4, 3,
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
			10, 11, 12,
		)
		tr := a.T()

		Convey("The matrix statistics are correct", func() {
			So(tr.Size(), ShouldEqual, 12)
			So(tr.Shape(), ShouldResemble, []int{3, 4})
		})

		Convey("FlatItem and FlatItemSet work", func() {
			tr.FlatItemSet(-1, 7)
			So(tr.Item(1, 2), ShouldEqual, 8)
			So(tr.Item(1, 3), ShouldEqual, -1)
			So(tr.FlatItem(6), ShouldEqual, 8)
			So(tr.FlatItem(7), ShouldEqual, -1)
		})

		Convey("Item and ItemSet work", func() {
			tr.ItemSet(-1, 2, 3)
			So(tr.Item(1, 2), ShouldEqual, 8)
			So(tr.Item(2, 3), ShouldEqual, -1)
			So(tr.FlatItem(11), ShouldEqual, -1)
			So(tr.FlatItem(6), ShouldEqual, 8)
		})

		Convey("Array() is correct", func() {
			So(a.Array(), ShouldResemble, []float64{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,
				10, 11, 12,
			})
			So(tr.Array(), ShouldResemble, []float64{
				1, 4, 7, 10,
				2, 5, 8, 11,
				3, 6, 9, 12,
			})
		})

		Convey("Copy() is correct", func() {
			So(tr.Copy().Array(), ShouldResemble, []float64{
				1, 4, 7, 10,
				2, 5, 8, 11,
				3, 6, 9, 12,
			})
		})

		Convey("Ravel() is correct", func() {
			So(tr.Ravel().Array(), ShouldResemble, []float64{
				1, 4, 7, 10,
				2, 5, 8, 11,
				3, 6, 9, 12,
			})
		})
	})
}

func TestDenseVisit(t *testing.T) {
	Convey("Given a dense array", t, func() {
		a := M(4, 3,
			0, 2, 3,
			4, 0, 6,
			7, 8, 0,
			10, 11, 12,
		)

		Convey("Visit sees all items", func() {
			saw := Zeros(a.Shape()...)
			b := Zeros(a.Shape()...)
			count := 0
			a.Visit(func(pos []int, value float64) bool {
				count++
				b.ItemSet(value, pos...)
				saw.ItemSet(1, pos...)
				return true
			})
			So(count, ShouldEqual, 12)
			So(saw.CountNonzero(), ShouldEqual, 12)
			So(b.Array(), ShouldResemble, []float64{
				0, 2, 3,
				4, 0, 6,
				7, 8, 0,
				10, 11, 12,
			})
		})

		Convey("Visit stops early if f() returns false", func() {
			saw := Zeros(a.Shape()...)
			b := Zeros(a.Shape()...)
			count := 0
			a.Visit(func(pos []int, value float64) bool {
				count++
				b.ItemSet(value, pos...)
				saw.ItemSet(1, pos...)
				if saw.CountNonzero() >= 5 {
					return false
				}
				return true
			})
			So(count, ShouldEqual, 5)
			So(saw.CountNonzero(), ShouldEqual, 5)
			So(b.Array(), ShouldResemble, []float64{
				0, 2, 3,
				4, 0, 0,
				0, 0, 0,
				0, 0, 0,
			})
		})

		Convey("VisitNonzero sees just nonzero items", func() {
			saw := Zeros(a.Shape()...)
			b := Zeros(a.Shape()...)
			count := 0
			a.VisitNonzero(func(pos []int, value float64) bool {
				count++
				b.ItemSet(value, pos...)
				saw.ItemSet(1, pos...)
				return true
			})
			So(count, ShouldEqual, 9)
			So(saw.CountNonzero(), ShouldEqual, 9)
			So(b.Array(), ShouldResemble, []float64{
				0, 2, 3,
				4, 0, 6,
				7, 8, 0,
				10, 11, 12,
			})
		})

		Convey("VisitNonzero stops early if f() returns false", func() {
			saw := Zeros(a.Shape()...)
			b := Zeros(a.Shape()...)
			count := 0
			a.VisitNonzero(func(pos []int, value float64) bool {
				count++
				b.ItemSet(value, pos...)
				saw.ItemSet(1, pos...)
				if saw.CountNonzero() >= 5 {
					return false
				}
				return true
			})
			So(count, ShouldEqual, 5)
			So(saw.CountNonzero(), ShouldEqual, 5)
			So(b.Array(), ShouldResemble, []float64{
				0, 2, 3,
				4, 0, 6,
				7, 0, 0,
				0, 0, 0,
			})
		})

		Convey("T().Visit sees all items", func() {
			saw := Zeros(a.T().Shape()...)
			b := Zeros(a.T().Shape()...)
			count := 0
			a.T().Visit(func(pos []int, value float64) bool {
				count++
				b.ItemSet(value, pos...)
				saw.ItemSet(1, pos...)
				return true
			})
			So(count, ShouldEqual, 12)
			So(saw.CountNonzero(), ShouldEqual, 12)
			So(b.Array(), ShouldResemble, []float64{
				0, 4, 7, 10,
				2, 0, 8, 11,
				3, 6, 0, 12,
			})
		})

		Convey("T().Visit stops early if f() returns false", func() {
			saw := Zeros(a.T().Shape()...)
			b := Zeros(a.T().Shape()...)
			count := 0
			a.T().Visit(func(pos []int, value float64) bool {
				count++
				b.ItemSet(value, pos...)
				saw.ItemSet(1, pos...)
				if saw.CountNonzero() >= 5 {
					return false
				}
				return true
			})
			So(count, ShouldEqual, 5)
			So(saw.CountNonzero(), ShouldEqual, 5)
			So(b.Array(), ShouldResemble, []float64{
				0, 4, 0, 0,
				2, 0, 0, 0,
				3, 0, 0, 0,
			})
		})

		Convey("T().VisitNonzero sees just nonzero items", func() {
			saw := Zeros(a.T().Shape()...)
			b := Zeros(a.T().Shape()...)
			count := 0
			a.T().VisitNonzero(func(pos []int, value float64) bool {
				count++
				b.ItemSet(value, pos...)
				saw.ItemSet(1, pos...)
				return true
			})
			So(count, ShouldEqual, 9)
			So(saw.CountNonzero(), ShouldEqual, 9)
			So(b.Array(), ShouldResemble, []float64{
				0, 4, 7, 10,
				2, 0, 8, 11,
				3, 6, 0, 12,
			})
		})

		Convey("T().VisitNonzero stops early if f() returns false", func() {
			saw := Zeros(a.T().Shape()...)
			b := Zeros(a.T().Shape()...)
			count := 0
			a.T().VisitNonzero(func(pos []int, value float64) bool {
				count++
				b.ItemSet(value, pos...)
				saw.ItemSet(1, pos...)
				if saw.CountNonzero() >= 5 {
					return false
				}
				return true
			})
			So(count, ShouldEqual, 5)
			So(saw.CountNonzero(), ShouldEqual, 5)
			So(b.Array(), ShouldResemble, []float64{
				0, 4, 7, 0,
				2, 0, 0, 0,
				3, 6, 0, 0,
			})
		})
	})
}

func Test1DArray(t *testing.T) {
	Convey("Given an array with shape 5", t, func() {
		array := Dense(5)

		Convey("All is false", func() {
			So(array.All(), ShouldBeFalse)
		})

		Convey("Any is false", func() {
			So(array.Any(), ShouldBeFalse)
		})

		Convey("Size is 5", func() {
			So(array.Size(), ShouldEqual, 5)
		})

		Convey("NDim is 1", func() {
			So(array.NDim(), ShouldEqual, 1)
		})

		Convey("Shape is (5)", func() {
			So(array.Shape(), ShouldResemble, []int{5})
		})

		Convey("M() works", func() {
			m := array.M()
			So(m.Shape(), ShouldResemble, []int{5, 1})
			So(m.Array(), ShouldResemble, []float64{0, 0, 0, 0, 0})
		})

		Convey("Item() returns all zeros", func() {
			for i0 := 0; i0 < 5; i0++ {
				So(array.Item(i0), ShouldEqual, 0)
			}
		})

		Convey("Sum() is zero", func() {
			So(array.Sum(), ShouldEqual, 0)
		})

		Convey("When I call Normalize()", func() {
			array = array.Normalize()

			Convey("Item() returns all zeros", func() {
				for i0 := 0; i0 < 5; i0++ {
					So(array.Item(i0), ShouldEqual, 0)
				}
			})
		})

		Convey("When I call ItemSet", func() {
			array.ItemSet(1, 1)
			array.ItemSet(3, 3)

			Convey("All is false", func() {
				So(array.All(), ShouldBeFalse)
			})

			Convey("Any is true", func() {
				So(array.Any(), ShouldBeTrue)
			})

			Convey("Item() returns updates", func() {
				for i0 := 0; i0 < 5; i0++ {
					switch i0 {
					case 1:
						So(array.Item(i0), ShouldEqual, 1)
					case 3:
						So(array.Item(i0), ShouldEqual, 3)
					default:
						So(array.Item(i0), ShouldEqual, 0)
					}
				}
			})

			Convey("FlatItem returns the correct values", func() {
				for i := 0; i < array.Size(); i++ {
					switch i {
					case 1:
						So(array.FlatItem(i), ShouldEqual, 1)
					case 3:
						So(array.FlatItem(i), ShouldEqual, 3)
					default:
						So(array.FlatItem(i), ShouldEqual, 0)
					}
				}
			})

			Convey("Sum() is correct", func() {
				So(array.Sum(), ShouldEqual, 4)
			})

			Convey("When I call Normalize", func() {
				array = array.Normalize()

				Convey("Item() is correct", func() {
					for i0 := 0; i0 < 5; i0++ {
						switch i0 {
						case 1:
							So(array.Item(i0), ShouldEqual, 0.25)
						case 3:
							So(array.Item(i0), ShouldEqual, 0.75)
						default:
							So(array.Item(i0), ShouldEqual, 0)
						}
					}
				})

				Convey("Sum() is 1", func() {
					So(array.Sum(), ShouldEqual, 1)
				})
			})
		})

		Convey("When I call FlatItemSet", func() {
			array.Fill(0)
			array.FlatItemSet(1, 1)
			array.FlatItemSet(3, 3)

			Convey("All is false", func() {
				So(array.All(), ShouldBeFalse)
			})

			Convey("Any is true", func() {
				So(array.Any(), ShouldBeTrue)
			})

			Convey("Item() returns updates", func() {
				for i0 := 0; i0 < 5; i0++ {
					switch i0 {
					case 1:
						So(array.Item(i0), ShouldEqual, 1)
					case 3:
						So(array.Item(i0), ShouldEqual, 3)
					default:
						So(array.Item(i0), ShouldEqual, 0)
					}
				}
			})

			Convey("FlatItem returns the correct values", func() {
				for i := 0; i < array.Size(); i++ {
					switch i {
					case 1:
						So(array.FlatItem(i), ShouldEqual, 1)
					case 3:
						So(array.FlatItem(i), ShouldEqual, 3)
					default:
						So(array.FlatItem(i), ShouldEqual, 0)
					}
				}
			})
		})

		Convey("When I call Fill", func() {
			array.Fill(2)

			Convey("All is true", func() {
				So(array.All(), ShouldBeTrue)
			})

			Convey("Any is true", func() {
				So(array.Any(), ShouldBeTrue)
			})

			Convey("Item() returns updates", func() {
				for i0 := 0; i0 < 5; i0++ {
					So(array.Item(i0), ShouldEqual, 2)
				}
			})

			Convey("When I call Normalize", func() {
				array = array.Normalize()

				Convey("Item() is correct", func() {
					for i0 := 0; i0 < 5; i0++ {
						So(array.Item(i0), ShouldEqual, 0.2)
					}
				})

				Convey("Sum() is 1", func() {
					So(array.Sum(), ShouldEqual, 1)
				})
			})
		})
	})

	Convey("Given a random array with shape 5", t, func() {
		array := Rand(5)

		Convey("A slice with no indices panics", func() {
			So(func() { array.Slice([]int{}, []int{}) }, ShouldPanic)
		})

		Convey("A slice with too many indices panics", func() {
			So(func() { array.Slice([]int{1, 2}, []int{3, 4}) }, ShouldPanic)
		})

		Convey("The slice from [2:0] panics", func() {
			So(func() { array.Slice([]int{2}, []int{0}) }, ShouldPanic)
		})

		Convey("The slice from [-1:-3] panics", func() {
			So(func() { array.Slice([]int{-1}, []int{-3}) }, ShouldPanic)
		})

		Convey("The slice from [0:2] is correct", func() {
			slice := array.Slice([]int{0}, []int{2})
			So(slice.Shape(), ShouldResemble, []int{2})
			next := 0
			for i0 := 0; i0 < 2; i0++ {
				So(slice.FlatItem(next), ShouldEqual, array.Item(i0))
				next++
			}
		})

		Convey("The slice from [3:5] is correct", func() {
			slice := array.Slice([]int{3}, []int{5})
			So(slice.Shape(), ShouldResemble, []int{2})
			next := 0
			for i0 := 3; i0 < 5; i0++ {
				So(slice.FlatItem(next), ShouldEqual, array.Item(i0))
				next++
			}
		})

		Convey("The slice from [3:-1] is correct", func() {
			slice := array.Slice([]int{3}, []int{-1})
			So(slice.Shape(), ShouldResemble, []int{2})
			next := 0
			for i0 := 3; i0 < 5; i0++ {
				So(slice.FlatItem(next), ShouldEqual, array.Item(i0))
				next++
			}
		})

		Convey("The slice from [-3:-1] is correct", func() {
			slice := array.Slice([]int{-3}, []int{-1})
			So(slice.Shape(), ShouldResemble, []int{2})
			next := 0
			for i0 := 3; i0 < 5; i0++ {
				So(slice.FlatItem(next), ShouldEqual, array.Item(i0))
				next++
			}
		})
	})

	Convey("Given two 1D arrays of equal length", t, func() {
		a1 := Dense(4)
		a1.Fill(1)
		a2 := Dense(4)
		a2.Fill(2)

		Convey("Concat works along axis 0", func() {
			a3 := a1.Concat(0, a2)
			So(a3.Shape(), ShouldResemble, []int{8})
			for i0 := 0; i0 < 8; i0++ {
				if i0 < 4 {
					So(a3.Item(i0), ShouldEqual, 1)
				} else {
					So(a3.Item(i0), ShouldEqual, 2)
				}
			}
		})

		Convey("Concat works along axis 1", func() {
			a3 := a1.Concat(1, a2)
			So(a3.Shape(), ShouldResemble, []int{4, 2})
			for i0 := 0; i0 < 4; i0++ {
				for i1 := 0; i1 < 2; i1++ {
					if i1 == 0 {
						So(a3.Item(i0, i1), ShouldEqual, 1)
					} else {
						So(a3.Item(i0, i1), ShouldEqual, 2)
					}
				}
			}
		})

		Convey("Concat panics along axis 2", func() {
			So(func() { a1.Concat(2, a2) }, ShouldPanic)
		})
	})
}

func Test2DArray(t *testing.T) {
	Convey("Given an array with shape 5, 3", t, func() {
		array := Dense(5, 3)

		Convey("All is false", func() {
			So(array.All(), ShouldBeFalse)
		})

		Convey("Any is false", func() {
			So(array.Any(), ShouldBeFalse)
		})

		Convey("Size is 15", func() {
			So(array.Size(), ShouldEqual, 15)
		})

		Convey("NDim is 2", func() {
			So(array.NDim(), ShouldEqual, 2)
		})

		Convey("Shape is (5, 3)", func() {
			So(array.Shape(), ShouldResemble, []int{5, 3})
		})

		Convey("M() works", func() {
			m := array.M()
			So(m.Shape(), ShouldResemble, []int{5, 3})
			So(m.Array(), ShouldResemble, []float64{
				0, 0, 0,
				0, 0, 0,
				0, 0, 0,
				0, 0, 0,
				0, 0, 0,
			})
		})

		Convey("Item() returns all zeros", func() {
			for i0 := 0; i0 < 5; i0++ {
				for i1 := 0; i1 < 3; i1++ {
					So(array.Item(i0, i1), ShouldEqual, 0)
				}
			}
		})

		Convey("Sum() is zero", func() {
			So(array.Sum(), ShouldEqual, 0)
		})

		Convey("When I call Normalize()", func() {
			array = array.Normalize()
			Convey("Item() returns all zeros", func() {
				for i0 := 0; i0 < 5; i0++ {
					for i1 := 0; i1 < 3; i1++ {
						So(array.Item(i0, i1), ShouldEqual, 0)
					}
				}
			})
		})

		Convey("When I call ItemSet", func() {
			array.ItemSet(1, 1, 0)
			array.ItemSet(2, 3, 2)

			Convey("All is false", func() {
				So(array.All(), ShouldBeFalse)
			})

			Convey("Any is true", func() {
				So(array.Any(), ShouldBeTrue)
			})

			Convey("Item() returns updates", func() {
				for i0 := 0; i0 < 5; i0++ {
					for i1 := 0; i1 < 3; i1++ {
						if i0 == 1 && i1 == 0 {
							So(array.Item(i0, i1), ShouldEqual, 1)
						} else if i0 == 3 && i1 == 2 {
							So(array.Item(i0, i1), ShouldEqual, 2)
						} else {
							So(array.Item(i0, i1), ShouldEqual, 0)
						}
					}
				}
			})

			Convey("FlatItem returns the correct values", func() {
				for i := 0; i < array.Size(); i++ {
					switch i {
					case 1*3 + 0:
						So(array.FlatItem(i), ShouldEqual, 1)
					case 3*3 + 2:
						So(array.FlatItem(i), ShouldEqual, 2)
					default:
						So(array.FlatItem(i), ShouldEqual, 0)
					}
				}
			})

			Convey("Sum() is correct", func() {
				So(array.Sum(), ShouldEqual, 3)
			})

			Convey("When I call Normalize", func() {
				array = array.Normalize()

				Convey("Item() is correct", func() {
					for i0 := 0; i0 < 5; i0++ {
						for i1 := 0; i1 < 3; i1++ {
							if i0 == 1 && i1 == 0 {
								So(array.Item(i0, i1), ShouldEqual, 1.0/3)
							} else if i0 == 3 && i1 == 2 {
								So(array.Item(i0, i1), ShouldEqual, 2.0/3)
							} else {
								So(array.Item(i0, i1), ShouldEqual, 0)
							}
						}
					}
				})

				Convey("Sum() is 1", func() {
					So(array.Sum(), ShouldEqual, 1)
				})
			})
		})

		Convey("When I call FlatItemSet", func() {
			array.Fill(0)
			array.FlatItemSet(1, 1*3+0)
			array.FlatItemSet(2, 3*3+2)

			Convey("All is false", func() {
				So(array.All(), ShouldBeFalse)
			})

			Convey("Any is true", func() {
				So(array.Any(), ShouldBeTrue)
			})

			Convey("Item() returns updates", func() {
				for i0 := 0; i0 < 5; i0++ {
					for i1 := 0; i1 < 3; i1++ {
						if i0 == 1 && i1 == 0 {
							So(array.Item(i0, i1), ShouldEqual, 1)
						} else if i0 == 3 && i1 == 2 {
							So(array.Item(i0, i1), ShouldEqual, 2)
						} else {
							So(array.Item(i0, i1), ShouldEqual, 0)
						}
					}
				}
			})

			Convey("FlatItem returns the correct values", func() {
				for i := 0; i < array.Size(); i++ {
					switch i {
					case 1*3 + 0:
						So(array.FlatItem(i), ShouldEqual, 1)
					case 3*3 + 2:
						So(array.FlatItem(i), ShouldEqual, 2)
					default:
						So(array.FlatItem(i), ShouldEqual, 0)
					}
				}
			})
		})

		Convey("When I call Fill", func() {
			array.Fill(3)

			Convey("All is true", func() {
				So(array.All(), ShouldBeTrue)
			})

			Convey("Any is true", func() {
				So(array.Any(), ShouldBeTrue)
			})

			Convey("Item() returns updates", func() {
				for i0 := 0; i0 < 5; i0++ {
					for i1 := 0; i1 < 3; i1++ {
						So(array.Item(i0, i1), ShouldEqual, 3)
					}
				}
			})

			Convey("When I call Normalize", func() {
				array = array.Normalize()

				Convey("Item() is correct", func() {
					for i0 := 0; i0 < 5; i0++ {
						for i1 := 0; i1 < 3; i1++ {
							So(array.Item(i0, i1), ShouldEqual, 1.0/15)
						}
					}
				})

				Convey("Sum() is 1", func() {
					So(array.Sum(), ShouldBeBetween, 1-Eps, 1)
				})
			})
		})
	})

	Convey("Given a random array with shape 5,3", t, func() {
		array := Rand(5, 3)

		Convey("When I call T", func() {
			tr := array.M().T()

			Convey("The transpose shape is 3x5", func() {
				So(tr.Shape(), ShouldResemble, []int{3, 5})
			})

			Convey("The transpose size is 15", func() {
				So(tr.Size(), ShouldEqual, 15)
			})

			Convey("FlatItem works", func() {
				next := 0
				for i0 := 0; i0 < 3; i0++ {
					for i1 := 0; i1 < 5; i1++ {
						So(tr.FlatItem(next), ShouldEqual, array.Item(i1, i0))
						next++
					}
				}
			})

			Convey("Item works", func() {
				for i0 := 0; i0 < 3; i0++ {
					for i1 := 0; i1 < 5; i1++ {
						So(tr.Item(i0, i1), ShouldEqual, array.Item(i1, i0))
					}
				}
			})
		})

		Convey("A slice with no indices panics", func() {
			So(func() { array.Slice([]int{}, []int{}) }, ShouldPanic)
		})

		Convey("A slice with too few indices panics", func() {
			So(func() { array.Slice([]int{1}, []int{2}) }, ShouldPanic)
		})

		Convey("A slice with too many indices panics", func() {
			So(func() { array.Slice([]int{1, 2, 3}, []int{4, 5, 6}) }, ShouldPanic)
		})

		Convey("The slice from [2:0, 0:1] panics", func() {
			So(func() { array.Slice([]int{2, 0}, []int{0, 1}) }, ShouldPanic)
		})

		Convey("The slice from [-1:-3, 0:1] panics", func() {
			So(func() { array.Slice([]int{-1, 0}, []int{-3, 1}) }, ShouldPanic)
		})

		Convey("The slice from [0:2, 1:3] is correct", func() {
			slice := array.Slice([]int{0, 1}, []int{2, 3})
			So(slice.Shape(), ShouldResemble, []int{2, 2})
			next := 0
			for i0 := 0; i0 < 2; i0++ {
				for i1 := 1; i1 < 3; i1++ {
					So(slice.FlatItem(next), ShouldEqual, array.Item(i0, i1))
					next++
				}
			}
		})

		Convey("The slice from [3:5,0:2] is correct", func() {
			slice := array.Slice([]int{3, 0}, []int{5, 2})
			So(slice.Shape(), ShouldResemble, []int{2, 2})
			next := 0
			for i0 := 3; i0 < 5; i0++ {
				for i1 := 0; i1 < 2; i1++ {
					So(slice.FlatItem(next), ShouldEqual, array.Item(i0, i1))
					next++
				}
			}
		})

		Convey("The slice from [3:-1,0:-2] is correct", func() {
			slice := array.Slice([]int{3, 0}, []int{-1, -2})
			So(slice.Shape(), ShouldResemble, []int{2, 2})
			next := 0
			for i0 := 3; i0 < 5; i0++ {
				for i1 := 0; i1 < 2; i1++ {
					So(slice.FlatItem(next), ShouldEqual, array.Item(i0, i1))
					next++
				}
			}
		})

		Convey("The slice from [-3:-1,-3:-2] is correct", func() {
			slice := array.Slice([]int{-3, -3}, []int{-1, -2})
			So(slice.Shape(), ShouldResemble, []int{2, 1})
			next := 0
			for i0 := 3; i0 < 5; i0++ {
				for i1 := 1; i1 < 2; i1++ {
					So(slice.FlatItem(next), ShouldEqual, array.Item(i0, i1))
					next++
				}
			}
		})
	})

	Convey("Given two 2D arrays of equal length", t, func() {
		a1 := Dense(4, 3)
		a1.Fill(1)
		a2 := Dense(4, 3)
		a2.Fill(2)

		Convey("Concat works along axis 0", func() {
			a3 := a1.Concat(0, a2)
			So(a3.Shape(), ShouldResemble, []int{8, 3})
			for i0 := 0; i0 < 8; i0++ {
				for i1 := 0; i1 < 3; i1++ {
					if i0 < 4 {
						So(a3.Item(i0, i1), ShouldEqual, 1)
					} else {
						So(a3.Item(i0, i1), ShouldEqual, 2)
					}
				}
			}
		})

		Convey("Concat works along axis 1", func() {
			a3 := a1.Concat(1, a2)
			So(a3.Shape(), ShouldResemble, []int{4, 6})
			for i0 := 0; i0 < 4; i0++ {
				for i1 := 0; i1 < 6; i1++ {
					if i1 < 3 {
						So(a3.Item(i0, i1), ShouldEqual, 1)
					} else {
						So(a3.Item(i0, i1), ShouldEqual, 2)
					}
				}
			}
		})

		Convey("Concat works along axis 2", func() {
			a3 := a1.Concat(2, a2)
			So(a3.Shape(), ShouldResemble, []int{4, 3, 2})
			for i0 := 0; i0 < 4; i0++ {
				for i1 := 0; i1 < 3; i1++ {
					for i2 := 0; i2 < 2; i2++ {
						if i2 == 0 {
							So(a3.Item(i0, i1, i2), ShouldEqual, 1)
						} else {
							So(a3.Item(i0, i1, i2), ShouldEqual, 2)
						}
					}
				}
			}
		})

		Convey("Concat panics along axis 3", func() {
			So(func() { a1.Concat(3, a2) }, ShouldPanic)
		})
	})

	Convey("Given a 2x3 and 3x4 array", t, func() {
		left := Rand(2, 3).M()
		right := Rand(3, 4).M()
		Convey("MProd() works", func() {
			result := left.MProd(right)
			So(result.Shape(), ShouldResemble, []int{2, 4})
			for i0 := 0; i0 < 2; i0++ {
				for i1 := 0; i1 < 4; i1++ {
					c := left.Item(i0, 0)*right.Item(0, i1) +
						left.Item(i0, 1)*right.Item(1, i1) +
						left.Item(i0, 2)*right.Item(2, i1)
					So(result.Item(i0, i1), ShouldEqual, c)
				}
			}
		})
	})
}

func Test3DArray(t *testing.T) {
	Convey("Given an array with shape 5, 3, 2", t, func() {
		array := Dense(5, 3, 2)

		Convey("All is false", func() {
			So(array.All(), ShouldBeFalse)
		})

		Convey("Any is false", func() {
			So(array.Any(), ShouldBeFalse)
		})

		Convey("Size is 30", func() {
			So(array.Size(), ShouldEqual, 30)
		})

		Convey("NDim is 3", func() {
			So(array.NDim(), ShouldEqual, 3)
		})

		Convey("Shape is (5, 3, 2)", func() {
			So(array.Shape(), ShouldResemble, []int{5, 3, 2})
		})

		Convey("M() panics", func() {
			So(func() { array.M() }, ShouldPanic)
		})

		Convey("Item() returns all zeros", func() {
			for i0 := 0; i0 < 5; i0++ {
				for i1 := 0; i1 < 3; i1++ {
					for i2 := 0; i2 < 2; i2++ {
						So(array.Item(i0, i1, i2), ShouldEqual, 0)
					}
				}
			}
		})

		Convey("Sum() is zero", func() {
			So(array.Sum(), ShouldEqual, 0)
		})

		Convey("When I call Normalize()", func() {
			array = array.Normalize()
			Convey("Item() returns all zeros", func() {
				for i0 := 0; i0 < 5; i0++ {
					for i1 := 0; i1 < 3; i1++ {
						for i2 := 0; i2 < 2; i2++ {
							So(array.Item(i0, i1, i2), ShouldEqual, 0)
						}
					}
				}
			})
		})

		Convey("When I call ItemSet", func() {
			array.ItemSet(1, 1, 0, 0)
			array.ItemSet(2, 3, 2, 0)
			array.ItemSet(3, 3, 2, 1)

			Convey("All is false", func() {
				So(array.All(), ShouldBeFalse)
			})

			Convey("Any is true", func() {
				So(array.Any(), ShouldBeTrue)
			})

			Convey("Min is correct", func() {
				So(array.Min(), ShouldEqual, 0)
			})

			Convey("Max is correct", func() {
				So(array.Max(), ShouldEqual, 3)
			})

			Convey("Item() returns updates", func() {
				for i0 := 0; i0 < 5; i0++ {
					for i1 := 0; i1 < 3; i1++ {
						for i2 := 0; i2 < 2; i2++ {
							if i0 == 1 && i1 == 0 && i2 == 0 {
								So(array.Item(i0, i1, i2), ShouldEqual, 1)
							} else if i0 == 3 && i1 == 2 && i2 == 0 {
								So(array.Item(i0, i1, i2), ShouldEqual, 2)
							} else if i0 == 3 && i1 == 2 && i2 == 1 {
								So(array.Item(i0, i1, i2), ShouldEqual, 3)
							} else {
								So(array.Item(i0, i1, i2), ShouldEqual, 0)
							}
						}
					}
				}
			})

			Convey("FlatItem returns the correct values", func() {
				for i := 0; i < array.Size(); i++ {
					switch i {
					case 1*3*2 + 0*2 + 0:
						So(array.FlatItem(i), ShouldEqual, 1)
					case 3*3*2 + 2*2 + 0:
						So(array.FlatItem(i), ShouldEqual, 2)
					case 3*3*2 + 2*2 + 1:
						So(array.FlatItem(i), ShouldEqual, 3)
					default:
						So(array.FlatItem(i), ShouldEqual, 0)
					}
				}
			})

			Convey("Sum() is correct", func() {
				So(array.Sum(), ShouldEqual, 6)
			})

			Convey("When I call Normalize", func() {
				array = array.Normalize()

				Convey("Item() is correct", func() {
					for i0 := 0; i0 < 5; i0++ {
						for i1 := 0; i1 < 3; i1++ {
							for i2 := 0; i2 < 2; i2++ {
								if i0 == 1 && i1 == 0 && i2 == 0 {
									So(array.Item(i0, i1, i2), ShouldEqual, 1.0/6)
								} else if i0 == 3 && i1 == 2 && i2 == 0 {
									So(array.Item(i0, i1, i2), ShouldEqual, 1.0/3)
								} else if i0 == 3 && i1 == 2 && i2 == 1 {
									So(array.Item(i0, i1, i2), ShouldEqual, 0.5)
								} else {
									So(array.Item(i0, i1, i2), ShouldEqual, 0)
								}
							}
						}
					}
				})

				Convey("Sum() is 1", func() {
					So(array.Sum(), ShouldEqual, 1)
				})
			})
		})

		Convey("When I call FlatItemSet", func() {
			array.FlatItemSet(1, 1*3*2+0*2+0)
			array.FlatItemSet(2, 3*3*2+2*2+0)
			array.FlatItemSet(3, 3*3*2+2*2+1)

			Convey("All is false", func() {
				So(array.All(), ShouldBeFalse)
			})

			Convey("Any is true", func() {
				So(array.Any(), ShouldBeTrue)
			})

			Convey("Item() returns updates", func() {
				for i0 := 0; i0 < 5; i0++ {
					for i1 := 0; i1 < 3; i1++ {
						for i2 := 0; i2 < 2; i2++ {
							if i0 == 1 && i1 == 0 && i2 == 0 {
								So(array.Item(i0, i1, i2), ShouldEqual, 1)
							} else if i0 == 3 && i1 == 2 && i2 == 0 {
								So(array.Item(i0, i1, i2), ShouldEqual, 2)
							} else if i0 == 3 && i1 == 2 && i2 == 1 {
								So(array.Item(i0, i1, i2), ShouldEqual, 3)
							} else {
								So(array.Item(i0, i1, i2), ShouldEqual, 0)
							}
						}
					}
				}
			})

			Convey("FlatItem returns the correct values", func() {
				for i := 0; i < array.Size(); i++ {
					switch i {
					case 1*3*2 + 0*2 + 0:
						So(array.FlatItem(i), ShouldEqual, 1)
					case 3*3*2 + 2*2 + 0:
						So(array.FlatItem(i), ShouldEqual, 2)
					case 3*3*2 + 2*2 + 1:
						So(array.FlatItem(i), ShouldEqual, 3)
					default:
						So(array.FlatItem(i), ShouldEqual, 0)
					}
				}
			})
		})

		Convey("When I call Fill", func() {
			array.Fill(0.5)

			Convey("All is true", func() {
				So(array.All(), ShouldBeTrue)
			})

			Convey("Any is true", func() {
				So(array.Any(), ShouldBeTrue)
			})

			Convey("Item() returns updates", func() {
				for i0 := 0; i0 < 5; i0++ {
					for i1 := 0; i1 < 3; i1++ {
						for i2 := 0; i2 < 2; i2++ {
							So(array.Item(i0, i1, i2), ShouldEqual, 0.5)
						}
					}
				}
			})

			Convey("When I call Normalize", func() {
				array = array.Normalize()

				Convey("Item() is correct", func() {
					for i0 := 0; i0 < 5; i0++ {
						for i1 := 0; i1 < 3; i1++ {
							for i2 := 0; i2 < 2; i2++ {
								So(array.Item(i0, i1, i2), ShouldEqual, 1.0/30)
							}
						}
					}
				})

				Convey("Sum() is 1", func() {
					So(array.Sum(), ShouldBeBetween, 1-Eps, 1)
				})
			})
		})
	})

	Convey("Given a random array with shape 5,3,2", t, func() {
		array := Rand(5, 3, 2)

		Convey("A slice with no indices panics", func() {
			So(func() { array.Slice([]int{}, []int{}) }, ShouldPanic)
		})

		Convey("A slice with too few indices panics", func() {
			So(func() { array.Slice([]int{1, 2}, []int{2, 3}) }, ShouldPanic)
		})

		Convey("A slice with too many indices panics", func() {
			So(func() { array.Slice([]int{1, 2, 3, 4}, []int{4, 5, 6, 7}) }, ShouldPanic)
		})

		Convey("The slice from [2:0, 0:1,0:1] panics", func() {
			So(func() { array.Slice([]int{2, 0, 0}, []int{0, 1, 1}) }, ShouldPanic)
		})

		Convey("The slice from [-1:-3, 0:1] panics", func() {
			So(func() { array.Slice([]int{-1, 0}, []int{-3, 1}) }, ShouldPanic)
		})

		Convey("The slice from [0:2, 1:3,0:1] is correct", func() {
			slice := array.Slice([]int{0, 1, 0}, []int{2, 3, 1})
			So(slice.Shape(), ShouldResemble, []int{2, 2, 1})
			next := 0
			for i0 := 0; i0 < 2; i0++ {
				for i1 := 1; i1 < 3; i1++ {
					for i2 := 0; i2 < 1; i2++ {
						So(slice.FlatItem(next), ShouldEqual, array.Item(i0, i1, i2))
						next++
					}
				}
			}
		})

		Convey("The slice from [3:5,0:2,0:2] is correct", func() {
			slice := array.Slice([]int{3, 0, 0}, []int{5, 2, 2})
			So(slice.Shape(), ShouldResemble, []int{2, 2, 2})
			next := 0
			for i0 := 3; i0 < 5; i0++ {
				for i1 := 0; i1 < 2; i1++ {
					for i2 := 0; i2 < 2; i2++ {
						So(slice.FlatItem(next), ShouldEqual, array.Item(i0, i1, i2))
						next++
					}
				}
			}
		})

		Convey("The slice from [3:-1,0:-2,0:-1] is correct", func() {
			slice := array.Slice([]int{3, 0, 0}, []int{-1, -2, -1})
			So(slice.Shape(), ShouldResemble, []int{2, 2, 2})
			next := 0
			for i0 := 3; i0 < 5; i0++ {
				for i1 := 0; i1 < 2; i1++ {
					for i2 := 0; i2 < 2; i2++ {
						So(slice.FlatItem(next), ShouldEqual, array.Item(i0, i1, i2))
						next++
					}
				}
			}
		})

		Convey("The slice from [-3:-1,-3:-2,-2:-1] is correct", func() {
			slice := array.Slice([]int{-3, -3, -2}, []int{-1, -2, -1})
			So(slice.Shape(), ShouldResemble, []int{2, 1, 1})
			next := 0
			for i0 := 3; i0 < 5; i0++ {
				for i1 := 1; i1 < 2; i1++ {
					for i2 := 1; i2 < 2; i2++ {
						So(slice.FlatItem(next), ShouldEqual, array.Item(i0, i1, i2))
						next++
					}
				}
			}
		})
	})

	Convey("Given two 3D arrays of equal length", t, func() {
		a1 := Dense(4, 3, 2)
		a1.Fill(1)
		a2 := Dense(4, 3, 2)
		a2.Fill(2)

		Convey("Concat works along axis 0", func() {
			a3 := a1.Concat(0, a2)
			So(a3.Shape(), ShouldResemble, []int{8, 3, 2})
			for i0 := 0; i0 < 8; i0++ {
				for i1 := 0; i1 < 3; i1++ {
					for i2 := 0; i2 < 2; i2++ {
						if i0 < 4 {
							So(a3.Item(i0, i1, i2), ShouldEqual, 1)
						} else {
							So(a3.Item(i0, i1, i2), ShouldEqual, 2)
						}
					}
				}
			}
		})

		Convey("Concat works along axis 1", func() {
			a3 := a1.Concat(1, a2)
			So(a3.Shape(), ShouldResemble, []int{4, 6, 2})
			for i0 := 0; i0 < 4; i0++ {
				for i1 := 0; i1 < 6; i1++ {
					for i2 := 0; i2 < 2; i2++ {
						if i1 < 3 {
							So(a3.Item(i0, i1, i2), ShouldEqual, 1)
						} else {
							So(a3.Item(i0, i1, i2), ShouldEqual, 2)
						}
					}
				}
			}
		})

		Convey("Concat works along axis 2", func() {
			a3 := a1.Concat(2, a2)
			So(a3.Shape(), ShouldResemble, []int{4, 3, 4})
			for i0 := 0; i0 < 4; i0++ {
				for i1 := 0; i1 < 3; i1++ {
					for i2 := 0; i2 < 4; i2++ {
						if i2 < 2 {
							So(a3.Item(i0, i1, i2), ShouldEqual, 1)
						} else {
							So(a3.Item(i0, i1, i2), ShouldEqual, 2)
						}
					}
				}
			}
		})

		Convey("Concat works along axis 3", func() {
			a3 := a1.Concat(3, a2)
			So(a3.Shape(), ShouldResemble, []int{4, 3, 2, 2})
			for i0 := 0; i0 < 4; i0++ {
				for i1 := 0; i1 < 3; i1++ {
					for i2 := 0; i2 < 2; i2++ {
						for i3 := 0; i3 < 2; i3++ {
							if i3 == 0 {
								So(a3.Item(i0, i1, i2, i3), ShouldEqual, 1)
							} else {
								So(a3.Item(i0, i1, i2, i3), ShouldEqual, 2)
							}
						}
					}
				}
			}
		})

		Convey("Concat panics along axis 4", func() {
			So(func() { a1.Concat(4, a2) }, ShouldPanic)
		})
	})
}
