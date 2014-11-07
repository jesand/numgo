package matrix

import (
	. "github.com/smartystreets/goconvey/convey"
	"math"
	"testing"
)

func cooDiag(values ...float64) Matrix {
	m := SparseCoo(len(values), len(values))
	for i, v := range values {
		m.ItemSet(v, i, i)
	}
	return m
}

func TestSparseCooAddDivMulSub(t *testing.T) {
	Convey("Given two sparse diag matrices", t, func() {
		d1 := cooDiag(1, 2, 3, 4)
		d2 := cooDiag(5, 6, 7, 8)

		Convey("Add works", func() {
			a := d1.Add(d2)
			So(a.Shape(), ShouldResemble, []int{4, 4})
			So(a.Array(), ShouldResemble, []float64{
				6, 0, 0, 0,
				0, 8, 0, 0,
				0, 0, 10, 0,
				0, 0, 0, 12,
			})
		})

		Convey("Div works", func() {
			a := d2.Div(d1)
			So(a.Shape(), ShouldResemble, []int{4, 4})
			So(a.Array(), ShouldResemble, []float64{
				5, 0, 0, 0,
				0, 3, 0, 0,
				0, 0, 7. / 3, 0,
				0, 0, 0, 2,
			})
		})

		Convey("Prod works", func() {
			a := d2.Prod(d1)
			So(a.Shape(), ShouldResemble, []int{4, 4})
			So(a.Array(), ShouldResemble, []float64{
				5, 0, 0, 0,
				0, 12, 0, 0,
				0, 0, 21, 0,
				0, 0, 0, 32,
			})
		})

		Convey("Sub works", func() {
			a := d2.Sub(d1)
			So(a.Shape(), ShouldResemble, []int{4, 4})
			So(a.Array(), ShouldResemble, []float64{
				4, 0, 0, 0,
				0, 4, 0, 0,
				0, 0, 4, 0,
				0, 0, 0, 4,
			})
		})
	})
}

func TestSparseCooAllAny(t *testing.T) {
	Convey("Given partial and empty arrays", t, func() {
		partial := cooDiag(1, 2, 3)
		empty := cooDiag(0, 0, 0)

		Convey("All() is correct", func() {
			So(partial.All(), ShouldBeFalse)
			So(empty.All(), ShouldBeFalse)
		})

		Convey("Any() is correct", func() {
			So(partial.Any(), ShouldBeTrue)
			So(empty.Any(), ShouldBeFalse)
		})
	})

	Convey("Given pos, neg, and mixed arrays", t, func() {
		pos := cooDiag(1, 2, 3)
		neg := cooDiag(-1, -2, -3)
		mixed := cooDiag(1, -3, 3)
		fge := func(v float64) bool { return v >= 0 }
		fgt := func(v float64) bool { return v > 0 }
		f2ge := func(v1, v2 float64) bool { return v1 >= v2 }
		f2gt := func(v1, v2 float64) bool { return v1 > v2 }

		Convey("AllF() is correct", func() {
			So(pos.AllF(fge), ShouldBeTrue)
			So(mixed.AllF(fge), ShouldBeFalse)
			So(neg.AllF(fge), ShouldBeFalse)
		})

		Convey("AnyF() is correct", func() {
			So(pos.AnyF(fgt), ShouldBeTrue)
			So(mixed.AnyF(fgt), ShouldBeTrue)
			So(neg.AnyF(fgt), ShouldBeFalse)
		})

		Convey("AllF2() is correct", func() {
			So(pos.AllF2(f2ge, neg), ShouldBeTrue)
			So(mixed.AllF2(f2ge, neg), ShouldBeFalse)
			So(neg.AllF2(f2ge, neg), ShouldBeTrue)
		})

		Convey("AnyF2() is correct", func() {
			So(pos.AnyF2(f2gt, neg), ShouldBeTrue)
			So(mixed.AnyF2(f2gt, neg), ShouldBeTrue)
			So(neg.AnyF2(f2gt, neg), ShouldBeFalse)
		})
	})
}

func TestSparseCooApply(t *testing.T) {
	Convey("Apply works", t, func() {
		a := cooDiag(1, 2, 3)
		a2 := a.Apply(func(v float64) float64 { return 2 * v })
		So(a2.Shape(), ShouldResemble, []int{3, 3})
		So(a2.Array(), ShouldResemble, []float64{
			2, 0, 0,
			0, 4, 0,
			0, 0, 6,
		})
	})
}

func TestSparseCooColColSetCols(t *testing.T) {
	Convey("Given an array", t, func() {
		a := cooDiag(1, 2, 3)

		Convey("Cols is correct", func() {
			So(a.Cols(), ShouldEqual, 3)
		})

		Convey("Col panics with invalid input", func() {
			So(func() { a.Col(-1) }, ShouldPanic)
			So(func() { a.Col(3) }, ShouldPanic)
		})

		Convey("Col works", func() {
			So(a.Col(0), ShouldResemble, []float64{1, 0, 0})
			So(a.Col(1), ShouldResemble, []float64{0, 2, 0})
			So(a.Col(2), ShouldResemble, []float64{0, 0, 3})
		})

		Convey("ColSet panics with invalid input", func() {
			So(func() { a.ColSet(-1, []float64{0, 1, 2}) }, ShouldPanic)
			So(func() { a.ColSet(3, []float64{0, 1, 2}) }, ShouldPanic)
			So(func() { a.ColSet(0, []float64{0, 1}) }, ShouldPanic)
			So(func() { a.ColSet(0, []float64{0, 1, 2, 3}) }, ShouldPanic)
		})

		Convey("ColSet works", func() {
			a.ColSet(0, []float64{0, 1, 2})
			So(a.Array(), ShouldResemble, []float64{
				0, 0, 0,
				1, 2, 0,
				2, 0, 3,
			})
			a.ColSet(1, []float64{0, 1, 2})
			So(a.Array(), ShouldResemble, []float64{
				0, 0, 0,
				1, 1, 0,
				2, 2, 3,
			})
			a.ColSet(2, []float64{0, 1, 2})
			So(a.Array(), ShouldResemble, []float64{
				0, 0, 0,
				1, 1, 1,
				2, 2, 2,
			})
		})
	})
}

func TestSparseCooDiag(t *testing.T) {
	Convey("Given an array", t, func() {
		a := SparseCoo(3, 4)
		a.ItemSet(1.0, 0, 0)
		a.ItemSet(2.0, 1, 1)
		a.ItemSet(3.0, 2, 2)

		Convey("Diag() works", func() {
			d := a.Diag()
			So(d.Shape(), ShouldResemble, []int{3, 1})
			So(d.Array(), ShouldResemble, []float64{1, 2, 3})
		})

		Convey("Transpose Diag() works", func() {
			d := a.T().M().Diag()
			So(d.Shape(), ShouldResemble, []int{3, 1})
			So(d.Array(), ShouldResemble, []float64{1, 2, 3})
		})
	})
}

func TestSparseCooInverseNormLDivide(t *testing.T) {
	Convey("Given an invertible diagonal matrix", t, func() {
		m := cooDiag(2, 3, 5)

		Convey("When I take the inverse", func() {
			mi, err := m.Inverse()
			So(err, ShouldBeNil)

			Convey("The inverse is correct", func() {
				So(mi.Shape(), ShouldResemble, []int{3, 3})
				So(mi.Array(), ShouldResemble, []float64{
					.5, 0, 0,
					0, 1. / 3, 0,
					0, 0, .2,
				})
			})

			Convey("The inverse gives us back I", func() {
				i := m.MProd(mi)
				So(i.Shape(), ShouldResemble, []int{3, 3})
				So(i.Array(), ShouldResemble, []float64{
					1, 0, 0,
					0, 1, 0,
					0, 0, 1,
				})
			})
		})
	})

	Convey("Given a 3x3 matrix", t, func() {
		m := cooDiag(2, 3, 5)

		Convey("The 1-norm is correct", func() {
			So(m.Norm(1), ShouldEqual, 5)
		})

		Convey("The 2-norm is correct", func() {
			So(m.Norm(2), ShouldEqual, 5)
		})

		Convey("The inf-norm is correct", func() {
			So(m.Norm(math.Inf(1)), ShouldEqual, 5)
		})
	})

	Convey("Given a simple division problem", t, func() {
		a := cooDiag(4, 6, 8)
		b := cooDiag(2, 2, 2)

		Convey("When I solve the system", func() {
			x := a.LDivide(b)

			Convey("I get the correct solution", func() {
				So(x.Shape(), ShouldResemble, []int{3, 3})
				So(x.Array(), ShouldResemble, []float64{
					.5, 0, 0,
					0, 1. / 3, 0,
					0, 0, .25,
				})
			})

			Convey("The product ax = b is true", func() {
				b2 := a.MProd(x)
				So(b2.Shape(), ShouldResemble, []int{3, 3})
				So(b2.Equal(b), ShouldBeTrue)
			})
		})
	})
}

func TestSparseCooItemMath(t *testing.T) {
	Convey("Given a diag array", t, func() {
		a := cooDiag(1, 2, 3)

		Convey("When I call ItemAdd", func() {
			a2 := a.ItemAdd(1)
			Convey("The result is correct", func() {
				So(a2.Array(), ShouldResemble, []float64{
					2, 1, 1,
					1, 3, 1,
					1, 1, 4,
				})
			})
		})

		Convey("When I call ItemDiv", func() {
			a2 := a.ItemDiv(2)
			Convey("The result is correct", func() {
				So(a2.Array(), ShouldResemble, []float64{
					.5, 0, 0,
					0, 1, 0,
					0, 0, 1.5,
				})
			})
		})

		Convey("When I call ItemProd", func() {
			a2 := a.ItemProd(2)
			Convey("The result is correct", func() {
				So(a2.Array(), ShouldResemble, []float64{
					2, 0, 0,
					0, 4, 0,
					0, 0, 6,
				})
			})
		})

		Convey("When I call ItemSub", func() {
			a2 := a.ItemSub(1)
			Convey("The result is correct", func() {
				So(a2.Array(), ShouldResemble, []float64{
					0, -1, -1,
					-1, 1, -1,
					-1, -1, 2,
				})
			})
		})
	})
}

func TestSparseCooVisit(t *testing.T) {
	Convey("Given a sparse coo array", t, func() {
		a := SparseCoo(4, 3)
		for row := 0; row < 4; row++ {
			for col := 0; col < 3; col++ {
				if row != col {
					a.ItemSet(float64(row*3+col+1), row, col)
				}
			}
		}

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
			b.VisitNonzero(func(pos []int, value float64) bool {
				So(value, ShouldEqual, a.Item(pos...))
				return true
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
			b.VisitNonzero(func(pos []int, value float64) bool {
				So(value, ShouldEqual, a.Item(pos...))
				return true
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
			b.VisitNonzero(func(pos []int, value float64) bool {
				So(value, ShouldEqual, a.T().Item(pos...))
				return true
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
			b.VisitNonzero(func(pos []int, value float64) bool {
				So(value, ShouldEqual, a.T().Item(pos...))
				return true
			})
		})
	})
}

func TestSparseCooMaxMin(t *testing.T) {
	Convey("Given positive and negative diagonal arrays", t, func() {
		pos := cooDiag(1, 2, 3)
		neg := cooDiag(-1, -2, -3)

		Convey("Max is right", func() {
			So(pos.Max(), ShouldEqual, 3)
			So(neg.Max(), ShouldEqual, 0)
		})

		Convey("Min is right", func() {
			So(pos.Min(), ShouldEqual, 0)
			So(neg.Min(), ShouldEqual, -3)
		})
	})
}

func TestSparseCooRowRowSetRows(t *testing.T) {
	Convey("Given an array", t, func() {
		a := cooDiag(1, 2, 3)

		Convey("Rows is correct", func() {
			So(a.Rows(), ShouldEqual, 3)
		})

		Convey("Row panics with invalid input", func() {
			So(func() { a.Row(-1) }, ShouldPanic)
			So(func() { a.Row(3) }, ShouldPanic)
		})

		Convey("Row works", func() {
			So(a.Row(0), ShouldResemble, []float64{1, 0, 0})
			So(a.Row(1), ShouldResemble, []float64{0, 2, 0})
			So(a.Row(2), ShouldResemble, []float64{0, 0, 3})
		})

		Convey("RowSet panics with invalid input", func() {
			So(func() { a.RowSet(-1, []float64{0, 1, 2}) }, ShouldPanic)
			So(func() { a.RowSet(3, []float64{0, 1, 2}) }, ShouldPanic)
			So(func() { a.RowSet(0, []float64{0, 1}) }, ShouldPanic)
			So(func() { a.RowSet(0, []float64{0, 1, 2, 3}) }, ShouldPanic)
		})

		Convey("RowSet works", func() {
			a.RowSet(0, []float64{0, 1, 2})
			So(a.Array(), ShouldResemble, []float64{
				0, 1, 2,
				0, 2, 0,
				0, 0, 3,
			})
			a.RowSet(1, []float64{0, 1, 2})
			So(a.Array(), ShouldResemble, []float64{
				0, 1, 2,
				0, 1, 2,
				0, 0, 3,
			})
			a.RowSet(2, []float64{0, 1, 2})
			So(a.Array(), ShouldResemble, []float64{
				0, 1, 2,
				0, 1, 2,
				0, 1, 2,
			})
		})
	})
}

func TestSparseCooTranspose(t *testing.T) {
	Convey("Given a transposed matrix", t, func() {
		a := SparseCoo(4, 3)
		a.ItemSet(1.0, 0, 1)
		a.ItemSet(2.0, 2, 0)
		a.ItemSet(3.0, 3, 2)
		tr := a.T()

		Convey("The matrix statistics are correct", func() {
			So(tr.Size(), ShouldEqual, 12)
			So(tr.Shape(), ShouldResemble, []int{3, 4})
		})

		Convey("FlatItem and FlatItemSet work", func() {
			tr.FlatItemSet(-1, 8)
			So(tr.Array(), ShouldResemble, []float64{
				0, 0, 2, 0,
				1, 0, 0, 0,
				-1, 0, 0, 3,
			})
			So(tr.Item(1, 0), ShouldEqual, 1)
			So(tr.Item(2, 0), ShouldEqual, -1)
			So(tr.FlatItem(4), ShouldEqual, 1)
			So(tr.FlatItem(8), ShouldEqual, -1)
		})

		Convey("Item and ItemSet work", func() {
			tr.ItemSet(-1, 2, 2)
			So(tr.Array(), ShouldResemble, []float64{
				0, 0, 2, 0,
				1, 0, 0, 0,
				0, 0, -1, 3,
			})
			So(tr.Item(2, 2), ShouldEqual, -1)
			So(tr.Item(2, 3), ShouldEqual, 3)
			So(tr.FlatItem(10), ShouldEqual, -1)
			So(tr.FlatItem(11), ShouldEqual, 3)
		})

		Convey("ItemAdd works", func() {
			res := tr.ItemAdd(1)
			So(res.Array(), ShouldResemble, []float64{
				1, 1, 3, 1,
				2, 1, 1, 1,
				1, 1, 1, 4,
			})
		})

		Convey("ItemDiv works", func() {
			res := tr.ItemDiv(2)
			So(res.Array(), ShouldResemble, []float64{
				0, 0, 1, 0,
				.5, 0, 0, 0,
				0, 0, 0, 1.5,
			})
		})

		Convey("ItemProd works", func() {
			res := tr.ItemProd(2)
			So(res.Array(), ShouldResemble, []float64{
				0, 0, 4, 0,
				2, 0, 0, 0,
				0, 0, 0, 6,
			})
		})

		Convey("ItemSub works", func() {
			res := tr.ItemSub(1)
			So(res.Array(), ShouldResemble, []float64{
				-1, -1, 1, -1,
				0, -1, -1, -1,
				-1, -1, -1, 2,
			})
		})

		Convey("Array() is correct", func() {
			So(a.Array(), ShouldResemble, []float64{
				0, 1, 0,
				0, 0, 0,
				2, 0, 0,
				0, 0, 3,
			})
			So(tr.Array(), ShouldResemble, []float64{
				0, 0, 2, 0,
				1, 0, 0, 0,
				0, 0, 0, 3,
			})
		})

		Convey("Copy() is correct", func() {
			So(tr.Copy().Array(), ShouldResemble, []float64{
				0, 0, 2, 0,
				1, 0, 0, 0,
				0, 0, 0, 3,
			})
		})

		Convey("Ravel() is correct", func() {
			So(tr.Ravel().Array(), ShouldResemble, []float64{
				0, 0, 2, 0,
				1, 0, 0, 0,
				0, 0, 0, 3,
			})
		})
	})
}

func TestSparseCooMatrix(t *testing.T) {
	Convey("Given a sparse matrix with shape 5, 3", t, func() {
		array := SparseCoo(5, 3)

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

		Convey("Item() panics with invalid indices", func() {
			So(func() { array.Item(5, 0) }, ShouldPanic)
			So(func() { array.Item(0, 3) }, ShouldPanic)
			So(func() { array.Item(0) }, ShouldPanic)
			So(func() { array.Item(0, 0, 0) }, ShouldPanic)
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
			norm := array.Normalize()
			Convey("Item() returns all zeros", func() {
				for i0 := 0; i0 < 5; i0++ {
					for i1 := 0; i1 < 3; i1++ {
						So(norm.Item(i0, i1), ShouldEqual, 0)
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

			Convey("Equal is correct", func() {
				So(array.Equal(Zeros(5, 3)), ShouldBeFalse)
				So(array.Equal(array.Copy()), ShouldBeTrue)
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
				norm := array.Normalize()

				Convey("Item() is correct", func() {
					for i0 := 0; i0 < 5; i0++ {
						for i1 := 0; i1 < 3; i1++ {
							if i0 == 1 && i1 == 0 {
								So(norm.Item(i0, i1), ShouldEqual, 1.0/3)
							} else if i0 == 3 && i1 == 2 {
								So(norm.Item(i0, i1), ShouldEqual, 2.0/3)
							} else {
								So(norm.Item(i0, i1), ShouldEqual, 0)
							}
						}
					}
				})

				Convey("Sum() is 1", func() {
					So(norm.Sum(), ShouldEqual, 1)
				})
			})
		})

		Convey("When I call FlatItemSet", func() {
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

		Convey("Fill panics", func() {
			So(func() { array.Fill(3) }, ShouldPanic)
		})
	})

	Convey("Given a random array with shape 5,3", t, func() {
		array := SparseRand(5, 3, 0.2)

		Convey("When I call T", func() {
			tr := array.T()

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

	Convey("Given two 2D sparse matrices of equal length", t, func() {
		a1 := SparseCoo(4, 3)
		a2 := SparseCoo(4, 3)
		for row := 0; row < 4; row++ {
			for col := 0; col < 3; col++ {
				a1.ItemSet(1, row, col)
				a2.ItemSet(2, row, col)
			}
		}

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
		left := SparseRand(2, 3, 0.2).M()
		right := SparseRand(3, 4, 0.2).M()
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
