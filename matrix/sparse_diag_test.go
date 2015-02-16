package matrix

import (
	. "github.com/smartystreets/goconvey/convey"
	"math"
	"testing"
)

func TestSparseDiagAddDivMulSub(t *testing.T) {
	Convey("Given two sparse diag matrices", t, func() {
		d1 := Diag(1, 2, 3, 4)
		d2 := Diag(5, 6, 7, 8)

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

func TestSparseDiagAllAny(t *testing.T) {
	Convey("Given partial and empty arrays", t, func() {
		partial := Diag(1, 2, 3)
		empty := Diag(0, 0, 0)

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
		pos := Diag(1, 2, 3)
		neg := Diag(-1, -2, -3)
		mixed := Diag(1, -3, 3)
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

func TestSparseDiagApply(t *testing.T) {
	Convey("Apply works", t, func() {
		a := Diag(1, 2, 3)
		a2 := a.Apply(func(v float64) float64 { return 2 * v })
		So(a2.Shape(), ShouldResemble, []int{3, 3})
		So(a2.Array(), ShouldResemble, []float64{
			2, 0, 0,
			0, 4, 0,
			0, 0, 6,
		})
	})
}

func TestSparseDiagConversion(t *testing.T) {
	Convey("Given an array", t, func() {
		a := SparseDiag(3, 4, 1, 3, 5)

		Convey("Conversion to Dense works", func() {
			b := a.Dense()
			So(b.Dense().Shape(), ShouldResemble, []int{3, 4})
			So(b.Dense().Array(), ShouldResemble, []float64{
				1, 0, 0, 0,
				0, 3, 0, 0,
				0, 0, 5, 0,
			})
		})

		Convey("Conversion of transpose to Dense works", func() {
			b := a.T().Dense()
			So(b.Dense().Shape(), ShouldResemble, []int{4, 3})
			So(b.Dense().Array(), ShouldResemble, []float64{
				1, 0, 0,
				0, 3, 0,
				0, 0, 5,
				0, 0, 0,
			})
		})

		Convey("Conversion to sparse coo works", func() {
			b := a.SparseCoo()
			So(b.Dense().Shape(), ShouldResemble, []int{3, 4})
			So(b.Dense().Array(), ShouldResemble, []float64{
				1, 0, 0, 0,
				0, 3, 0, 0,
				0, 0, 5, 0,
			})
		})

		Convey("Conversion of transpose to sparse coo works", func() {
			b := a.T().SparseCoo()
			So(b.Dense().Shape(), ShouldResemble, []int{4, 3})
			So(b.Dense().Array(), ShouldResemble, []float64{
				1, 0, 0,
				0, 3, 0,
				0, 0, 5,
				0, 0, 0,
			})
		})

		Convey("Conversion to diag works", func() {
			b := a.SparseDiag()
			So(b.Dense().Shape(), ShouldResemble, []int{3, 4})
			So(b.Dense().Array(), ShouldResemble, []float64{
				1, 0, 0, 0,
				0, 3, 0, 0,
				0, 0, 5, 0,
			})
		})

		Convey("Conversion of transpose to diag works", func() {
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

func TestSparseDiagColColSetCols(t *testing.T) {
	Convey("Given an array", t, func() {
		a := Diag(1, 2, 3)

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

		Convey("ColSet panics", func() {
			So(func() { a.ColSet(-1, []float64{0, 1, 2}) }, ShouldPanic)
			So(func() { a.ColSet(3, []float64{0, 1, 2}) }, ShouldPanic)
			So(func() { a.ColSet(0, []float64{0, 1}) }, ShouldPanic)
			So(func() { a.ColSet(0, []float64{0, 1, 2}) }, ShouldPanic)
			So(func() { a.ColSet(0, []float64{0, 1, 2, 3}) }, ShouldPanic)
		})
	})
}

func TestSparseDiagDiag(t *testing.T) {
	Convey("Given an array", t, func() {
		a := Diag(1, 2, 3)

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

func TestSparseDiagDist(t *testing.T) {
	Convey("Given a SparseDiag matrix", t, func() {
		m := Diag(1, 2, -1).M()

		Convey("Invalid distance types panic", func() {
			So(func() { m.Dist(DistType(-1)) }, ShouldPanic)
		})

		Convey("Euclidean distance works", func() {
			d := m.Dist(EuclideanDist)
			So(d.Array(), ShouldResemble, []float64{
				0, math.Sqrt(5), math.Sqrt(2),
				math.Sqrt(5), 0, math.Sqrt(5),
				math.Sqrt(2), math.Sqrt(5), 0,
			})
		})
	})
}

func TestSparseDiagInverseNormLDivide(t *testing.T) {
	Convey("Given an invertible diagonal matrix", t, func() {
		m := Diag(2, 3, 5)

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
		m := Diag(2, 3, 5)

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
		a := Diag(4, 6, 8)
		b := Diag(2, 2, 2)

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

func TestSparseDiagItemMath(t *testing.T) {
	Convey("Given a diag array", t, func() {
		a := Diag(1, 2, 3)

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

func TestSparseDiagMaxMin(t *testing.T) {
	Convey("Given positive and negative diagonal arrays", t, func() {
		pos := Diag(1, 2, 3)
		neg := Diag(-1, -2, -3)

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

func TestSparseDiagRowRowSetRows(t *testing.T) {
	Convey("Given an array", t, func() {
		a := Diag(1, 2, 3)

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

		Convey("RowSet panics", func() {
			So(func() { a.RowSet(-1, []float64{0, 1, 2}) }, ShouldPanic)
			So(func() { a.RowSet(3, []float64{0, 1, 2}) }, ShouldPanic)
			So(func() { a.RowSet(0, []float64{0, 1}) }, ShouldPanic)
			So(func() { a.RowSet(0, []float64{0, 1, 2}) }, ShouldPanic)
			So(func() { a.RowSet(0, []float64{0, 1, 2, 3}) }, ShouldPanic)
		})
	})
}

func TestSparseDiagVisit(t *testing.T) {
	Convey("Given a sparse diag array", t, func() {
		a := SparseDiag(4, 3, 1.0, 2.0, 3.0)

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
				1, 0, 0,
				0, 2, 0,
				0, 0, 3,
				0, 0, 0,
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
				1, 0, 0,
				0, 2, 0,
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
			So(count, ShouldEqual, 3)
			So(saw.CountNonzero(), ShouldEqual, 3)
			So(b.Array(), ShouldResemble, []float64{
				1, 0, 0,
				0, 2, 0,
				0, 0, 3,
				0, 0, 0,
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
				if saw.CountNonzero() >= 2 {
					return false
				}
				return true
			})
			So(count, ShouldEqual, 2)
			So(saw.CountNonzero(), ShouldEqual, 2)
			So(b.Array(), ShouldResemble, []float64{
				1, 0, 0,
				0, 2, 0,
				0, 0, 0,
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
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
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
				if saw.CountNonzero() >= 6 {
					return false
				}
				return true
			})
			So(count, ShouldEqual, 6)
			So(saw.CountNonzero(), ShouldEqual, 6)
			So(b.Array(), ShouldResemble, []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 0, 0,
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
			So(count, ShouldEqual, 3)
			So(saw.CountNonzero(), ShouldEqual, 3)
			So(b.Array(), ShouldResemble, []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 3, 0,
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
				if saw.CountNonzero() >= 2 {
					return false
				}
				return true
			})
			So(count, ShouldEqual, 2)
			So(saw.CountNonzero(), ShouldEqual, 2)
			So(b.Array(), ShouldResemble, []float64{
				1, 0, 0, 0,
				0, 2, 0, 0,
				0, 0, 0, 0,
			})
		})
	})
}

func TestSparseDiagMatrix(t *testing.T) {
	Convey("Given a sparse matrix with shape 5, 3", t, func() {
		array := SparseDiag(5, 3)

		Convey("All is false", func() {
			So(array.All(), ShouldBeFalse)
		})

		Convey("Any is false", func() {
			So(array.Any(), ShouldBeFalse)
		})

		Convey("CountNonzero is correct", func() {
			So(array.CountNonzero(), ShouldEqual, 0)
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

		Convey("Item() panics given invalid input", func() {
			So(func() { array.Item(0) }, ShouldPanic)
			So(func() { array.Item(0, 0, 0) }, ShouldPanic)
			So(func() { array.Item(5, 0) }, ShouldPanic)
			So(func() { array.Item(0, 3) }, ShouldPanic)
		})

		Convey("ItemSet() panics given invalid input", func() {
			So(func() { array.ItemSet(1.0, 0) }, ShouldPanic)
			So(func() { array.ItemSet(1.0, 0, 0, 0) }, ShouldPanic)
			So(func() { array.ItemSet(1.0, 5, 0) }, ShouldPanic)
			So(func() { array.ItemSet(1.0, 0, 3) }, ShouldPanic)
		})

		Convey("Item() returns the right values", func() {
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
			array.ItemSet(1, 1, 1)
			array.ItemSet(2, 2, 2)

			Convey("Off-diagonal elements panic", func() {
				So(func() { array.ItemSet(2, 3, 2) }, ShouldPanic)
			})

			Convey("All is false", func() {
				So(array.All(), ShouldBeFalse)
			})

			Convey("Any is true", func() {
				So(array.Any(), ShouldBeTrue)
			})

			Convey("Ravel is correct", func() {
				r := array.Ravel()
				So(r.Shape(), ShouldResemble, []int{15})
				So(r.Array(), ShouldResemble, []float64{
					0, 0, 0,
					0, 1, 0,
					0, 0, 2,
					0, 0, 0,
					0, 0, 0,
				})
			})

			Convey("Slice works", func() {
				r := array.Slice([]int{1, 1}, []int{3, 3})
				So(r.Shape(), ShouldResemble, []int{2, 2})
				So(r.Array(), ShouldResemble, []float64{
					1, 0,
					0, 2,
				})
			})

			Convey("CountNonzero is correct", func() {
				So(array.CountNonzero(), ShouldEqual, 2)
			})

			Convey("Equal is correct", func() {
				So(array.Equal(SparseDiag(5, 3, 0, 1, 2)), ShouldBeTrue)
			})

			Convey("Item() returns updates", func() {
				for i0 := 0; i0 < 5; i0++ {
					for i1 := 0; i1 < 3; i1++ {
						if i0 == 1 && i1 == 1 {
							So(array.Item(i0, i1), ShouldEqual, 1)
						} else if i0 == 2 && i1 == 2 {
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
					case 1*3 + 1:
						So(array.FlatItem(i), ShouldEqual, 1)
					case 2*3 + 2:
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
							if i0 == 1 && i1 == 1 {
								So(norm.Item(i0, i1), ShouldEqual, 1.0/3)
							} else if i0 == 2 && i1 == 2 {
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
			array.FlatItemSet(1, 1*3+1)
			array.FlatItemSet(2, 2*3+2)

			Convey("Off-diagonal elements panic", func() {
				So(func() { array.FlatItemSet(3, 3*3+2) }, ShouldPanic)
			})

			Convey("All is false", func() {
				So(array.All(), ShouldBeFalse)
			})

			Convey("Any is true", func() {
				So(array.Any(), ShouldBeTrue)
			})

			Convey("Item() returns updates", func() {
				for i0 := 0; i0 < 5; i0++ {
					for i1 := 0; i1 < 3; i1++ {
						if i0 == 1 && i1 == 1 {
							So(array.Item(i0, i1), ShouldEqual, 1)
						} else if i0 == 2 && i1 == 2 {
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
					case 1*3 + 1:
						So(array.FlatItem(i), ShouldEqual, 1)
					case 2*3 + 2:
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

	Convey("Given two 2D sparse matrices of equal length", t, func() {
		a1 := SparseDiag(4, 3, 1.0, 1.0, 1.0)
		a2 := SparseDiag(4, 3, 2.0, 2.0, 2.0)

		Convey("Concat works along axis 0", func() {
			a3 := a1.Concat(0, a2)
			So(a3.Shape(), ShouldResemble, []int{8, 3})
			for i0 := 0; i0 < 8; i0++ {
				for i1 := 0; i1 < 3; i1++ {
					if i0 == i1 {
						So(a3.Item(i0, i1), ShouldEqual, 1)
					} else if i0-4 == i1 {
						So(a3.Item(i0, i1), ShouldEqual, 2)
					} else {
						So(a3.Item(i0, i1), ShouldEqual, 0)
					}
				}
			}
		})

		Convey("Concat works along axis 1", func() {
			a3 := a1.Concat(1, a2)
			So(a3.Shape(), ShouldResemble, []int{4, 6})
			for i0 := 0; i0 < 4; i0++ {
				for i1 := 0; i1 < 6; i1++ {
					if i0 == i1 && i0 < 3 {
						So(a3.Item(i0, i1), ShouldEqual, 1)
					} else if i0 == i1-3 {
						So(a3.Item(i0, i1), ShouldEqual, 2)
					} else {
						So(a3.Item(i0, i1), ShouldEqual, 0)
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
						if i0 == i1 {
							if i2 == 0 {
								So(a3.Item(i0, i1, i2), ShouldEqual, 1)
							} else {
								So(a3.Item(i0, i1, i2), ShouldEqual, 2)
							}
						} else {
							So(a3.Item(i0, i1, i2), ShouldEqual, 0)
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
