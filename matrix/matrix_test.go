package matrix

import (
	. "github.com/smartystreets/goconvey/convey"
	"testing"
)

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

func TestInverse(t *testing.T) {
	Convey("Given an invertable square matrix", t, func() {
		m := A2(
			[]float64{4, 7},
			[]float64{2, 6},
		).M()

		Convey("When I take the inverse", func() {
			mi := m.Inverse()

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
}

func TestLDivide(t *testing.T) {
	Convey("Given a simple division problem", t, func() {
		a := A2(
			[]float64{8, 1, 6},
			[]float64{3, 5, 7},
			[]float64{4, 9, 2},
		).M()
		b := A2(
			[]float64{15},
			[]float64{15},
			[]float64{15},
		).M()

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
}
