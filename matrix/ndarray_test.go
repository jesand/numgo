package matrix

import (
	. "github.com/smartystreets/goconvey/convey"
	"testing"
)

func TestA(t *testing.T) {
	Convey("A() panics when data doesn't match dimensions", t, func() {
		So(func() { A([]int{5}) }, ShouldPanic)
		So(func() { A([]int{5}, 1, 2, 3, 4) }, ShouldPanic)
		So(func() { A([]int{5}, 1, 2, 3, 4, 5, 6) }, ShouldPanic)
		So(func() { A([]int{2, 3}, 1, 2, 3, 4, 5) }, ShouldPanic)
		So(func() { A([]int{2, 3}, 1, 2, 3, 4, 5, 6, 7) }, ShouldPanic)
	})

	Convey("Given a 1D array created with A", t, func() {
		m := A([]int{5}, 1, 2, 3, 4, 5)
		Convey("Shape() is 5", func() {
			So(m.Shape(), ShouldResemble, []int{5})
		})
		Convey("Size() is 5", func() {
			So(m.Size(), ShouldResemble, 5)
		})
		Convey("The data is correct", func() {
			So(m.Array(), ShouldResemble, []float64{1, 2, 3, 4, 5})
		})
	})

	Convey("Given a 2D array created with A", t, func() {
		m := A([]int{2, 3}, 1, 2, 3, 4, 5, 6)
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

func TestA1(t *testing.T) {
	Convey("Given a 1D array created with A1", t, func() {
		m := A1(1, 2, 3, 4, 5)
		Convey("Shape() is 5", func() {
			So(m.Shape(), ShouldResemble, []int{5})
		})
		Convey("Size() is 5", func() {
			So(m.Size(), ShouldResemble, 5)
		})
		Convey("The data is correct", func() {
			So(m.Array(), ShouldResemble, []float64{1, 2, 3, 4, 5})
		})
	})
}

func TestA2(t *testing.T) {
	Convey("A2 panics given arrays of differing lengths", t, func() {
		So(func() { A2([]float64{1}, []float64{1, 2}) }, ShouldPanic)
	})

	Convey("Given a 2D array created with A2", t, func() {
		m := A2([]float64{1, 2, 3}, []float64{4, 5, 6})
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

func TestDense(t *testing.T) {
	Convey("Given a dense array with shape 5, 3", t, func() {
		array := Dense(5, 3)

		Convey("Shape() is (5, 3)", func() {
			So(array.Shape(), ShouldResemble, []int{5, 3})
		})

		Convey("Size() is 15", func() {
			So(array.Size(), ShouldEqual, 15)
		})

		Convey("All values are zero", func() {
			for i := 0; i < array.Size(); i++ {
				So(array.FlatItem(i), ShouldEqual, 0)
			}
		})
	})
}

func TestOnes(t *testing.T) {
	Convey("Given a ones array with shape 5, 3", t, func() {
		array := Ones(5, 3)

		Convey("Shape() is (5, 3)", func() {
			So(array.Shape(), ShouldResemble, []int{5, 3})
		})

		Convey("Size() is 15", func() {
			So(array.Size(), ShouldEqual, 15)
		})

		Convey("All values are one", func() {
			for i := 0; i < array.Size(); i++ {
				So(array.FlatItem(i), ShouldEqual, 1)
			}
		})
	})
}

func TestRand(t *testing.T) {
	Convey("Given a random array with shape 5, 3", t, func() {
		array := Rand(5, 3)

		Convey("Shape() is (5, 3)", func() {
			So(array.Shape(), ShouldResemble, []int{5, 3})
		})

		Convey("Size() is 15", func() {
			So(array.Size(), ShouldEqual, 15)
		})

		Convey("All values are in (0, 1)", func() {
			for i := 0; i < array.Size(); i++ {
				So(array.FlatItem(i), ShouldBeBetween, 0, 1)
			}
		})
	})
}

func TestRandN(t *testing.T) {
	Convey("Given a random normal array with shape 5, 3", t, func() {
		array := RandN(5, 3)

		Convey("Shape() is (5, 3)", func() {
			So(array.Shape(), ShouldResemble, []int{5, 3})
		})

		Convey("Size() is 15", func() {
			So(array.Size(), ShouldEqual, 15)
		})

		Convey("All is true", func() {
			// There's some small chance of this being false; a one-off failure is ok
			So(array.All(), ShouldBeTrue)
		})
	})
}

func TestWithValue(t *testing.T) {
	Convey("Given a WithValue array with shape 5, 3", t, func() {
		array := WithValue(3.5, 5, 3)

		Convey("Shape() is (5, 3)", func() {
			So(array.Shape(), ShouldResemble, []int{5, 3})
		})

		Convey("Size() is 15", func() {
			So(array.Size(), ShouldEqual, 15)
		})

		Convey("All values are 3.5", func() {
			for i := 0; i < array.Size(); i++ {
				So(array.FlatItem(i), ShouldEqual, 3.5)
			}
		})
	})
}

func TestZeros(t *testing.T) {
	Convey("Given a zeros array with shape 5, 3", t, func() {
		array := Zeros(5, 3)

		Convey("Shape() is (5, 3)", func() {
			So(array.Shape(), ShouldResemble, []int{5, 3})
		})

		Convey("Size() is 15", func() {
			So(array.Size(), ShouldEqual, 15)
		})

		Convey("All values are zero", func() {
			for i := 0; i < array.Size(); i++ {
				So(array.FlatItem(i), ShouldEqual, 0)
			}
		})
	})
}
