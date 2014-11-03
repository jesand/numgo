package matrix

import (
	. "github.com/smartystreets/goconvey/convey"
	"testing"
)

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

func TestI(t *testing.T) {
	Convey("Given an identity array with shape 3, 3", t, func() {
		array := I(3, 3)

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

	Convey("Given an identity array with shape 3, 5", t, func() {
		array := I(3, 5)

		Convey("Shape() is (3, 5)", func() {
			So(array.Shape(), ShouldResemble, []int{3, 5})
		})

		Convey("Size() is 15", func() {
			So(array.Size(), ShouldEqual, 15)
		})

		Convey("Only diagonal values are one; others are zero", func() {
			for i0 := 0; i0 < 3; i0++ {
				for i1 := 0; i1 < 5; i1++ {
					if i0 == i1 {
						So(array.Item(i0, i1), ShouldEqual, 1)
					} else {
						So(array.Item(i0, i1), ShouldEqual, 0)
					}
				}
			}
		})
	})

	Convey("Given an identity array with shape 5, 3", t, func() {
		array := I(5, 3)

		Convey("Shape() is (5,3)", func() {
			So(array.Shape(), ShouldResemble, []int{5, 3})
		})

		Convey("Size() is 15", func() {
			So(array.Size(), ShouldEqual, 15)
		})

		Convey("Only diagonal values are one; others are zero", func() {
			for i0 := 0; i0 < 5; i0++ {
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
