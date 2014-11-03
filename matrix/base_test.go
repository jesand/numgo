package matrix

import (
	. "github.com/smartystreets/goconvey/convey"
	"testing"
)

func TestIndexing(t *testing.T) {
	Convey("Given a random array of shape (5,3,2)", t, func() {
		array := Rand(5, 3, 2)
		Convey("ndToFlat works with positive indices", func() {
			next := 0
			for i0 := 0; i0 < 5; i0++ {
				for i1 := 0; i1 < 3; i1++ {
					for i2 := 0; i2 < 2; i2++ {
						So(ndToFlat(array.Shape(), []int{i0, i1, i2}), ShouldEqual, next)
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
						So(flatToNd(array.Shape(), next), ShouldResemble, []int{i0, i1, i2})
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
						So(ndToFlat(array.Shape(), []int{i0, i1, i2}), ShouldEqual, next)
						next++
					}
				}
			}
		})

		Convey("flatToNd works with negative indices", func() {
			next := -array.Size()
			for i0 := 0; i0 < 5; i0++ {
				for i1 := 0; i1 < 3; i1++ {
					for i2 := 0; i2 < 2; i2++ {
						So(flatToNd(array.Shape(), next), ShouldResemble, []int{i0, i1, i2})
						next++
					}
				}
			}
		})
	})
}
