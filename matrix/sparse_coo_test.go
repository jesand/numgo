package matrix

import (
	. "github.com/smartystreets/goconvey/convey"
	"testing"
)

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
