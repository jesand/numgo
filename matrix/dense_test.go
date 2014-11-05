package matrix

import (
	. "github.com/smartystreets/goconvey/convey"
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

			Convey("When I call ItemAdd", func() {
				array = array.ItemAdd(0.5)
				Convey("Item() returns updates", func() {
					for i0 := 0; i0 < 5; i0++ {
						for i1 := 0; i1 < 3; i1++ {
							for i2 := 0; i2 < 2; i2++ {
								if i0 == 1 && i1 == 0 && i2 == 0 {
									So(array.Item(i0, i1, i2), ShouldEqual, 1.5)
								} else if i0 == 3 && i1 == 2 && i2 == 0 {
									So(array.Item(i0, i1, i2), ShouldEqual, 2.5)
								} else if i0 == 3 && i1 == 2 && i2 == 1 {
									So(array.Item(i0, i1, i2), ShouldEqual, 3.5)
								} else {
									So(array.Item(i0, i1, i2), ShouldEqual, 0.5)
								}
							}
						}
					}
				})
			})

			Convey("When I call ItemProd", func() {
				array = array.ItemProd(2)
				Convey("Item() returns updates", func() {
					for i0 := 0; i0 < 5; i0++ {
						for i1 := 0; i1 < 3; i1++ {
							for i2 := 0; i2 < 2; i2++ {
								if i0 == 1 && i1 == 0 && i2 == 0 {
									So(array.Item(i0, i1, i2), ShouldEqual, 2)
								} else if i0 == 3 && i1 == 2 && i2 == 0 {
									So(array.Item(i0, i1, i2), ShouldEqual, 4)
								} else if i0 == 3 && i1 == 2 && i2 == 1 {
									So(array.Item(i0, i1, i2), ShouldEqual, 6)
								} else {
									So(array.Item(i0, i1, i2), ShouldEqual, 0)
								}
							}
						}
					}
				})
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
