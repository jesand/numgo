package matrix

import (
	. "github.com/smartystreets/goconvey/convey"
	"testing"
)

// The smallest floating point difference permissible for near-equality
const Eps float64 = 1e-9

func Test1DArray(t *testing.T) {
	Convey("Given an array with shape 5", t, func() {
		array, err := Dense(5)
		So(err, ShouldBeNil)

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
			array.Normalize()

			Convey("Item() returns all zeros", func() {
				for i0 := 0; i0 < 5; i0++ {
					So(array.Item(i0), ShouldEqual, 0)
				}
			})
		})

		Convey("When I call ItemSet", func() {
			array.ItemSet(1, 1)
			array.ItemSet(3, 3)

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

			Convey("Sum() is correct", func() {
				So(array.Sum(), ShouldEqual, 4)
			})

			Convey("When I call Normalize", func() {
				array.Normalize()

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

		Convey("When I call Fill", func() {
			array.Fill(2)

			Convey("Item() returns updates", func() {
				for i0 := 0; i0 < 5; i0++ {
					So(array.Item(i0), ShouldEqual, 2)
				}
			})

			Convey("When I call Normalize", func() {
				array.Normalize()

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
}

func Test2DArray(t *testing.T) {
	Convey("Given an array with shape 5, 3", t, func() {
		array, err := Dense(5, 3)
		So(err, ShouldBeNil)

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
			array.Normalize()
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

			Convey("Sum() is correct", func() {
				So(array.Sum(), ShouldEqual, 3)
			})

			Convey("When I call Normalize", func() {
				array.Normalize()

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

		Convey("When I call Fill", func() {
			array.Fill(3)

			Convey("Item() returns updates", func() {
				for i0 := 0; i0 < 5; i0++ {
					for i1 := 0; i1 < 3; i1++ {
						So(array.Item(i0, i1), ShouldEqual, 3)
					}
				}
			})

			Convey("When I call Normalize", func() {
				array.Normalize()

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
}

func Test3DArray(t *testing.T) {
	Convey("Given an array with shape 5, 3, 2", t, func() {
		array, err := Dense(5, 3, 2)
		So(err, ShouldBeNil)

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
			array.Normalize()
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

			Convey("Sum() is correct", func() {
				So(array.Sum(), ShouldEqual, 6)
			})

			Convey("When I call Normalize", func() {
				array.Normalize()

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

		Convey("When I call Fill", func() {
			array.Fill(0.5)

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
				array.Normalize()

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
}

func Test4DArray(t *testing.T) {
	Convey("Given an array with shape 5, 3, 2, 3", t, func() {
		_, err := Dense(5, 3, 2, 3)
		Convey("An error is returned", func() {
			So(err, ShouldNotBeNil)
		})
	})
}
