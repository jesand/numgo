package container

import (
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestStack(t *testing.T) {
	var (
		s      Stack
		first  int
		second int
		third  int
		fourth interface{}
	)
	Convey("Given an uninitialized stack", t, func() {
		s = nil

		Convey("Then the stack is empty", func() {
			So(len(s), ShouldEqual, 0)
		})

		Convey("When I push 1, 2, and 3 individually", func() {
			s.Push(1)
			s.Push(2)
			s.Push(3)

			Convey("Then the stack length is 3", func() {
				So(len(s), ShouldEqual, 3)
				So(s.Size(), ShouldEqual, 3)
			})

			Convey("When I pop three numbers", func() {
				first, _ = s.Pop().(int)
				second, _ = s.Pop().(int)
				third, _ = s.Pop().(int)

				Convey("Then I get 3, 2, and 1", func() {
					So(first, ShouldEqual, 3)
					So(second, ShouldEqual, 2)
					So(third, ShouldEqual, 1)
				})

				Convey("Then the stack is empty", func() {
					So(len(s), ShouldEqual, 0)
				})

				Convey("When I pop again", func() {
					fourth, _ = s.Pop().(int)

					Convey("Then I get a zero value", func() {
						So(fourth, ShouldBeZeroValue)
					})

					Convey("Then the stack is empty", func() {
						So(len(s), ShouldEqual, 0)
					})
				})
			})
		})

		Convey("When I push 1, 2, and 3 all at once", func() {
			s.Push(1, 2, 3)

			Convey("Then the stack length is 3", func() {
				So(len(s), ShouldEqual, 3)
			})

			Convey("When I pop three numbers", func() {
				first, _ = s.Pop().(int)
				second, _ = s.Pop().(int)
				third, _ = s.Pop().(int)

				Convey("Then I get 3, 2, and 1", func() {
					So(first, ShouldEqual, 3)
					So(second, ShouldEqual, 2)
					So(third, ShouldEqual, 1)
				})

				Convey("Then the stack is empty", func() {
					So(len(s), ShouldEqual, 0)
				})

				Convey("When I pop again", func() {
					fourth, _ = s.Pop().(int)

					Convey("Then I get a zero value", func() {
						So(fourth, ShouldBeZeroValue)
					})

					Convey("Then the stack is empty", func() {
						So(len(s), ShouldEqual, 0)
					})
				})
			})
		})
	})
}

func TestQueue(t *testing.T) {
	var (
		q      Queue
		first  int
		second int
		third  int
		fourth interface{}
	)
	Convey("Given an uninitialized queue", t, func() {
		q = nil

		Convey("Then the queue is empty", func() {
			So(len(q), ShouldEqual, 0)
		})

		Convey("When I push 1, 2, and 3 individually", func() {
			q.Push(1)
			q.Push(2)
			q.Push(3)

			Convey("Then the queue length is 3", func() {
				So(len(q), ShouldEqual, 3)
				So(q.Size(), ShouldEqual, 3)
			})

			Convey("When I pop three numbers", func() {
				first, _ = q.Pop().(int)
				second, _ = q.Pop().(int)
				third, _ = q.Pop().(int)

				Convey("Then I get 1, 2, and 3", func() {
					So(first, ShouldEqual, 1)
					So(second, ShouldEqual, 2)
					So(third, ShouldEqual, 3)
				})

				Convey("Then the queue is empty", func() {
					So(len(q), ShouldEqual, 0)
				})

				Convey("When I pop again", func() {
					fourth, _ = q.Pop().(int)

					Convey("Then I get a zero value", func() {
						So(fourth, ShouldBeZeroValue)
					})

					Convey("Then the queue is empty", func() {
						So(len(q), ShouldEqual, 0)
					})
				})
			})
		})

		Convey("When I push 1, 2, and 3 all at once", func() {
			q.Push(1, 2, 3)

			Convey("Then the queue length is 3", func() {
				So(len(q), ShouldEqual, 3)
			})

			Convey("When I pop three numbers", func() {
				first, _ = q.Pop().(int)
				second, _ = q.Pop().(int)
				third, _ = q.Pop().(int)

				Convey("Then I get 1, 2, and 3", func() {
					So(first, ShouldEqual, 1)
					So(second, ShouldEqual, 2)
					So(third, ShouldEqual, 3)
				})

				Convey("Then the queue is empty", func() {
					So(len(q), ShouldEqual, 0)
				})

				Convey("When I pop again", func() {
					fourth, _ = q.Pop().(int)

					Convey("Then I get a zero value", func() {
						So(fourth, ShouldBeZeroValue)
					})

					Convey("Then the queue is empty", func() {
						So(len(q), ShouldEqual, 0)
					})
				})
			})
		})
	})
}
