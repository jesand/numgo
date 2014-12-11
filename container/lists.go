package container

// A container of generic values
type List interface {

	// Add one or more items to the container
	Push(value ...interface{})

	// Remove an item from the container, or return nil
	Pop() interface{}

	// Ask how many items are in the container
	Size() int
}

// A stack is a list of anything
type Stack []interface{}

// Push one or more items onto the stack
func (s *Stack) Push(value ...interface{}) {
	*s = append(*s, value...)
}

// Pop an item from the stack in LIFO order
func (s *Stack) Pop() interface{} {
	if len(*s) > 0 {
		v := (*s)[len(*s)-1]
		*s = (*s)[:len(*s)-1]
		return v
	} else {
		return nil
	}
}

// Ask how many items are in the container
func (s Stack) Size() int {
	return len(s)
}

// A queue is a list of anything
type Queue []interface{}

// Push one or more items onto the queue
func (q *Queue) Push(value ...interface{}) {
	*q = append(*q, value...)
}

// Pop an item from the queue in FIFO order
func (q *Queue) Pop() interface{} {
	if len(*q) > 0 {
		v := (*q)[0]
		*q = (*q)[1:]
		return v
	} else {
		return nil
	}
}

// Ask how many items are in the container
func (s Queue) Size() int {
	return len(s)
}
