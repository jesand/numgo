// The matrix package contains various utilities for dealing with raw matrices.
// The interface is loosely based on the NumPy package in Python.
package matrix

import (
	"fmt"
	"math/rand"
)

// A NDArray is an n-dimensional array of numbers which can be manipulated in
// various ways. Concrete implementations can differ; for instance, sparse
// and dense representations are possible.
type NDArray interface {

	// Return the element-wise sum of this array and one or more others
	Add(others ...NDArray) NDArray

	// Returns true if and only if all items are nonzero
	All() bool

	// Returns true if and only if any item is nonzero
	Any() bool

	// Apply a function to all elements
	Apply(f func(float64) float64)

	// Returns the array as a matrix. This is only possible for 1D and 2D arrays;
	// 1D arrays of length n are converted into n x 1 vectors.
	ToMatrix() Matrix

	// Create a new array by concatenating this with another array along the
	// specified axis. The array shapes must be equal along all other axes.
	// It is legal to add a new axis.
	Concat(axis int, others ...NDArray) NDArray

	// Returns a duplicate of this array
	Copy() NDArray

	// Return the element-wise quotient of this array and one or more others
	Div(others ...NDArray) NDArray

	// Returns true if and only if all elements in the two arrays are equal
	Equal(other NDArray) bool

	// Set all array elements to the given value
	Fill(value float64)

	// Get an array element in a flattened verison of this array
	FlatItem(index int) float64

	// Set an array element in a flattened version of this array
	FlatItemSet(value float64, index int)

	// Get an array element
	Item(index ...int) float64

	// Add a scalar value to each array element
	ItemAdd(value float64)

	// Divide each array element by a scalar value
	ItemDiv(value float64)

	// Multiply each array element by a scalar value
	ItemProd(value float64)

	// Subtract a scalar value from each array element
	ItemSub(value float64)

	// Set an array element
	ItemSet(value float64, index ...int)

	// Get the value of the largest array element
	Max() float64

	// Get the value of the smallest array element
	Min() float64

	// Return the element-wise product of this array and one or more others
	Prod(others ...NDArray) NDArray

	// The number of dimensions in the matrix
	NDim() int

	// Normalize the array to sum to 1, or do nothing if all items are 0.
	Normalize()

	// Get a 1D copy of the array, in 'C' order: rightmost axes change fastest
	Ravel() NDArray

	// A slice giving the size of all array dimensions
	Shape() []int

	// The total number of elements in the matrix
	Size() int

	// Get an array containing a rectangular slice of this array.
	// `from` and `to` should both have one index per axis. The indices
	// in `from` and `to` define the first and just-past-last indices you wish
	// to select along each axis.
	Slice(from []int, to []int) NDArray

	// Return the element-wise difference of this array and one or more others
	Sub(others ...NDArray) NDArray

	// Return the sum of all array elements
	Sum() float64

	// Return the same matrix, but with axes transposed. The same data is used,
	// for speed and memory efficiency. Use Copy() to create a new array.
	// A 1D array is unchanged; create a 2D analog to rotate a vector.
	Transpose() NDArray
}

// Create an array from literal data
func A(shape []int, array ...float64) NDArray {
	size := 1
	for _, sz := range shape {
		size *= sz
	}
	if len(array) != size {
		panic(fmt.Sprintf("Expected %d array elements but got %d", size, len(array)))
	}
	return &DenseF64Array{
		shape: shape,
		array: array,
	}
}

// Create a 1D array
func A1(value []float64) NDArray {
	array := &DenseF64Array{
		shape: []int{len(value)},
		array: make([]float64, len(value)),
	}
	copy(array.array[:], value[:])
	return array
}

// Create a 2D array
func A2(value [][]float64) NDArray {
	array := &DenseF64Array{
		shape: []int{len(value), len(value[0])},
		array: make([]float64, len(value)*len(value[0])),
	}
	for i0 := 0; i0 < array.shape[0]; i0++ {
		for i1 := 0; i1 < array.shape[1]; i1++ {
			array.ItemSet(value[i0][i1], i0, i1)
		}
	}
	return array
}

// Create an NDArray of float64 values, initialized to zero
func Dense(size ...int) NDArray {
	totalSize := 1
	for _, sz := range size {
		totalSize *= sz
	}
	return &DenseF64Array{
		shape: size,
		array: make([]float64, totalSize),
	}
}

// Create an NDArray of float64 values, initialized to value
func WithValue(value float64, size ...int) NDArray {
	array := Dense(size...)
	array.Fill(value)
	return array
}

// Create an NDArray of float64 values, initialized to zero
func Zeros(size ...int) NDArray {
	return Dense(size...)
}

// Create an NDArray of float64 values, initialized to one
func Ones(size ...int) NDArray {
	return WithValue(1.0, size...)
}

// Create an identity matrix of the specified dimensionality. If a single size
// is used, a size x size identity matrix will be created.
func I(size ...int) NDArray {
	if len(size) == 1 {
		size = []int{size[0], size[0]}
	}
	array := Dense(size...)

	index := make([]int, len(size))
	for i := 0; i < size[0]; i++ {
		skip := false
		for j := range index {
			index[j] = i
			if i >= size[j] {
				skip = true
			}
		}
		if !skip {
			array.ItemSet(1.0, index...)
		}
	}

	return array
}

// Create a dense NDArray of float64 values, initialized to random values
func Rand(size ...int) NDArray {
	array := Dense(size...)

	max := array.Size()
	for i := 0; i < max; i++ {
		array.FlatItemSet(rand.Float64(), i)
	}

	return array
}
