// The matrix package contains various utilities for dealing with raw matrices.
// The interface is loosely based on the NumPy package in Python.
//
// The key interfaces here are:
//
// * NDArray – A multidimensional array, with dense and (2D-only) sparse
//   implementations.
// * Matrix – A two-dimensional array, with various methods only available when
//   working in two dimensions.  A two-dimensional NDArray can be
//   trivially converted to the Matrix type by calling arr.M().
//
// When possible, function implementations take advantage of matrix sparsity.
// For instance, MProd(), the matrix multiplication function, performs the
// minimum amount of work required based on the types of its arguments.
//
// Certain linear algebra methods, particularly in the Matrix interface, rely
// on BLAS. In order to use it, you will need to register an appropriate engine.
// See the documentation at https://github.com/gonum/blas for details. You can
// register a default (native Go) engine by calling InitDefaultBlas().
// If you see a panic message like
// "mat64: no blas engine registered: call Register()"
// then you need to register a BLAS engine. If you don't see this error, then
// you probably don't need to worry about it.
package matrix

import (
	"fmt"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

// ArraySparsity indicates the representation type of the matrix
type ArraySparsity int

const (
	DenseArray ArraySparsity = iota
	SparseCooMatrix
	SparseDiagMatrix
)

// A NDArray is an n-dimensional array of numbers which can be manipulated in
// various ways. Concrete implementations can differ; for instance, sparse
// and dense representations are possible.
type NDArray interface {

	// Return the element-wise sum of this array and one or more others
	Add(others ...NDArray) NDArray

	// Returns true if and only if all items are nonzero
	All() bool

	// Returns true if f is true for all array elements
	AllF(f func(v float64) bool) bool

	// Returns true if f is true for all pairs of array elements in the same position
	AllF2(f func(v1, v2 float64) bool, other NDArray) bool

	// Returns true if and only if any item is nonzero
	Any() bool

	// Returns true if f is true for any array element
	AnyF(f func(v float64) bool) bool

	// Returns true if f is true for any pair of array elements in the same position
	AnyF2(f func(v1, v2 float64) bool, other NDArray) bool

	// Return the result of applying a function to all elements
	Apply(f func(float64) float64) NDArray

	// Get the matrix data as a flattened 1D array; sparse matrices will make
	// a copy first.
	Array() []float64

	// Create a new array by concatenating this with another array along the
	// specified axis. The array shapes must be equal along all other axes.
	// It is legal to add a new axis.
	Concat(axis int, others ...NDArray) NDArray

	// Returns a duplicate of this array
	Copy() NDArray

	// Counts the number of nonzero elements in the array
	CountNonzero() int

	// Returns a dense copy of the array
	Dense() NDArray

	// Return the element-wise quotient of this array and one or more others.
	// This function defines 0 / 0 = 0, so it's useful for sparse arrays.
	Div(others ...NDArray) NDArray

	// Returns true if and only if all elements in the two arrays are equal
	Equal(other NDArray) bool

	// Set all array elements to the given value
	Fill(value float64)

	// Get an array element in a flattened verison of this array
	FlatItem(index int) float64

	// Set an array element in a flattened version of this array
	FlatItemSet(value float64, index int)

	// Return an iterator over populated matrix entries
	FlatIter() FlatNDArrayIterator

	// Get an array element
	Item(index ...int) float64

	// Return the result of adding a scalar value to each array element
	ItemAdd(value float64) NDArray

	// Return the result of dividing each array element by a scalar value
	ItemDiv(value float64) NDArray

	// Return the reuslt of multiplying each array element by a scalar value
	ItemProd(value float64) NDArray

	// Return the result of subtracting a scalar value from each array element
	ItemSub(value float64) NDArray

	// Set an array element
	ItemSet(value float64, index ...int)

	// Return an iterator over populated matrix entries
	Iter() CoordNDArrayIterator

	// Returns the array as a matrix. This is only possible for 1D and 2D arrays;
	// 1D arrays of length n are converted into n x 1 vectors.
	M() Matrix

	// Get the value of the largest array element
	Max() float64

	// Get the value of the smallest array element
	Min() float64

	// Return the element-wise product of this array and one or more others
	Prod(others ...NDArray) NDArray

	// The number of dimensions in the matrix
	NDim() int

	// Return a copy of the array, normalized to sum to 1
	Normalize() NDArray

	// Get a 1D copy of the array, in 'C' order: rightmost axes change fastest
	Ravel() NDArray

	// A slice giving the size of all array dimensions
	Shape() []int

	// The total number of elements in the matrix
	Size() int

	// Get an array containing a rectangular slice of this array.
	// `from` and `to` should both have one index per axis. The indices
	// in `from` and `to` define the first and just-past-last indices you wish
	// to select along each axis. Negative indexing is supported: when slicing,
	// index -1 refers to the item just past the last and -arr.Size() refers to
	// the first element.
	Slice(from []int, to []int) NDArray

	// Ask whether the matrix has a sparse representation (useful for optimization)
	Sparsity() ArraySparsity

	// Return the element-wise difference of this array and one or more others
	Sub(others ...NDArray) NDArray

	// Return the sum of all array elements
	Sum() float64

	// Return the same matrix, but with axes transposed. The same data is used,
	// for speed and memory efficiency. Use Copy() to create a new array.
	// A 1D array is unchanged; create a 2D analog to rotate a vector.
	T() NDArray
}

// Create an array from literal data
func A(shape []int, values ...float64) NDArray {
	size := 1
	for _, sz := range shape {
		size *= sz
	}
	if len(values) != size {
		panic(fmt.Sprintf("Expected %d array elements but got %d", size, len(values)))
	}
	array := &DenseF64Array{
		shape: shape,
		array: make([]float64, len(values)),
	}
	copy(array.array[:], values[:])
	return array
}

// Create a 1D array
func A1(values ...float64) NDArray {
	return A([]int{len(values)}, values...)
}

// Create a 2D array
func A2(values ...[]float64) NDArray {
	array := &DenseF64Array{
		shape: []int{len(values), len(values[0])},
		array: make([]float64, len(values)*len(values[0])),
	}
	for i0 := 0; i0 < array.shape[0]; i0++ {
		if len(values[i0]) != array.shape[1] {
			panic(fmt.Sprintf("A2 got inconsistent array lengths %d and %d", array.shape[1], len(values[i0])))
		}
		for i1 := 0; i1 < array.shape[1]; i1++ {
			array.ItemSet(values[i0][i1], i0, i1)
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

// Create a dense NDArray of float64 values, initialized to uniformly random
// values in [0, 1).
func Rand(size ...int) NDArray {
	array := Dense(size...)

	max := array.Size()
	for i := 0; i < max; i++ {
		array.FlatItemSet(rand.Float64(), i)
	}

	return array
}

// Create a dense NDArray of float64 values, initialized to random values in
// [-math.MaxFloat64, +math.MaxFloat64] distributed on the standard Normal
// distribution.
func RandN(size ...int) NDArray {
	array := Dense(size...)

	max := array.Size()
	for i := 0; i < max; i++ {
		array.FlatItemSet(rand.NormFloat64(), i)
	}

	return array
}
