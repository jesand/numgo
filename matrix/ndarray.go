// The matrix package contains various utilities for dealing with raw matrices.
// The interface is loosely based on the NumPy package in Python.
package matrix

import (
	"errors"
)

// A NDArray is an n-dimensional array of numbers which can be manipulated in
// various ways. Concrete implementations can differ; for instance, sparse
// and dense representations are possible.
type NDArray interface {

	// The total number of elements in the matrix
	Size() int

	// The number of dimensions in the matrix
	NDim() int

	// A slice giving the size of all array dimensions
	Shape() []int

	// Get an array element
	Item(index ...int) float64

	// Set an array element
	ItemSet(value float64, index ...int)

	// Set all array elements to the given value
	Fill(value float64)

	// Return the sum of all array elements
	Sum() float64

	// Normalize the array to sum to 1, or do nothing if all items are 0.
	Normalize()
}

// A one-dimensional NDArray with dense representation
type Dense1DArray []float64

// The total number of elements in the matrix
func (array Dense1DArray) Size() int {
	return len(array)
}

// The number of dimensions in the matrix
func (array Dense1DArray) NDim() int {
	return 1
}

// A slice giving the size of all array dimensions
func (array Dense1DArray) Shape() []int {
	return []int{len(array)}
}

// Get an array element
func (array Dense1DArray) Item(index ...int) float64 {
	return array[index[0]]
}

// Set an array element
func (array Dense1DArray) ItemSet(value float64, index ...int) {
	array[index[0]] = value
}

// Set all array elements to the given value
func (array Dense1DArray) Fill(value float64) {
	for idx := range array {
		array[idx] = value
	}
}

// Return the sum of all array elements
func (array Dense1DArray) Sum() float64 {
	var result float64
	for _, v := range array {
		result += v
	}
	return result
}

// Normalize the array to sum to 1, or do nothing if all items are 0.
func (array *Dense1DArray) Normalize() {
	s := array.Sum()
	if s != 0 && s != 1 {
		for idx := range *array {
			(*array)[idx] /= s
		}
	}
}

// A two-dimensional NDArray with dense representation
type Dense2DArray [][]float64

// The total number of elements in the matrix
func (array Dense2DArray) Size() int {
	return len(array) * len(array[0])
}

// The number of dimensions in the matrix
func (array Dense2DArray) NDim() int {
	return 2
}

// A slice giving the size of all array dimensions
func (array Dense2DArray) Shape() []int {
	return []int{len(array), len(array[0])}
}

// Get an array element
func (array Dense2DArray) Item(index ...int) float64 {
	return array[index[0]][index[1]]
}

// Set an array element
func (array Dense2DArray) ItemSet(value float64, index ...int) {
	array[index[0]][index[1]] = value
}

// Set all array elements to the given value
func (array Dense2DArray) Fill(value float64) {
	for i0 := range array {
		for i1 := range array[i0] {
			array[i0][i1] = value
		}
	}
}

// Return the sum of all array elements
func (array Dense2DArray) Sum() float64 {
	var result float64
	for i0 := range array {
		for _, v := range array[i0] {
			result += v
		}
	}
	return result
}

// Normalize the array to sum to 1, or do nothing if all items are 0.
func (array *Dense2DArray) Normalize() {
	s := array.Sum()
	if s != 0 && s != 1 {
		for i0 := range *array {
			for i1 := range (*array)[i0] {
				(*array)[i0][i1] /= s
			}
		}
	}
}

// A three-dimensional NDArray with dense representation
type Dense3DArray [][][]float64

// The total number of elements in the matrix
func (array Dense3DArray) Size() int {
	return len(array) * len(array[0]) * len(array[0][0])
}

// The number of dimensions in the matrix
func (array Dense3DArray) NDim() int {
	return 3
}

// A slice giving the size of all array dimensions
func (array Dense3DArray) Shape() []int {
	return []int{len(array), len(array[0]), len(array[0][0])}
}

// Get an array element
func (array Dense3DArray) Item(index ...int) float64 {
	return array[index[0]][index[1]][index[2]]
}

// Set an array element
func (array Dense3DArray) ItemSet(value float64, index ...int) {
	array[index[0]][index[1]][index[2]] = value
}

// Set all array elements to the given value
func (array Dense3DArray) Fill(value float64) {
	for i0 := range array {
		for i1 := range array[i0] {
			for i2 := range array[i0][i1] {
				array[i0][i1][i2] = value
			}
		}
	}
}

// Return the sum of all array elements
func (array Dense3DArray) Sum() float64 {
	var result float64
	for i0 := range array {
		for i1 := range array[i0] {
			for _, v := range array[i0][i1] {
				result += v
			}
		}
	}
	return result
}

// Normalize the array to sum to 1, or do nothing if all items are 0.
func (array *Dense3DArray) Normalize() {
	s := array.Sum()
	if s != 0 && s != 1 {
		for i0 := range *array {
			for i1 := range (*array)[i0] {
				for i2 := range (*array)[i0][i1] {
					(*array)[i0][i1][i2] /= s
				}
			}
		}
	}
}

// Create an NDArray of float64 values, initialized to zero
func Dense(size ...int) (NDArray, error) {
	switch len(size) {
	case 1:
		array := make(Dense1DArray, size[0])
		return &array, nil
	case 2:
		array := make(Dense2DArray, size[0])
		for i0 := range array {
			array[i0] = make([]float64, size[1])
		}
		return &array, nil
	case 3:
		array := make(Dense3DArray, size[0])
		for i0 := range array {
			array[i0] = make([][]float64, size[1])
			for i1 := range array[i0] {
				array[i0][i1] = make([]float64, size[2])
			}
		}
		return &array, nil
	default:
		return nil, errors.New("Unsupported number of dimensions")
	}
}
