package matrix

import (
	"fmt"
)

// A one-dimensional NDArray with dense representation
type DenseF64Array struct {
	shape     []int
	array     []float64
	transpose bool
}

// Return the element-wise sum of this array and one or more others
func (array DenseF64Array) Add(other ...NDArray) NDArray {
	return Add(&array, other...)
}

// Returns true if and only if all items are nonzero
func (array DenseF64Array) All() bool {
	return All(&array)
}

// Returns true if and only if any item is nonzero
func (array DenseF64Array) Any() bool {
	return Any(&array)
}

// Returns true if f is true for any array element
func (array DenseF64Array) AnyF(f func(v float64) bool) bool {
	return AnyF(&array, f)
}

// Returns true if f is true for any pair of array elements in the same position
func (array DenseF64Array) AnyF2(f func(v1, v2 float64) bool, other NDArray) bool {
	return AnyF2(&array, f, other)
}

// Return the result of applying a function to all elements
func (array DenseF64Array) Apply(f func(float64) float64) NDArray {
	result := array.copy()
	for i, val := range result.array {
		result.array[i] = f(val)
	}
	return result
}

// Get the matrix data as a 1D array
func (array DenseF64Array) Array() []float64 {
	return array.array
}

// Set the values of the items on a given column
func (array DenseF64Array) ColSet(col int, values []float64) {
	if len(array.shape) != 2 {
		panic(fmt.Sprintf("Can't ColSet a %d-dim array", len(array.shape)))
	} else if col < 0 || col >= array.shape[1] {
		panic(fmt.Sprintf("RowSet can't set col %d of a %d-col array", col, array.shape[1]))
	} else if len(values) != array.shape[0] {
		panic(fmt.Sprintf("ColSet has %d rows but got %d values", array.shape[0], len(values)))
	}
	for row := 0; row < array.shape[0]; row++ {
		array.ItemSet(values[row], row, col)
	}
}

// Get the number of columns
func (array DenseF64Array) Cols() int {
	if len(array.shape) != 2 {
		panic(fmt.Sprintf("Can't count columns for a %d-dim array", len(array.shape)))
	}
	return array.shape[1]
}

// Create a new array by concatenating this with another array along the
// specified axis. The array shapes must be equal along all other axes.
// It is legal to add a new axis.
func (array DenseF64Array) Concat(axis int, others ...NDArray) NDArray {
	return Concat(axis, &array, others...)
}

// Returns a duplicate of this array
func (array DenseF64Array) Copy() NDArray {
	return array.copy()
}

// Returns a duplicate of this array, preserving type
func (array DenseF64Array) copy() *DenseF64Array {
	result := &DenseF64Array{
		shape: make([]int, len(array.shape)),
		array: make([]float64, len(array.array)),
	}
	if array.transpose {
		result.shape[0], result.shape[1] = array.shape[1], array.shape[0]
		for i0 := 0; i0 < array.shape[0]; i0++ {
			for i1 := 0; i1 < array.shape[1]; i1++ {
				result.ItemSet(array.Item(i0, i1), i0, i1)
			}
		}
	} else {
		copy(result.shape[:], array.shape[:])
		copy(result.array[:], array.array[:])
	}
	return result
}

// Get a column vector containing the main diagonal elements of the matrix
func (array DenseF64Array) Diag() Matrix {
	if len(array.shape) != 2 {
		panic(fmt.Sprintf("Can't take diag of a %d-dim array", len(array.shape)))
	}
	size := array.shape[0]
	if array.shape[1] < size {
		size = array.shape[1]
	}
	result := Dense(size, 1).M()
	for i := 0; i < size; i++ {
		result.ItemSet(array.Item(i, i), i, 0)
	}
	return result
}

// Return the element-wise quotient of this array and one or more others
func (array DenseF64Array) Div(other ...NDArray) NDArray {
	return Div(&array, other...)
}

// Returns true if and only if all elements in the two arrays are equal
func (array DenseF64Array) Equal(other NDArray) bool {
	return Equal(&array, other)
}

// Set all array elements to the given value
func (array DenseF64Array) Fill(value float64) {
	Fill(&array, value)
}

// Get an array element in a flattened verison of this array
func (array DenseF64Array) FlatItem(index int) float64 {
	if array.transpose {
		nd := flatToNd(array.shape, index)
		index = ndToFlat([]int{array.shape[1], array.shape[0]}, []int{nd[1], nd[0]})
	}
	return array.array[index]
}

// Set an array element in a flattened version of this array
func (array DenseF64Array) FlatItemSet(value float64, index int) {
	if array.transpose {
		nd := flatToNd(array.shape, index)
		index = ndToFlat([]int{array.shape[1], array.shape[0]}, []int{nd[1], nd[0]})
	}
	array.array[index] = value
}

// Get the matrix inverse
func (array DenseF64Array) Inverse() Matrix {
	if len(array.shape) != 2 {
		panic(fmt.Sprintf("Can't take inverse of a %d-dim array", len(array.shape)))
	}
	return Inverse(&array)
}

// Get an array element
func (array DenseF64Array) Item(index ...int) float64 {
	shape := array.shape
	if array.transpose {
		index[0], index[1] = index[1], index[0]
		shape = []int{array.shape[1], array.shape[0]}
	}
	return array.array[ndToFlat(shape, index)]
}

// Add a scalar value to each array element
func (array *DenseF64Array) ItemAdd(value float64) NDArray {
	result := array.copy()
	for idx := range result.array {
		result.array[idx] += value
	}
	return result
}

// Divide each array element by a scalar value
func (array *DenseF64Array) ItemDiv(value float64) NDArray {
	result := array.copy()
	for idx := range result.array {
		result.array[idx] /= value
	}
	return result
}

// Multiply each array element by a scalar value
func (array *DenseF64Array) ItemProd(value float64) NDArray {
	result := array.copy()
	for idx := range result.array {
		result.array[idx] *= value
	}
	return result
}

// Subtract a scalar value from each array element
func (array *DenseF64Array) ItemSub(value float64) NDArray {
	result := array.copy()
	for idx := range result.array {
		result.array[idx] -= value
	}
	return result
}

// Set an array element
func (array DenseF64Array) ItemSet(value float64, index ...int) {
	shape := array.shape
	if array.transpose {
		index[0], index[1] = index[1], index[0]
		shape = []int{array.shape[1], array.shape[0]}
	}
	array.array[ndToFlat(shape, index)] = value
}

// Solve for x, where ax = b.
func (array DenseF64Array) LDivide(b Matrix) Matrix {
	if len(array.shape) != 2 {
		panic(fmt.Sprintf("Can't LDivide a %d-dim array", len(array.shape)))
	}
	return LDivide(&array, b)
}

// Get the result of matrix multiplication between this and some other
// array(s). All arrays must have two dimensions, and the dimensions must
// be aligned correctly for multiplication.
// If A is m x p and B is p x n, then C = A.MProd(B) is the m x n matrix
// with C[i, j] = \sum_{k=1}^p A[i,k] * B[k,j].
func (array DenseF64Array) MProd(others ...Matrix) Matrix {
	return MProd(&array, others...)
}

// Get the value of the largest array element
func (array DenseF64Array) Max() float64 {
	return Max(&array)
}

// Get the value of the smallest array element
func (array DenseF64Array) Min() float64 {
	return Min(&array)
}

// The number of dimensions in the matrix
func (array DenseF64Array) NDim() int {
	return len(array.shape)
}

// Get the matrix norm of the specified ordinality (1, 2, infinity, ...)
func (array DenseF64Array) Norm(ord float64) float64 {
	return Norm(&array, ord)
}

// Return a copy of the array, normalized to sum to 1
func (array *DenseF64Array) Normalize() NDArray {
	return Normalize(array)
}

// Return the element-wise product of this array and one or more others
func (array DenseF64Array) Prod(other ...NDArray) NDArray {
	return Prod(&array, other...)
}

// Get a 1D copy of the array, in 'C' order: rightmost axes change fastest
func (array DenseF64Array) Ravel() NDArray {
	return Ravel(&array)
}

// Set the values of the items on a given row
func (array DenseF64Array) RowSet(row int, values []float64) {
	if len(array.shape) != 2 {
		panic(fmt.Sprintf("Can't RowSet a %d-dim array", len(array.shape)))
	} else if row < 0 || row >= array.shape[0] {
		panic(fmt.Sprintf("RowSet can't set row %d of a %d-row array", row, array.shape[0]))
	} else if len(values) != array.shape[1] {
		panic(fmt.Sprintf("RowSet has %d columns but got %d values", array.shape[1], len(values)))
	}
	for col := 0; col < array.shape[1]; col++ {
		array.ItemSet(values[col], row, col)
	}
}

// Get the number of rows
func (array DenseF64Array) Rows() int {
	if len(array.shape) != 2 {
		panic(fmt.Sprintf("Can't count rows for a %d-dim array", len(array.shape)))
	}
	return array.shape[0]
}

// A slice giving the size of all array dimensions
func (array DenseF64Array) Shape() []int {
	return array.shape
}

// The total number of elements in the matrix
func (array DenseF64Array) Size() int {
	return len(array.array)
}

// Get an array containing a rectangular slice of this array.
// `from` and `to` should both have one index per axis. The indices
// in `from` and `to` define the first and just-past-last indices you wish
// to select along each axis.
func (array DenseF64Array) Slice(from []int, to []int) NDArray {
	return Slice(&array, from, to)
}

// Return the element-wise difference of this array and one or more others
func (array DenseF64Array) Sub(other ...NDArray) NDArray {
	return Sub(&array, other...)
}

// Return the sum of all array elements
func (array DenseF64Array) Sum() float64 {
	return Sum(&array)
}

// Returns the array as a matrix. This is only possible for 1D and 2D arrays;
// 1D arrays of length n are converted into n x 1 vectors.
func (array DenseF64Array) M() Matrix {
	switch array.NDim() {
	default:
		panic(fmt.Sprintf("Cannot convert a %d-dim array into a matrix", array.NDim()))

	case 1:
		return &DenseF64Array{
			shape:     []int{array.shape[0], 1},
			array:     array.array,
			transpose: array.transpose,
		}

	case 2:
		return &array
	}
}

// Return the same matrix, but with axes transposed. The same data is used,
// for speed and memory efficiency. Use Copy() to create a new array.
// A 1D array is unchanged; create a 2D analog to rotate a vector.
func (array DenseF64Array) T() NDArray {
	switch len(array.shape) {
	default:
		panic(fmt.Sprintf("Can't take transpose of %d-dim array", len(array.shape)))

	case 1:
		return &array

	case 2:
		return &DenseF64Array{
			shape:     []int{array.shape[1], array.shape[0]},
			array:     array.array,
			transpose: true,
		}
	}
}
