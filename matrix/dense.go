package matrix

import (
	"fmt"
)

// An n-dimensional NDArray with dense representation
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

// Returns true if f is true for all array elements
func (array DenseF64Array) AllF(f func(v float64) bool) bool {
	return AllF(&array, f)
}

// Returns true if f is true for all pairs of array elements in the same position
func (array DenseF64Array) AllF2(f func(v1, v2 float64) bool, other NDArray) bool {
	return AllF2(&array, f, other)
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

// Get the matrix data as a flattened 1D array; sparse matrices will make
// a copy first.
func (array DenseF64Array) Array() []float64 {
	if array.transpose {
		return array.copy().array
	} else {
		return array.array
	}
}

// Set the values of the items on a given column
func (array DenseF64Array) ColSet(col int, values []float64) {
	if col < 0 || col >= array.shape[1] {
		panic(fmt.Sprintf("ColSet can't set col %d of a %d-col array", col, array.shape[1]))
	} else if len(values) != array.shape[0] {
		panic(fmt.Sprintf("ColSet has %d rows but got %d values", array.shape[0], len(values)))
	}
	for row := 0; row < array.shape[0]; row++ {
		array.ItemSet(values[row], row, col)
	}
}

// Get a particular column for read-only access. May or may not be a copy.
func (array DenseF64Array) Col(col int) []float64 {
	if col < 0 || col >= array.shape[1] {
		panic(fmt.Sprintf("Can't get column %d from a %dx%d array", col, array.shape[0], array.shape[1]))
	}
	result := make([]float64, array.shape[0])
	for row := 0; row < array.shape[0]; row++ {
		result[row] = array.Item(row, col)
	}
	return result
}

// Get the number of columns
func (array DenseF64Array) Cols() int {
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
	copy(result.shape[:], array.shape[:])
	if array.transpose {
		for i0 := 0; i0 < array.shape[0]; i0++ {
			for i1 := 0; i1 < array.shape[1]; i1++ {
				result.ItemSet(array.Item(i0, i1), i0, i1)
			}
		}
	} else {
		copy(result.array[:], array.array[:])
	}
	return result
}

// Counts the number of nonzero elements in the array
func (array DenseF64Array) CountNonzero() int {
	count := 0
	for _, v := range array.array {
		if v != 0 {
			count++
		}
	}
	return count
}

// Returns a dense copy of the array
func (array DenseF64Array) Dense() NDArray {
	return array.copy()
}

// Get a column vector containing the main diagonal elements of the matrix
func (array DenseF64Array) Diag() Matrix {
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

// Return the element-wise quotient of this array and one or more others.
// This function defines 0 / 0 = 0, so it's useful for sparse arrays.
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
func (array DenseF64Array) Inverse() (Matrix, error) {
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
	if row < 0 || row >= array.shape[0] {
		panic(fmt.Sprintf("RowSet can't set row %d of a %d-row array", row, array.shape[0]))
	} else if len(values) != array.shape[1] {
		panic(fmt.Sprintf("RowSet has %d columns but got %d values", array.shape[1], len(values)))
	}
	for col := 0; col < array.shape[1]; col++ {
		array.ItemSet(values[col], row, col)
	}
}

// Get a particular row for read-only access. May or may not be a copy.
func (array DenseF64Array) Row(row int) []float64 {
	if row < 0 || row >= array.shape[0] {
		panic(fmt.Sprintf("Can't get row %d from a %dx%d array", row, array.shape[0], array.shape[1]))
	}
	start := ndToFlat(array.shape, []int{row, 0})
	return array.array[start : start+array.shape[1]]
}

// Get the number of rows
func (array DenseF64Array) Rows() int {
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

// Ask whether the matrix has a sparse representation (useful for optimization)
func (array DenseF64Array) Sparsity() ArraySparsity {
	return DenseArray
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
func (array DenseF64Array) T() Matrix {
	return &DenseF64Array{
		shape:     []int{array.shape[1], array.shape[0]},
		array:     array.array,
		transpose: !array.transpose,
	}
}

// Visit all matrix elements, invoking a method on each. If the method
// returns false, iteration is aborted and VisitNonzero() returns false.
// Otherwise, it returns true.
func (array DenseF64Array) Visit(f func(pos []int, value float64) bool) bool {
	for flat, value := range array.array {
		var pos []int
		if array.transpose {
			pOrig := flatToNd([]int{array.shape[1], array.shape[0]}, flat)
			pos = []int{pOrig[1], pOrig[0]}
		} else {
			pos = flatToNd(array.shape, flat)
		}
		if !f(pos, value) {
			return false
		}
	}
	return true
}

// Visit just nonzero elements, invoking a method on each. If the method
// returns false, iteration is aborted and VisitNonzero() returns false.
// Otherwise, it returns true.
func (array DenseF64Array) VisitNonzero(f func(pos []int, value float64) bool) bool {
	for flat, value := range array.array {
		if value != 0 {
			var pos []int
			if array.transpose {
				pOrig := flatToNd([]int{array.shape[1], array.shape[0]}, flat)
				pos = []int{pOrig[1], pOrig[0]}
			} else {
				pos = flatToNd(array.shape, flat)
			}
			if !f(pos, value) {
				return false
			}
		}
	}
	return true
}
