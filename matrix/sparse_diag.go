package matrix

import (
	"fmt"
)

// A sparse 2D Matrix with diagonal representation: only the main diagonal is
// stored; all other values are zero.
type SparseDiagF64Matrix struct {
	shape []int
	diag  []float64
}

// Return the element-wise sum of this array and one or more others
func (array SparseDiagF64Matrix) Add(other ...NDArray) NDArray {
	return Add(&array, other...)
}

// Returns true if and only if all items are nonzero
func (array SparseDiagF64Matrix) All() bool {
	return All(&array)
}

// Returns true if f is true for all array elements
func (array SparseDiagF64Matrix) AllF(f func(v float64) bool) bool {
	return AllF(&array, f)
}

// Returns true if f is true for all pairs of array elements in the same position
func (array SparseDiagF64Matrix) AllF2(f func(v1, v2 float64) bool, other NDArray) bool {
	return AllF2(&array, f, other)
}

// Returns true if and only if any item is nonzero
func (array SparseDiagF64Matrix) Any() bool {
	return Any(&array)
}

// Returns true if f is true for any array element
func (array SparseDiagF64Matrix) AnyF(f func(v float64) bool) bool {
	return AnyF(&array, f)
}

// Returns true if f is true for any pair of array elements in the same position
func (array SparseDiagF64Matrix) AnyF2(f func(v1, v2 float64) bool, other NDArray) bool {
	return AnyF2(&array, f, other)
}

// Return the result of applying a function to all elements
func (array SparseDiagF64Matrix) Apply(f func(float64) float64) NDArray {
	return Apply(&array, f)
}

// Get the matrix data as a flattened 1D array; sparse matrices will make
// a copy first.
func (array SparseDiagF64Matrix) Array() []float64 {
	return array.Dense().Array()
}

// Set the values of the items on a given column
func (array SparseDiagF64Matrix) ColSet(col int, values []float64) {
	if col < 0 || col >= array.shape[1] {
		panic(fmt.Sprintf("ColSet can't set col %d of a %d-col array", col, array.shape[1]))
	} else if len(values) != array.shape[0] {
		panic(fmt.Sprintf("ColSet has %d rows but got %d values", array.shape[0], len(values)))
	}
	for row := 0; row < array.shape[0]; row++ {
		if row != col {
			if values[row] != 0 {
				panic(fmt.Sprintf("ColSet can't set cell (%d, %d) of a %dx%d sparse diagonal matrix", row, col, array.shape[0], array.shape[1]))
			}
		} else {
			array.diag[row] = values[row]
		}
	}
}

// Get a particular column for read-only access. May or may not be a copy.
func (array SparseDiagF64Matrix) Col(col int) []float64 {
	if col < 0 || col >= array.shape[1] {
		panic(fmt.Sprintf("Can't get column %d from a %dx%d array", col, array.shape[0], array.shape[1]))
	}
	result := make([]float64, array.shape[1])
	result[col] = array.diag[col]
	return result
}

// Get the number of columns
func (array SparseDiagF64Matrix) Cols() int {
	return array.shape[1]
}

// Create a new array by concatenating this with another array along the
// specified axis. The array shapes must be equal along all other axes.
// It is legal to add a new axis.
func (array SparseDiagF64Matrix) Concat(axis int, others ...NDArray) NDArray {
	return Concat(axis, &array, others...)
}

// Returns a duplicate of this array
func (array SparseDiagF64Matrix) Copy() NDArray {
	return array.copy()
}

// Returns a duplicate of this array, preserving type
func (array SparseDiagF64Matrix) copy() *SparseDiagF64Matrix {
	result := &SparseDiagF64Matrix{
		shape: make([]int, len(array.shape)),
		diag:  make([]float64, len(array.diag)),
	}
	copy(result.shape[:], array.shape[:])
	copy(result.diag[:], array.diag[:])
	return result
}

// Counts the number of nonzero elements in the array
func (array SparseDiagF64Matrix) CountNonzero() int {
	count := 0
	for _, v := range array.diag {
		if v != 0 {
			count++
		}
	}
	return count
}

// Returns a dense copy of the array
func (array SparseDiagF64Matrix) Dense() NDArray {
	result := Dense(array.shape...)
	for pos, val := range array.diag {
		result.ItemSet(val, pos, pos)
	}
	return result
}

// Get a column vector containing the main diagonal elements of the matrix
func (array SparseDiagF64Matrix) Diag() Matrix {
	return A([]int{len(array.diag), 1}, array.diag...).M()
}

// Return the element-wise quotient of this array and one or more others.
// This function defines 0 / 0 = 0, so it's useful for sparse arrays.
func (array SparseDiagF64Matrix) Div(other ...NDArray) NDArray {
	return Div(&array, other...)
}

// Returns true if and only if all elements in the two arrays are equal
func (array SparseDiagF64Matrix) Equal(other NDArray) bool {
	return Equal(&array, other)
}

// Set all array elements to the given value
func (array SparseDiagF64Matrix) Fill(value float64) {
	panic("Can't Fill() a sparse diagonal matrix")
}

// Get an array element in a flattened verison of this array
func (array SparseDiagF64Matrix) FlatItem(index int) float64 {
	coord := flatToNd(array.shape, index)
	if coord[0] != coord[1] || coord[0] >= len(array.diag) {
		return 0
	}
	return array.diag[coord[0]]
}

// Set an array element in a flattened version of this array
func (array SparseDiagF64Matrix) FlatItemSet(value float64, index int) {
	coord := flatToNd(array.shape, index)
	if coord[0] != coord[1] || coord[0] >= len(array.diag) {
		panic(fmt.Sprintf("FlatItemSet index %v invalid for sparse diagonal array shape %v", index, array.shape))
	}
	array.diag[coord[0]] = value
}

// Return an iterator over populated matrix entries
func (array *SparseDiagF64Matrix) FlatIter() FlatNDArrayIterator {
	return &sparseDiagIterator{
		array:   array,
		diagPos: 0,
	}
}

// Get the matrix inverse
func (array SparseDiagF64Matrix) Inverse() Matrix {
	return Inverse(&array)
}

// Get an array element
func (array SparseDiagF64Matrix) Item(index ...int) float64 {
	if len(index) != 2 || index[0] >= array.shape[0] || index[1] >= array.shape[1] {
		panic(fmt.Sprintf("Item indices %v invalid for array shape %v", index, array.shape))
	} else if index[0] != index[1] || index[0] >= len(array.diag) {
		return 0
	}
	return array.diag[index[0]]
}

// Add a scalar value to each array element
func (array *SparseDiagF64Matrix) ItemAdd(value float64) NDArray {
	result := &DenseF64Array{
		shape: make([]int, 2),
		array: make([]float64, array.shape[0]*array.shape[1]),
	}
	copy(result.shape[:], array.shape[:])
	flat := 0
	for row := 0; row < array.shape[0]; row++ {
		for col := 0; col < array.shape[1]; col++ {
			if row == col {
				result.array[flat] = array.diag[row] + value
			} else {
				result.array[flat] = value
			}
			flat++
		}
	}
	return result
}

// Divide each array element by a scalar value
func (array *SparseDiagF64Matrix) ItemDiv(value float64) NDArray {
	result := array.copy()
	for i := 0; i < len(result.diag); i++ {
		result.diag[i] /= value
	}
	return result
}

// Multiply each array element by a scalar value
func (array *SparseDiagF64Matrix) ItemProd(value float64) NDArray {
	result := array.copy()
	for i := 0; i < len(result.diag); i++ {
		result.diag[i] *= value
	}
	return result
}

// Subtract a scalar value from each array element
func (array *SparseDiagF64Matrix) ItemSub(value float64) NDArray {
	result := &DenseF64Array{
		shape: make([]int, 2),
		array: make([]float64, array.shape[0]*array.shape[1]),
	}
	copy(result.shape[:], array.shape[:])
	flat := 0
	for row := 0; row < array.shape[0]; row++ {
		for col := 0; col < array.shape[1]; col++ {
			if row == col {
				result.array[flat] = array.diag[row] - value
			} else {
				result.array[flat] = -value
			}
			flat++
		}
	}
	return result
}

// Set an array element
func (array SparseDiagF64Matrix) ItemSet(value float64, index ...int) {
	if len(index) != 2 || index[0] >= array.shape[0] || index[1] >= array.shape[1] {
		panic(fmt.Sprintf("ItemSet indices %v invalid for array shape %v", index, array.shape))
	} else if index[0] != index[1] {
		panic(fmt.Sprintf("ItemSet indices %v invalid for sparse diagonal array", index))
	}
	array.diag[index[0]] = value
}

// Return an iterator over populated matrix entries
func (array *SparseDiagF64Matrix) Iter() CoordNDArrayIterator {
	return &sparseDiagIterator{
		array:   array,
		diagPos: 0,
	}
}

// Solve for x, where ax = b.
func (array SparseDiagF64Matrix) LDivide(b Matrix) Matrix {
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
func (array SparseDiagF64Matrix) MProd(others ...Matrix) Matrix {
	return MProd(&array, others...)
}

// Get the value of the largest array element
func (array SparseDiagF64Matrix) Max() float64 {
	return Max(&array)
}

// Get the value of the smallest array element
func (array SparseDiagF64Matrix) Min() float64 {
	return Min(&array)
}

// The number of dimensions in the matrix
func (array SparseDiagF64Matrix) NDim() int {
	return len(array.shape)
}

// Get the matrix norm of the specified ordinality (1, 2, infinity, ...)
func (array SparseDiagF64Matrix) Norm(ord float64) float64 {
	return Norm(&array, ord)
}

// Return a copy of the array, normalized to sum to 1
func (array *SparseDiagF64Matrix) Normalize() NDArray {
	return Normalize(array)
}

// Return the element-wise product of this array and one or more others
func (array SparseDiagF64Matrix) Prod(other ...NDArray) NDArray {
	return Prod(&array, other...)
}

// Get a 1D copy of the array, in 'C' order: rightmost axes change fastest
func (array SparseDiagF64Matrix) Ravel() NDArray {
	return Ravel(&array)
}

// Set the values of the items on a given row
func (array SparseDiagF64Matrix) RowSet(row int, values []float64) {
	if row < 0 || row >= array.shape[0] {
		panic(fmt.Sprintf("RowSet can't set row %d of a %d-row array", row, array.shape[0]))
	} else if len(values) != array.shape[1] {
		panic(fmt.Sprintf("RowSet has %d columns but got %d values", array.shape[1], len(values)))
	}
	for col := 0; col < array.shape[1]; col++ {
		if row != col {
			if values[col] != 0 {
				panic(fmt.Sprintf("RowSet can't set cell (%d, %d) of a %dx%d sparse diagonal matrix", row, col, array.shape[0], array.shape[1]))
			}
		} else {
			array.diag[col] = values[col]
		}
	}
}

// Get a particular row for read-only access. May or may not be a copy.
func (array SparseDiagF64Matrix) Row(row int) []float64 {
	if row < 0 || row >= array.shape[0] {
		panic(fmt.Sprintf("Can't get row %d from a %dx%d array", row, array.shape[0], array.shape[1]))
	}
	result := make([]float64, array.shape[0])
	result[row] = array.diag[row]
	return result
}

// Get the number of rows
func (array SparseDiagF64Matrix) Rows() int {
	return array.shape[0]
}

// A slice giving the size of all array dimensions
func (array SparseDiagF64Matrix) Shape() []int {
	return array.shape
}

// The total number of elements in the matrix
func (array SparseDiagF64Matrix) Size() int {
	return array.shape[0] * array.shape[1]
}

// Get an array containing a rectangular slice of this array.
// `from` and `to` should both have one index per axis. The indices
// in `from` and `to` define the first and just-past-last indices you wish
// to select along each axis.
func (array SparseDiagF64Matrix) Slice(from []int, to []int) NDArray {
	return Slice(&array, from, to)
}

// Ask whether the matrix has a sparse representation (useful for optimization)
func (array SparseDiagF64Matrix) Sparsity() ArraySparsity {
	return SparseDiagMatrix
}

// Return the element-wise difference of this array and one or more others
func (array SparseDiagF64Matrix) Sub(other ...NDArray) NDArray {
	return Sub(&array, other...)
}

// Return the sum of all array elements
func (array SparseDiagF64Matrix) Sum() float64 {
	return Sum(&array)
}

// Returns the array as a matrix.
func (array SparseDiagF64Matrix) M() Matrix {
	return &array
}

// Return the same matrix, but with axes transposed. The same data is used,
// for speed and memory efficiency. Use Copy() to create a new array.
func (array SparseDiagF64Matrix) T() NDArray {
	return &array
}

// Iterates over all array elements
type sparseDiagIterator struct {
	array   *SparseDiagF64Matrix
	diagPos int
}

// Ask whether there are more values to iterate over.
func (iter sparseDiagIterator) HasNext() bool {
	return iter.diagPos < len(iter.array.diag)
}

// Return the value and coordinates of the next entry
func (iter *sparseDiagIterator) Next() (float64, []int) {
	if iter.diagPos >= len(iter.array.diag) {
		return 0, nil
	}
	pos := iter.diagPos
	iter.diagPos++
	return iter.array.diag[pos], []int{pos, pos}
}

// Return the value and flat index of the next entry
func (iter *sparseDiagIterator) FlatNext() (float64, int) {
	if iter.diagPos >= len(iter.array.diag) {
		return 0, 0
	}
	pos := iter.diagPos
	iter.diagPos++
	return iter.array.diag[pos], ndToFlat(iter.array.shape, []int{pos, pos})
}
