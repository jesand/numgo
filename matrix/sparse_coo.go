package matrix

import (
	"fmt"
)

// A sparse 2D Matrix with coordinate representation
type sparseCooF64Matrix struct {
	shape     []int
	values    []map[int]float64
	transpose bool
}

// Return the element-wise sum of this array and one or more others
func (array sparseCooF64Matrix) Add(other ...NDArray) NDArray {
	return Add(&array, other...)
}

// Returns true if and only if all items are nonzero
func (array sparseCooF64Matrix) All() bool {
	return All(&array)
}

// Returns true if f is true for all array elements
func (array sparseCooF64Matrix) AllF(f func(v float64) bool) bool {
	return AllF(&array, f)
}

// Returns true if f is true for all pairs of array elements in the same position
func (array sparseCooF64Matrix) AllF2(f func(v1, v2 float64) bool, other NDArray) bool {
	return AllF2(&array, f, other)
}

// Returns true if and only if any item is nonzero
func (array sparseCooF64Matrix) Any() bool {
	return Any(&array)
}

// Returns true if f is true for any array element
func (array sparseCooF64Matrix) AnyF(f func(v float64) bool) bool {
	return AnyF(&array, f)
}

// Returns true if f is true for any pair of array elements in the same position
func (array sparseCooF64Matrix) AnyF2(f func(v1, v2 float64) bool, other NDArray) bool {
	return AnyF2(&array, f, other)
}

// Return the result of applying a function to all elements
func (array sparseCooF64Matrix) Apply(f func(float64) float64) NDArray {
	return Apply(&array, f)
}

// Get the matrix data as a flattened 1D array; sparse matrices will make
// a copy first.
func (array sparseCooF64Matrix) Array() []float64 {
	return array.Dense().Array()
}

// Set the values of the items on a given column
func (array *sparseCooF64Matrix) ColSet(col int, values []float64) {
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
func (array sparseCooF64Matrix) Col(col int) []float64 {
	if col < 0 || col >= array.shape[1] {
		panic(fmt.Sprintf("Can't get column %d from a %dx%d array", col, array.shape[0], array.shape[1]))
	}
	result := make([]float64, array.shape[1])
	for row, val := range array.values {
		result[row] = val[col]
	}
	return result
}

// Get the number of columns
func (array sparseCooF64Matrix) Cols() int {
	return array.shape[1]
}

// Create a new array by concatenating this with another array along the
// specified axis. The array shapes must be equal along all other axes.
// It is legal to add a new axis.
func (array sparseCooF64Matrix) Concat(axis int, others ...NDArray) NDArray {
	return Concat(axis, &array, others...)
}

// Returns a duplicate of this array
func (array sparseCooF64Matrix) Copy() NDArray {
	return array.copy()
}

// Returns a duplicate of this array, preserving type
func (array sparseCooF64Matrix) copy() *sparseCooF64Matrix {
	result := SparseCoo(array.shape[0], array.shape[1]).(*sparseCooF64Matrix)
	if array.transpose {
		for row, val := range array.values {
			for col, v := range val {
				result.values[col][row] = v
			}
		}
	} else {
		for row, val := range array.values {
			for col, v := range val {
				result.values[row][col] = v
			}
		}
	}
	return result
}

// Counts the number of nonzero elements in the array
func (array sparseCooF64Matrix) CountNonzero() int {
	count := 0
	for _, val := range array.values {
		count += len(val)
	}
	return count
}

// Returns a dense copy of the array
func (array sparseCooF64Matrix) Dense() NDArray {
	var result NDArray
	result = Dense(array.shape...)
	for row, val := range array.values {
		for col, v := range val {
			if array.transpose {
				result.ItemSet(v, col, row)
			} else {
				result.ItemSet(v, row, col)
			}
		}
	}
	return result
}

// Get a column vector containing the main diagonal elements of the matrix
func (array sparseCooF64Matrix) Diag() Matrix {
	size := array.shape[0]
	if array.shape[1] < size {
		size = array.shape[1]
	}
	result := Dense(size, 1).M()
	for row, val := range array.values {
		result.ItemSet(val[row], row, 0)
	}
	return result
}

// Treat the rows as points, and get the pairwise distance between them.
// Returns a distance matrix D such that D_i,j is the distance between
// rows i and j.
func (array sparseCooF64Matrix) Dist(t DistType) Matrix {
	return Dist(&array, t)
}

// Return the element-wise quotient of this array and one or more others.
// This function defines 0 / 0 = 0, so it's useful for sparse arrays.
func (array sparseCooF64Matrix) Div(other ...NDArray) NDArray {
	return Div(&array, other...)
}

// Returns true if and only if all elements in the two arrays are equal
func (array sparseCooF64Matrix) Equal(other NDArray) bool {
	return Equal(&array, other)
}

// Set all array elements to the given value
func (array sparseCooF64Matrix) Fill(value float64) {
	panic("Can't Fill() a sparse coo matrix")
}

// Get the coordinates for the item at the specified flat position
func (array sparseCooF64Matrix) FlatCoord(index int) []int {
	return flatToNd(array.shape, index)
}

// Get an array element in a flattened verison of this array
func (array sparseCooF64Matrix) FlatItem(index int) float64 {
	// This is ok with transpose, because array.Item does transposition for us
	nd := flatToNd(array.shape, index)
	return array.Item(nd[0], nd[1])
}

// Set an array element in a flattened version of this array
func (array *sparseCooF64Matrix) FlatItemSet(value float64, index int) {
	nd := flatToNd(array.shape, index)
	if array.transpose {
		array.ItemSet(value, nd[0], nd[1])
	}
	array.ItemSet(value, nd[0], nd[1])
}

// Get the matrix inverse
func (array sparseCooF64Matrix) Inverse() (Matrix, error) {
	return Inverse(&array)
}

// Get an array element
func (array sparseCooF64Matrix) Item(index ...int) float64 {
	if len(index) != 2 || index[0] >= array.shape[0] || index[1] >= array.shape[1] {
		panic(fmt.Sprintf("Item indices %v invalid for array shape %v", index, array.shape))
	}
	if array.transpose {
		index[0], index[1] = index[1], index[0]
	}
	return array.values[index[0]][index[1]]
}

// Add a scalar value to each array element
func (array *sparseCooF64Matrix) ItemAdd(value float64) NDArray {
	result := WithValue(value, array.shape...)
	for row, val := range array.values {
		for col, v := range val {
			if array.transpose {
				result.ItemSet(v+value, col, row)
			} else {
				result.ItemSet(v+value, row, col)
			}
		}
	}
	return result
}

// Divide each array element by a scalar value
func (array *sparseCooF64Matrix) ItemDiv(value float64) NDArray {
	result := Dense(array.shape...)
	for row, val := range array.values {
		for col, v := range val {
			if array.transpose {
				result.ItemSet(v/value, col, row)
			} else {
				result.ItemSet(v/value, row, col)
			}
		}
	}
	return result
}

// Multiply each array element by a scalar value
func (array *sparseCooF64Matrix) ItemProd(value float64) NDArray {
	result := Dense(array.shape...)
	for row, val := range array.values {
		for col, v := range val {
			if array.transpose {
				result.ItemSet(v*value, col, row)
			} else {
				result.ItemSet(v*value, row, col)
			}
		}
	}
	return result
}

// Subtract a scalar value from each array element
func (array *sparseCooF64Matrix) ItemSub(value float64) NDArray {
	result := WithValue(-value, array.shape...)
	for row, val := range array.values {
		for col, v := range val {
			if array.transpose {
				result.ItemSet(v-value, col, row)
			} else {
				result.ItemSet(v-value, row, col)
			}
		}
	}
	return result
}

// Set an array element
func (array *sparseCooF64Matrix) ItemSet(value float64, index ...int) {
	if len(index) != 2 || index[0] >= array.shape[0] || index[1] >= array.shape[1] {
		panic(fmt.Sprintf("Item indices %v invalid for array shape %v", index, array.shape))
	}
	if array.transpose {
		index[0], index[1] = index[1], index[0]
	}
	if value == 0 {
		delete(array.values[index[0]], index[1])
	} else {
		array.values[index[0]][index[1]] = value
	}
}

// Solve for x, where ax = b.
func (array sparseCooF64Matrix) LDivide(b Matrix) Matrix {
	return LDivide(&array, b)
}

// Get the result of matrix multiplication between this and some other
// array(s). All arrays must have two dimensions, and the dimensions must
// be aligned correctly for multiplication.
// If A is m x p and B is p x n, then C = A.MProd(B) is the m x n matrix
// with C[i, j] = \sum_{k=1}^p A[i,k] * B[k,j].
func (array sparseCooF64Matrix) MProd(others ...Matrix) Matrix {
	return MProd(&array, others...)
}

// Get the value of the largest array element
func (array sparseCooF64Matrix) Max() float64 {
	return Max(&array)
}

// Get the value of the smallest array element
func (array sparseCooF64Matrix) Min() float64 {
	return Min(&array)
}

// The number of dimensions in the matrix
func (array sparseCooF64Matrix) NDim() int {
	return len(array.shape)
}

// Get the matrix norm of the specified ordinality (1, 2, infinity, ...)
func (array sparseCooF64Matrix) Norm(ord float64) float64 {
	return Norm(&array, ord)
}

// Return a copy of the array, normalized to sum to 1
func (array *sparseCooF64Matrix) Normalize() NDArray {
	return Normalize(array)
}

// Return the element-wise product of this array and one or more others
func (array sparseCooF64Matrix) Prod(other ...NDArray) NDArray {
	return Prod(&array, other...)
}

// Get a 1D copy of the array, in 'C' order: rightmost axes change fastest
func (array sparseCooF64Matrix) Ravel() NDArray {
	return Ravel(&array)
}

// Set the values of the items on a given row
func (array *sparseCooF64Matrix) RowSet(row int, values []float64) {
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
func (array sparseCooF64Matrix) Row(row int) []float64 {
	if row < 0 || row >= array.shape[0] {
		panic(fmt.Sprintf("Can't get row %d from a %dx%d array", row, array.shape[0], array.shape[1]))
	}
	result := make([]float64, array.shape[0])
	for col, val := range array.values[row] {
		result[col] = val
	}
	return result
}

// Get the number of rows
func (array sparseCooF64Matrix) Rows() int {
	return array.shape[0]
}

// A slice giving the size of all array dimensions
func (array sparseCooF64Matrix) Shape() []int {
	return array.shape
}

// The total number of elements in the matrix
func (array sparseCooF64Matrix) Size() int {
	return array.shape[0] * array.shape[1]
}

// Get an array containing a rectangular slice of this array.
// `from` and `to` should both have one index per axis. The indices
// in `from` and `to` define the first and just-past-last indices you wish
// to select along each axis.
func (array sparseCooF64Matrix) Slice(from []int, to []int) NDArray {
	return Slice(&array, from, to)
}

// Return a sparse coo copy of the matrix. The method will panic
// if any off-diagonal elements are nonzero.
func (array sparseCooF64Matrix) SparseCoo() Matrix {
	return array.copy()
}

// Return a sparse diag copy of the matrix. The method will panic
// if any off-diagonal elements are nonzero.
func (array sparseCooF64Matrix) SparseDiag() Matrix {
	m := SparseDiag(array.shape[0], array.shape[1])
	array.VisitNonzero(func(pos []int, value float64) bool {
		m.ItemSet(value, pos[0], pos[1])
		return true
	})
	return m
}

// Ask whether the matrix has a sparse representation (useful for optimization)
func (array sparseCooF64Matrix) Sparsity() ArraySparsity {
	return SparseCooMatrix
}

// Return the element-wise difference of this array and one or more others
func (array sparseCooF64Matrix) Sub(other ...NDArray) NDArray {
	return Sub(&array, other...)
}

// Return the sum of all array elements
func (array sparseCooF64Matrix) Sum() float64 {
	return Sum(&array)
}

// Returns the array as a matrix. This is only possible for 1D and 2D arrays;
// 1D arrays of length n are converted into n x 1 vectors.
func (array sparseCooF64Matrix) M() Matrix {
	return &array
}

// Return the same matrix, but with axes transposed. The same data is used,
// for speed and memory efficiency. Use Copy() to create a new array.
func (array sparseCooF64Matrix) T() Matrix {
	return &sparseCooF64Matrix{
		shape:     []int{array.shape[1], array.shape[0]},
		values:    array.values,
		transpose: !array.transpose,
	}
}

// Visit all matrix elements, invoking a method on each. If the method
// returns false, iteration is aborted and VisitNonzero() returns false.
// Otherwise, it returns true.
func (array sparseCooF64Matrix) Visit(f func(pos []int, value float64) bool) bool {
	for row := 0; row < array.shape[0]; row++ {
		for col := 0; col < array.shape[1]; col++ {
			if array.transpose {
				if !f([]int{row, col}, array.values[col][row]) {
					return false
				}
			} else {
				if !f([]int{row, col}, array.values[row][col]) {
					return false
				}
			}
		}
	}
	return true
}

// Visit just nonzero elements, invoking a method on each. If the method
// returns false, iteration is aborted and VisitNonzero() returns false.
// Otherwise, it returns true.
func (array sparseCooF64Matrix) VisitNonzero(f func(pos []int, value float64) bool) bool {
	for row, val := range array.values {
		for col, v := range val {
			if array.transpose {
				if !f([]int{col, row}, v) {
					return false
				}
			} else {
				if !f([]int{row, col}, v) {
					return false
				}
			}
		}
	}
	return true
}
