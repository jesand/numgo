package matrix

import (
	"fmt"
)

type cooValue struct {
	pos   [2]int
	value float64
}

// A sparse 2D Matrix with coordinate representation: each entry is stored as a
// (x, y, value) triple.
type SparseCooF64Matrix struct {
	shape     []int
	values    []cooValue
	transpose bool
}

// Return the element-wise sum of this array and one or more others
func (array SparseCooF64Matrix) Add(other ...NDArray) NDArray {
	return Add(&array, other...)
}

// Returns true if and only if all items are nonzero
func (array SparseCooF64Matrix) All() bool {
	return All(&array)
}

// Returns true if f is true for all array elements
func (array SparseCooF64Matrix) AllF(f func(v float64) bool) bool {
	return AllF(&array, f)
}

// Returns true if f is true for all pairs of array elements in the same position
func (array SparseCooF64Matrix) AllF2(f func(v1, v2 float64) bool, other NDArray) bool {
	return AllF2(&array, f, other)
}

// Returns true if and only if any item is nonzero
func (array SparseCooF64Matrix) Any() bool {
	return Any(&array)
}

// Returns true if f is true for any array element
func (array SparseCooF64Matrix) AnyF(f func(v float64) bool) bool {
	return AnyF(&array, f)
}

// Returns true if f is true for any pair of array elements in the same position
func (array SparseCooF64Matrix) AnyF2(f func(v1, v2 float64) bool, other NDArray) bool {
	return AnyF2(&array, f, other)
}

// Return the result of applying a function to all elements
func (array SparseCooF64Matrix) Apply(f func(float64) float64) NDArray {
	return Apply(&array, f)
}

// Get the matrix data as a flattened 1D array; sparse matrices will make
// a copy first.
func (array SparseCooF64Matrix) Array() []float64 {
	return array.Dense().Array()
}

// Set the values of the items on a given column
func (array SparseCooF64Matrix) ColSet(col int, values []float64) {
	if len(array.shape) != 2 {
		panic(fmt.Sprintf("Can't ColSet a %d-dim array", len(array.shape)))
	} else if col < 0 || col >= array.shape[1] {
		panic(fmt.Sprintf("ColSet can't set col %d of a %d-col array", col, array.shape[1]))
	} else if len(values) != array.shape[0] {
		panic(fmt.Sprintf("ColSet has %d rows but got %d values", array.shape[0], len(values)))
	}
	for row := 0; row < array.shape[0]; row++ {
		array.ItemSet(values[row], row, col)
	}
}

// Get a particular column for read-only access. May or may not be a copy.
func (array SparseCooF64Matrix) Col(col int) []float64 {
	if len(array.shape) != 2 {
		panic(fmt.Sprintf("Can't get columns for a %d-dim array", len(array.shape)))
	} else if col < 0 || col >= array.shape[1] {
		panic(fmt.Sprintf("Can't get column %d from a %dx%d array", col, array.shape[0], array.shape[1]))
	}
	result := make([]float64, array.shape[1])
	for _, v := range array.values {
		if v.pos[1] == col {
			result[v.pos[0]] = v.value
		}
	}
	return result
}

// Get the number of columns
func (array SparseCooF64Matrix) Cols() int {
	return array.shape[1]
}

// Create a new array by concatenating this with another array along the
// specified axis. The array shapes must be equal along all other axes.
// It is legal to add a new axis.
func (array SparseCooF64Matrix) Concat(axis int, others ...NDArray) NDArray {
	return Concat(axis, &array, others...)
}

// Returns a duplicate of this array
func (array SparseCooF64Matrix) Copy() NDArray {
	return array.copy()
}

// Returns a duplicate of this array, preserving type
func (array SparseCooF64Matrix) copy() *SparseCooF64Matrix {
	result := &SparseCooF64Matrix{
		shape:  make([]int, len(array.shape)),
		values: make([]cooValue, len(array.values)),
	}
	if array.transpose {
		result.shape[0], result.shape[1] = array.shape[1], array.shape[0]
		for i, v := range array.values {
			result.values[i] = cooValue{
				pos:   [2]int{v.pos[1], v.pos[0]},
				value: v.value,
			}
		}
	} else {
		copy(result.shape[:], array.shape[:])
		copy(result.values[:], array.values[:])
	}
	return result
}

// Counts the number of nonzero elements in the array
func (array SparseCooF64Matrix) CountNonzero() int {
	count := 0
	for _, v := range array.values {
		if v.value != 0 {
			count++
		}
	}
	return count
}

// Returns a dense copy of the array
func (array SparseCooF64Matrix) Dense() NDArray {
	var result NDArray
	if array.transpose {
		result = Dense(array.shape[1], array.shape[0])
	} else {
		result = Dense(array.shape[0], array.shape[1])
	}
	for _, val := range array.values {
		if array.transpose {
			result.ItemSet(val.value, val.pos[1], val.pos[0])
		} else {
			result.ItemSet(val.value, val.pos[0], val.pos[1])
		}
	}
	return result
}

// Get a column vector containing the main diagonal elements of the matrix
func (array SparseCooF64Matrix) Diag() Matrix {
	if len(array.shape) != 2 {
		panic(fmt.Sprintf("Can't take diag of a %d-dim array", len(array.shape)))
	}
	size := array.shape[0]
	if array.shape[1] < size {
		size = array.shape[1]
	}
	result := Dense(size, 1).M()
	for _, v := range array.values {
		if v.pos[0] == v.pos[1] {
			result.ItemSet(v.value, v.pos[0], 0)
		}
	}
	return result
}

// Return the element-wise quotient of this array and one or more others.
// This function defines 0 / 0 = 0, so it's useful for sparse arrays.
func (array SparseCooF64Matrix) Div(other ...NDArray) NDArray {
	return Div(&array, other...)
}

// Returns true if and only if all elements in the two arrays are equal
func (array SparseCooF64Matrix) Equal(other NDArray) bool {
	return Equal(&array, other)
}

// Set all array elements to the given value
func (array SparseCooF64Matrix) Fill(value float64) {
	panic("Can't Fill() a sparse coo matrix")
}

// Get an array element in a flattened verison of this array
func (array SparseCooF64Matrix) FlatItem(index int) float64 {
	// This is ok with transpose, because array.Item does transposition for us
	nd := flatToNd(array.shape, index)
	return array.Item(nd[0], nd[1])
}

// Set an array element in a flattened version of this array
func (array *SparseCooF64Matrix) FlatItemSet(value float64, index int) {
	nd := flatToNd(array.shape, index)
	if array.transpose {
		array.ItemSet(value, nd[1], nd[0])
	}
	array.ItemSet(value, nd[0], nd[1])
}

// Return an iterator over populated matrix entries
func (array *SparseCooF64Matrix) FlatIter() FlatNDArrayIterator {
	return &sparseCooIterator{
		array:    array,
		valuePos: 0,
	}
}

// Get the matrix inverse
func (array SparseCooF64Matrix) Inverse() Matrix {
	return Inverse(&array)
}

// Get an array element
func (array SparseCooF64Matrix) Item(index ...int) float64 {
	if len(index) != 2 || index[0] >= array.shape[0] || index[1] >= array.shape[1] {
		panic(fmt.Sprintf("Item indices %v invalid for array shape %v", index, array.shape))
	}
	if array.transpose {
		index[0], index[1] = index[1], index[0]
	}
	// TODO(jesand): this linear search should be done much more quickly
	for _, v := range array.values {
		if v.pos[0] == index[0] && v.pos[1] == index[1] {
			return v.value
		}
	}
	return 0
}

// Add a scalar value to each array element
func (array *SparseCooF64Matrix) ItemAdd(value float64) NDArray {
	sh := []int{array.shape[0], array.shape[1]}
	if array.transpose {
		sh[0], sh[1] = sh[1], sh[0]
	}
	result := WithValue(value, sh...)
	for _, v := range array.values {
		if array.transpose {
			result.ItemSet(v.value+value, v.pos[1], v.pos[0])
		} else {
			result.ItemSet(v.value+value, v.pos[0], v.pos[1])
		}
	}
	return result
}

// Divide each array element by a scalar value
func (array *SparseCooF64Matrix) ItemDiv(value float64) NDArray {
	sh := []int{array.shape[0], array.shape[1]}
	if array.transpose {
		sh[0], sh[1] = sh[1], sh[0]
	}
	result := Dense(sh...)
	for _, v := range array.values {
		if array.transpose {
			result.ItemSet(v.value/value, v.pos[1], v.pos[0])
		} else {
			result.ItemSet(v.value/value, v.pos[0], v.pos[1])
		}
	}
	return result
}

// Multiply each array element by a scalar value
func (array *SparseCooF64Matrix) ItemProd(value float64) NDArray {
	sh := []int{array.shape[0], array.shape[1]}
	if array.transpose {
		sh[0], sh[1] = sh[1], sh[0]
	}
	result := Dense(sh...)
	for _, v := range array.values {
		if array.transpose {
			result.ItemSet(v.value*value, v.pos[1], v.pos[0])
		} else {
			result.ItemSet(v.value*value, v.pos[0], v.pos[1])
		}
	}
	return result
}

// Subtract a scalar value from each array element
func (array *SparseCooF64Matrix) ItemSub(value float64) NDArray {
	sh := []int{array.shape[0], array.shape[1]}
	if array.transpose {
		sh[0], sh[1] = sh[1], sh[0]
	}
	result := WithValue(-value, sh...)
	for _, v := range array.values {
		if array.transpose {
			result.ItemSet(v.value-value, v.pos[1], v.pos[0])
		} else {
			result.ItemSet(v.value-value, v.pos[0], v.pos[1])
		}
	}
	return result
}

// Set an array element
func (array *SparseCooF64Matrix) ItemSet(value float64, index ...int) {
	if array.transpose {
		index[0], index[1] = index[1], index[0]
	}
	// TODO(jesand): this linear search should be done much more quickly
	for idx, v := range array.values {
		if v.pos[0] == index[0] && v.pos[1] == index[1] {
			array.values[idx].value = value
			return
		}
	}
	if value != 0 {
		array.values = append(array.values, cooValue{
			pos:   [2]int{index[0], index[1]},
			value: value,
		})
	}
}

// Return an iterator over populated matrix entries
func (array *SparseCooF64Matrix) Iter() CoordNDArrayIterator {
	return &sparseCooIterator{
		array:    array,
		valuePos: 0,
	}
}

// Solve for x, where ax = b.
func (array SparseCooF64Matrix) LDivide(b Matrix) Matrix {
	return LDivide(&array, b)
}

// Get the result of matrix multiplication between this and some other
// array(s). All arrays must have two dimensions, and the dimensions must
// be aligned correctly for multiplication.
// If A is m x p and B is p x n, then C = A.MProd(B) is the m x n matrix
// with C[i, j] = \sum_{k=1}^p A[i,k] * B[k,j].
func (array SparseCooF64Matrix) MProd(others ...Matrix) Matrix {
	return MProd(&array, others...)
}

// Get the value of the largest array element
func (array SparseCooF64Matrix) Max() float64 {
	return Max(&array)
}

// Get the value of the smallest array element
func (array SparseCooF64Matrix) Min() float64 {
	return Min(&array)
}

// The number of dimensions in the matrix
func (array SparseCooF64Matrix) NDim() int {
	return len(array.shape)
}

// Get the matrix norm of the specified ordinality (1, 2, infinity, ...)
func (array SparseCooF64Matrix) Norm(ord float64) float64 {
	return Norm(&array, ord)
}

// Return a copy of the array, normalized to sum to 1
func (array *SparseCooF64Matrix) Normalize() NDArray {
	return Normalize(array)
}

// Return the element-wise product of this array and one or more others
func (array SparseCooF64Matrix) Prod(other ...NDArray) NDArray {
	return Prod(&array, other...)
}

// Get a 1D copy of the array, in 'C' order: rightmost axes change fastest
func (array SparseCooF64Matrix) Ravel() NDArray {
	return Ravel(&array)
}

// Set the values of the items on a given row
func (array SparseCooF64Matrix) RowSet(row int, values []float64) {
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

// Get a particular row for read-only access. May or may not be a copy.
func (array SparseCooF64Matrix) Row(row int) []float64 {
	if len(array.shape) != 2 {
		panic(fmt.Sprintf("Can't get rows for a %d-dim array", len(array.shape)))
	} else if row < 0 || row >= array.shape[0] {
		panic(fmt.Sprintf("Can't get row %d from a %dx%d array", row, array.shape[0], array.shape[1]))
	}
	result := make([]float64, array.shape[0])
	for _, v := range array.values {
		if v.pos[0] == row {
			result[v.pos[1]] = v.value
		}
	}
	return result
}

// Get the number of rows
func (array SparseCooF64Matrix) Rows() int {
	return array.shape[0]
}

// A slice giving the size of all array dimensions
func (array SparseCooF64Matrix) Shape() []int {
	return array.shape
}

// The total number of elements in the matrix
func (array SparseCooF64Matrix) Size() int {
	return array.shape[0] * array.shape[1]
}

// Get an array containing a rectangular slice of this array.
// `from` and `to` should both have one index per axis. The indices
// in `from` and `to` define the first and just-past-last indices you wish
// to select along each axis.
func (array SparseCooF64Matrix) Slice(from []int, to []int) NDArray {
	return Slice(&array, from, to)
}

// Ask whether the matrix has a sparse representation (useful for optimization)
func (array SparseCooF64Matrix) Sparsity() ArraySparsity {
	return SparseCooMatrix
}

// Return the element-wise difference of this array and one or more others
func (array SparseCooF64Matrix) Sub(other ...NDArray) NDArray {
	return Sub(&array, other...)
}

// Return the sum of all array elements
func (array SparseCooF64Matrix) Sum() float64 {
	return Sum(&array)
}

// Returns the array as a matrix. This is only possible for 1D and 2D arrays;
// 1D arrays of length n are converted into n x 1 vectors.
func (array SparseCooF64Matrix) M() Matrix {
	return &array
}

// Return the same matrix, but with axes transposed. The same data is used,
// for speed and memory efficiency. Use Copy() to create a new array.
// A 1D array is unchanged; create a 2D analog to rotate a vector.
func (array SparseCooF64Matrix) T() NDArray {
	return &SparseCooF64Matrix{
		shape:     []int{array.shape[1], array.shape[0]},
		values:    array.values,
		transpose: !array.transpose,
	}
}

// Iterates over all array elements
type sparseCooIterator struct {
	array    *SparseCooF64Matrix
	valuePos int
}

// Ask whether there are more values to iterate over.
func (iter sparseCooIterator) HasNext() bool {
	return iter.valuePos < len(iter.array.values)
}

// Return the value and coordinates of the next entry
func (iter *sparseCooIterator) Next() (float64, []int) {
	if iter.valuePos >= len(iter.array.values) {
		return 0, nil
	}
	pos := iter.valuePos
	iter.valuePos++
	return iter.array.values[pos].value, iter.array.values[pos].pos[:]
}

// Return the value and flat index of the next entry
func (iter *sparseCooIterator) FlatNext() (float64, int) {
	if iter.valuePos >= len(iter.array.values) {
		return 0, 0
	}
	pos := iter.valuePos
	iter.valuePos++
	return iter.array.values[pos].value, ndToFlat(iter.array.shape, iter.array.values[pos].pos[:])
}
