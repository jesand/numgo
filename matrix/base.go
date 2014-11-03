package matrix

import (
	"fmt"
	"math"
)

// Get the flat index for the specified indices. Negative indexing is supported:
// an index of -1 refers to the final array element.
func ndToFlat(shape []int, index []int) int {
	if len(index) != len(shape) {
		panic(fmt.Sprintf("Wrong number of indices for %d-dim array", len(shape)))
	}
	flat := 0
	for i := range shape {
		if index[i] < 0 {
			flat += index[i] + shape[i]
		} else {
			flat += index[i]
		}
		if i < len(shape)-1 {
			flat *= shape[i+1]
		}
	}
	return flat
}

// Get the indices for the specified flat index.
func flatToNd(shape []int, flat int) []int {
	if flat < 0 {
		size := 1
		for _, v := range shape {
			size *= v
		}
		flat += size
	}
	index := make([]int, len(shape))
	for axis := 0; axis < len(index); axis++ {
		index[axis] = flat
		for a2 := axis + 1; a2 < len(index); a2++ {
			index[axis] /= shape[a2]
		}
		index[axis] %= shape[axis]
	}
	return index
}

// Return the element-wise sum of this array and one or more others
func Add(array NDArray, others ...NDArray) NDArray {
	result := array.Copy()
	sh := array.Shape()
	for _, o := range others {
		sh2 := o.Shape()
		if len(sh2) != len(sh) {
			panic("Can't add arrays with differing dimensionality")
		}
		for i := range sh {
			if sh[i] != sh2[i] {
				panic("Can't add arrays with differing dimensionality")
			}
		}
	}
	size := array.Size()
	for _, o := range others {
		for idx := 0; idx < size; idx++ {
			result.FlatItemSet(result.FlatItem(idx)+o.FlatItem(idx), idx)
		}
	}
	return result
}

// Returns true if and only if all items are nonzero
func All(array NDArray) bool {
	size := array.Size()
	if size == 0 {
		return false
	}
	for i := 0; i < size; i++ {
		if array.FlatItem(i) == 0 {
			return false
		}
	}
	return true
}

// Returns true if and only if any item is nonzero
func Any(array NDArray) bool {
	size := array.Size()
	for i := 0; i < size; i++ {
		if array.FlatItem(i) != 0 {
			return true
		}
	}
	return false
}

// Apply a function to all elements
func Apply(array NDArray, f func(float64) float64) {
	size := array.Size()
	for i := 0; i < size; i++ {
		value := f(array.FlatItem(i))
		array.FlatItemSet(value, i)
	}
}

// Create a new array by concatenating this with one or more others along the
// specified axis. The array shapes must be equal along all other axes.
// It is legal to add a new axis.
func Concat(axis int, array NDArray, others ...NDArray) NDArray {
	if len(others) < 1 {
		return array
	}

	// Calculate the new array shape
	shs := make([][]int, 1+len(others))
	shs[0] = array.Shape()
	if axis > len(shs[0]) {
		panic(fmt.Sprintf("Can't concat %d-d arrays along invalid axis %d", len(shs[0]), axis))
	}
	for i := 1; i < len(shs); i++ {
		shs[i] = others[i-1].Shape()
		if len(shs[0]) != len(shs[i]) {
			panic(fmt.Sprintf("Can't concat arrays with %d and %d dims", len(shs[0]), len(shs[i])))
		}
	}

	var shOut []int
	if axis == len(shs[0]) {
		shOut = make([]int, len(shs[0])+1)
	} else {
		shOut = make([]int, len(shs[0]))
	}
	for i := range shOut {
		if i != axis {
			for j := 1; j < len(shs); j++ {
				if shs[0][i] != shs[j][i] {
					panic(fmt.Sprintf("Can't concat arrays along axis %d with unequal size on axis %d", axis, i))
				}
			}
			shOut[i] = shs[0][i]
		} else if i < len(shs[0]) {
			for j := 0; j < len(shs); j++ {
				shOut[i] += shs[j][i]
			}
		} else {
			shOut[i] = len(shs)
		}
	}
	result := Dense(shOut...)

	// Copy the arrays
	size := result.Size()
	var (
		value float64
		src   []int
	)
	for i := 0; i < size; i++ {
		src = flatToNd(shOut, i)
		if axis < len(shs[0]) {
			// Not creating a new dimension
			for j := 0; j < len(shs); j++ {
				if src[axis] >= shs[j][axis] {
					src[axis] -= shs[j][axis]
				} else if j == 0 {
					value = array.Item(src...)
					break
				} else {
					value = others[j-1].Item(src...)
					break
				}
			}
		} else if src[axis] == 0 {
			value = array.Item(src[:axis]...)
		} else {
			value = others[src[axis]-1].Item(src[:axis]...)
		}

		result.FlatItemSet(value, i)
	}

	return result
}

// Return the element-wise quotient of this array and one or more others
func Div(array NDArray, others ...NDArray) NDArray {
	result := array.Copy()
	sh := array.Shape()
	for _, o := range others {
		sh2 := o.Shape()
		if len(sh2) != len(sh) {
			panic("Can't add arrays with differing dimensionality")
		}
		for i := range sh {
			if sh[i] != sh2[i] {
				panic("Can't add arrays with differing dimensionality")
			}
		}
	}
	size := array.Size()
	for _, o := range others {
		for idx := 0; idx < size; idx++ {
			result.FlatItemSet(result.FlatItem(idx)/o.FlatItem(idx), idx)
		}
	}
	return result
}

// Returns true if and only if all elements in the two arrays are equal
func Equal(array, other NDArray) bool {
	sh1 := array.Shape()
	sh2 := other.Shape()
	if len(sh1) != len(sh2) {
		return false
	}
	for d := 0; d < len(sh1); d++ {
		if sh1[d] != sh2[d] {
			return false
		}
	}
	size := array.Size()
	for i := 0; i < size; i++ {
		if array.FlatItem(i) != other.FlatItem(i) {
			return false
		}
	}
	return true
}

// Set all array elements to the given value
func Fill(array NDArray, value float64) {
	size := array.Size()
	for idx := 0; idx < size; idx++ {
		array.FlatItemSet(value, idx)
	}
}

// Add a scalar value to each array element
func ItemAdd(array NDArray, value float64) {
	size := array.Size()
	for idx := 0; idx < size; idx++ {
		array.FlatItemSet(array.FlatItem(idx)+value, idx)
	}
}

// Divide each array element by a scalar value
func ItemDiv(array NDArray, value float64) {
	size := array.Size()
	for idx := 0; idx < size; idx++ {
		array.FlatItemSet(array.FlatItem(idx)/value, idx)
	}
}

// Multiply each array element by a scalar value
func ItemProd(array NDArray, value float64) {
	size := array.Size()
	for idx := 0; idx < size; idx++ {
		array.FlatItemSet(array.FlatItem(idx)*value, idx)
	}
}

// Subtract a scalar value from each array element
func ItemSub(array NDArray, value float64) {
	size := array.Size()
	for idx := 0; idx < size; idx++ {
		array.FlatItemSet(array.FlatItem(idx)-value, idx)
	}
}

// Get the result of matrix multiplication between this and some other
// array(s). All arrays must have two dimensions, and the dimensions must
// be aligned correctly for multiplication.
// If A is m x p and B is p x n, then C = A.MProd(B) is the m x n matrix
// with C[i, j] = \sum_{k=1}^p A[i,k] * B[k,j].
func MProd(array Matrix, others ...Matrix) Matrix {
	if len(others) < 1 {
		return array.Copy().ToMatrix()
	} else if array.NDim() != 2 {
		panic(fmt.Sprintf("Can't MProd on a %d-dim array; must be 2D", array.NDim()))
	}
	var (
		left   = array
		leftSh = array.Shape()
		result Matrix
	)
	for _, right := range others {
		rightSh := right.Shape()
		if len(rightSh) != 2 {
			panic(fmt.Sprintf("Can't MProd a %d-dim array; must be 2D", len(rightSh)))
		} else if leftSh[1] != rightSh[0] {
			panic(fmt.Sprintf("Can't MProd a %d x %d to a %d x %d array; inner dimensions must match", leftSh[0], leftSh[1], rightSh[0], rightSh[1]))
		}
		result = Dense(leftSh[0], rightSh[1]).ToMatrix()
		for i := 0; i < leftSh[0]; i++ {
			for j := 0; j < rightSh[1]; j++ {
				value := 0.0
				for k := 0; k < leftSh[1]; k++ {
					value += left.Item(i, k) * right.Item(k, j)
				}
				result.ItemSet(value, i, j)
			}
		}
		left = result
		leftSh = result.Shape()
	}
	return result
}

// Get the value of the largest array element
func Max(array NDArray) float64 {
	max := math.Inf(-1)
	size := array.Size()
	for i := 0; i < size; i++ {
		v := array.FlatItem(i)
		if v > max {
			max = v
		}
	}
	return max
}

// Get the value of the smallest array element
func Min(array NDArray) float64 {
	min := math.Inf(+1)
	size := array.Size()
	for i := 0; i < size; i++ {
		v := array.FlatItem(i)
		if v < min {
			min = v
		}
	}
	return min
}

// Normalize the array to sum to 1, or do nothing if all items are 0.
func Normalize(array NDArray) {
	s := array.Sum()
	if s != 0 && s != 1 {
		array.ItemProd(1 / s)
	}
}

// Return the element-wise product of this array and one or more others
func Prod(array NDArray, others ...NDArray) NDArray {
	result := array.Copy()
	sh := array.Shape()
	for _, o := range others {
		sh2 := o.Shape()
		if len(sh2) != len(sh) {
			panic("Can't add arrays with differing dimensionality")
		}
		for i := range sh {
			if sh[i] != sh2[i] {
				panic("Can't add arrays with differing dimensionality")
			}
		}
	}
	size := array.Size()
	for _, o := range others {
		for idx := 0; idx < size; idx++ {
			result.FlatItemSet(result.FlatItem(idx)*o.FlatItem(idx), idx)
		}
	}
	return result
}

// Get a 1D copy of the array, in 'C' order: rightmost axes change fastest
func Ravel(array NDArray) NDArray {
	result := Dense(array.Size())
	size := array.Size()
	for i := 0; i < size; i++ {
		result.FlatItemSet(array.FlatItem(i), i)
	}
	return result
}

// Get an array containing a rectangular slice of this array.
// `from` and `to` should both have one index per axis. The indices
// in `from` and `to` define the first and just-past-last indices you wish
// to select along each axis. You can also use negative indices to represent the
// distance from the end of the array, where -1 represents the element just past
// the end of the array.
func Slice(array NDArray, from []int, to []int) NDArray {
	sh := array.Shape()
	if len(from) != len(sh) || len(to) != len(sh) {
		panic("Invalid Slice() indices: the arguments should have the same length as the array")
	}

	// Convert negative indices
	start := make([]int, len(sh))
	for idx, v := range from {
		if v < 0 {
			start[idx] = v + sh[idx] + 1
		} else {
			start[idx] = v
		}
	}
	stop := make([]int, len(sh))
	for idx, v := range to {
		if v < 0 {
			stop[idx] = v + sh[idx] + 1
		} else {
			stop[idx] = v
		}
		if stop[idx] < start[idx] {
			panic(fmt.Sprintf("Invalid Slice() indices: %d is before %d", to[idx], from[idx]))
		}
	}

	// Create an empty array
	shape := make([]int, len(sh))
	for idx := range shape {
		shape[idx] = stop[idx] - start[idx]
	}
	result := Dense(shape...)

	// Copy the values into the new array
	size := result.Size()
	index := make([]int, len(from))
	copy(index[:], start[:])

	for i := 0; i < size; i++ {
		result.FlatItemSet(array.Item(index...), i)
		for j := len(index) - 1; j >= 0; j-- {
			index[j]++
			if index[j] == stop[j] {
				index[j] = start[j]
			} else {
				break
			}
		}
	}

	return result
}

// Return the element-wise difference of this array and one or more others
func Sub(array NDArray, others ...NDArray) NDArray {
	result := array.Copy()
	sh := array.Shape()
	for _, o := range others {
		sh2 := o.Shape()
		if len(sh2) != len(sh) {
			panic("Can't add arrays with differing dimensionality")
		}
		for i := range sh {
			if sh[i] != sh2[i] {
				panic("Can't add arrays with differing dimensionality")
			}
		}
	}
	size := array.Size()
	for _, o := range others {
		for idx := 0; idx < size; idx++ {
			result.FlatItemSet(result.FlatItem(idx)-o.FlatItem(idx), idx)
		}
	}
	return result
}

// Return the sum of all array elements
func Sum(array NDArray) float64 {
	var result float64
	size := array.Size()
	for i := 0; i < size; i++ {
		result += array.FlatItem(i)
	}
	return result
}
