package matrix

import (
	"fmt"
	"math"
)

// Get the flat index for the specified indices. Negative indexing is supported:
// an index of -1 refers to the final array element.
func ndToFlat(shape []int, index []int) int {
	if len(index) != len(shape) {
		panic(fmt.Sprintf("Indices %v invalid for array shape %v", index, shape))
	}
	flat := 0
	for i := range shape {
		if index[i] >= shape[i] || index[i] < -shape[i] {
			panic(fmt.Sprintf("Indices %v invalid for array shape %v", index, shape))
		} else if index[i] < 0 {
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
	size := 1
	for _, v := range shape {
		size *= v
	}
	if flat >= size || flat < -size {
		panic(fmt.Sprintf("Flat index %v invalid for array shape %v", flat, shape))
	}
	if flat < 0 {
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
	var result NDArray
	sp := array.Sparsity()
	sh := array.Shape()
	for _, o := range others {
		switch o.Sparsity() {
		case DenseArray:
			sp = DenseArray
		case SparseCooMatrix:
			if sp == SparseDiagMatrix {
				sp = SparseCooMatrix
			}
		}
		sh2 := o.Shape()
		if len(sh2) != len(sh) {
			panic(fmt.Sprintf("Can't add arrays with shapes %v and %v", sh, sh2))
		}
		for i := range sh {
			if sh[i] != sh2[i] {
				panic(fmt.Sprintf("Can't add arrays with shapes %v and %v", sh, sh2))
			}
		}
	}

	switch sp {
	case array.Sparsity():
		result = array.Copy()
	case DenseArray:
		result = array.Dense()
	case SparseCooMatrix:
		result = SparseCoo(sh[0], sh[1])
		array.VisitNonzero(func(pos []int, value float64) bool {
			result.ItemSet(value, pos...)
			return true
		})
	}
	for _, o := range others {
		o.VisitNonzero(func(pos []int, value float64) bool {
			result.ItemSet(result.Item(pos...)+value, pos...)
			return true
		})
	}
	return result
}

// Returns true if and only if all items are nonzero
func All(array NDArray) bool {
	switch array.Sparsity() {
	case SparseDiagMatrix:
		return false
	default:
		return array.CountNonzero() == array.Size()
	}
}

// Returns true if f is true for all array elements
func AllF(array NDArray, f func(v float64) bool) bool {
	counted := 0
	all := array.VisitNonzero(func(pos []int, value float64) bool {
		counted++
		if !f(value) {
			return false
		}
		return true
	})
	if !all || (counted < array.Size() && !f(0)) {
		return false
	}
	return true
}

// Returns true if f is true for all pairs of array elements in the same position
func AllF2(array NDArray, f func(v1, v2 float64) bool, other NDArray) bool {
	sh1 := array.Shape()
	sh2 := other.Shape()
	if len(sh1) != len(sh2) {
		panic("AllF2() requires two arrays of the same shape")
	}
	for i := 0; i < len(sh1); i++ {
		if sh1[i] != sh2[i] {
			panic("AllF2() requires two arrays of the same shape")
		}
	}
	size := array.Size()
	for i := 0; i < size; i++ {
		if !f(array.FlatItem(i), other.FlatItem(i)) {
			return false
		}
	}
	return true
}

// Returns true if and only if any item is nonzero
func Any(array NDArray) bool {
	return !array.VisitNonzero(func(pos []int, value float64) bool {
		return false
	})
}

// Returns true if f is true for any array element
func AnyF(array NDArray, f func(v float64) bool) bool {
	counted := 0
	allFalse := array.VisitNonzero(func(pos []int, value float64) bool {
		counted++
		if f(value) {
			return false
		}
		return true
	})
	if !allFalse || (counted < array.Size() && f(0)) {
		return true
	}
	return false
}

// Returns true if f is true for any pair of array elements in the same position
func AnyF2(array NDArray, f func(v1, v2 float64) bool, other NDArray) bool {
	sh1 := array.Shape()
	sh2 := other.Shape()
	if len(sh1) != len(sh2) {
		panic("AnyF2() requires two arrays of the same shape")
	}
	for i := 0; i < len(sh1); i++ {
		if sh1[i] != sh2[i] {
			panic("AnyF2() requires two arrays of the same shape")
		}
	}
	size := array.Size()
	for i := 0; i < size; i++ {
		if f(array.FlatItem(i), other.FlatItem(i)) {
			return true
		}
	}
	return false
}

// Return the result of applying a function to all elements
func Apply(array NDArray, f func(float64) float64) NDArray {
	result := array.Dense()
	size := result.Size()
	for i := 0; i < size; i++ {
		value := f(result.FlatItem(i))
		result.FlatItemSet(value, i)
	}
	return result
}

// Create a new array by concatenating this with one or more others along the
// specified axis. The array shapes must be equal along all other axes.
// It is legal to add a new axis.
func Concat(axis int, array NDArray, others ...NDArray) NDArray {
	if len(others) < 1 {
		return array.Copy()
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

// Treat the rows as points, and get the pairwise distance between them.
// Returns a distance matrix D such that D_i,j is the distance between
// rows i and j.
func Dist(m Matrix, t DistType) Matrix {
	var dist = Dense(m.Rows(), m.Rows()).M()
	for i := 1; i < m.Rows(); i++ {
		ri := m.Row(i)
		for j := 0; j <= i; j++ {
			rj := m.Row(j)
			var v float64
			switch t {
			case EuclideanDist:
				for idx, riv := range ri {
					v += math.Pow(riv-rj[idx], 2)
				}
				v = math.Sqrt(v)
			default:
				panic(fmt.Sprintf("Can't calculate distance of invalid type %v", t))
			}
			dist.ItemSet(v, i, j)
			dist.ItemSet(v, j, i)
		}
	}
	return dist
}

// Return the element-wise quotient of this array and one or more others.
// This function defines 0 / 0 = 0, so it's useful for sparse arrays.
func Div(array NDArray, others ...NDArray) NDArray {
	sh := array.Shape()
	for _, o := range others {
		sh2 := o.Shape()
		if len(sh2) != len(sh) {
			panic(fmt.Sprintf("Can't divide arrays with shapes %v and %v", sh, sh2))
		}
		for i := range sh {
			if sh[i] != sh2[i] {
				panic(fmt.Sprintf("Can't divide arrays with shapes %v and %v", sh, sh2))
			}
		}
	}

	result := array.Copy()
	for _, o := range others {
		result.VisitNonzero(func(pos []int, value float64) bool {
			result.ItemSet(value/o.Item(pos...), pos...)
			return true
		})
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
	for idx := 0; idx < size; idx++ {
		if array.FlatItem(idx) != other.FlatItem(idx) {
			return false
		}
	}
	return true
}

// Set all array elements to the given value
func Fill(array NDArray, value float64) {
	if array.Sparsity() != DenseArray {
		panic("Can't Fill() a sparse array")
	}
	size := array.Size()
	for idx := 0; idx < size; idx++ {
		array.FlatItemSet(value, idx)
	}
}

// Add a scalar value to each array element
func ItemAdd(array NDArray, value float64) NDArray {
	if value == 0 {
		return array.Copy()
	}
	result := array.Dense()
	size := result.Size()
	for idx := 0; idx < size; idx++ {
		result.FlatItemSet(result.FlatItem(idx)+value, idx)
	}
	return result
}

// Divide each array element by a scalar value
func ItemDiv(array NDArray, value float64) NDArray {
	if value == 1 {
		return array.Copy()
	}
	result := array.Copy()
	result.VisitNonzero(func(pos []int, v float64) bool {
		result.ItemSet(v/value, pos...)
		return true
	})
	return result
}

// Multiply each array element by a scalar value
func ItemProd(array NDArray, value float64) NDArray {
	if value == 1 {
		return array.Copy()
	}
	result := array.Copy()
	result.VisitNonzero(func(pos []int, v float64) bool {
		result.ItemSet(v*value, pos...)
		return true
	})
	return result
}

// Subtract a scalar value from each array element
func ItemSub(array NDArray, value float64) NDArray {
	if value == 0 {
		return array.Copy()
	}
	result := array.Dense()
	size := result.Size()
	for idx := 0; idx < size; idx++ {
		result.FlatItemSet(result.FlatItem(idx)-value, idx)
	}
	return result
}

// Get the result of matrix multiplication between this and some other
// array(s). All arrays must have two dimensions, and the dimensions must
// be aligned correctly for multiplication.
// If A is m x p and B is p x n, then C = A.MProd(B) is the m x n matrix
// with C[i, j] = \sum_{k=1}^p A[i,k] * B[k,j].
func MProd(array Matrix, others ...Matrix) Matrix {
	if len(others) < 1 {
		return array.Copy().M()
	}
	var (
		left   = array
		leftSh = array.Shape()
		leftSp = array.Sparsity()
		result Matrix
	)
	for _, right := range others {
		rightSh := right.Shape()
		rightSp := right.Sparsity()
		if leftSh[1] != rightSh[0] {
			panic(fmt.Sprintf("Can't MProd a %dx%d to a %dx%d array; inner dimensions must match", leftSh[0], leftSh[1], rightSh[0], rightSh[1]))
		}

		if leftSp == SparseDiagMatrix {
			lDiag := left.Diag().Array()
			switch rightSp {
			case SparseDiagMatrix:
				rDiag := right.Diag().Array()
				resDiag := make([]float64, len(lDiag))
				for idx, v := range lDiag {
					resDiag[idx] = v * rDiag[idx]
				}
				result = Diag(resDiag...)
			case SparseCooMatrix:
				result = SparseCoo(leftSh[0], rightSh[1])
				spRes := result.(*sparseCooF64Matrix)
				right.VisitNonzero(func(pos []int, value float64) bool {
					spRes.values[pos[0]][pos[1]] += lDiag[pos[0]] * value
					return true
				})
			default:
				result = Dense(leftSh[0], rightSh[1]).M()
				resArr := result.Array()
				rArr := right.Array()
				for i := 0; i < leftSh[0]; i++ {
					for j := 0; j < rightSh[1]; j++ {
						resArr[i*rightSh[1]+j] = lDiag[i] * rArr[i*rightSh[1]+j]
					}
				}
			}

		} else if leftSp == SparseCooMatrix && rightSp == SparseCooMatrix {
			result = SparseCoo(leftSh[0], rightSh[1])
			spRes := result.(*sparseCooF64Matrix)
			spRight := right.(*sparseCooF64Matrix)
			left.VisitNonzero(func(pos []int, value float64) bool {
				for j := 0; j < rightSh[1]; j++ {
					spRes.values[pos[0]][j] += value * spRight.values[pos[1]][j]
				}
				return true
			})

		} else if rightSp == SparseDiagMatrix {
			rDiag := right.Diag().Array()
			if leftSp == SparseCooMatrix {
				resArr := make([]float64, leftSh[0]*rightSh[1])
				left.VisitNonzero(func(pos []int, value float64) bool {
					resArr[pos[0]*rightSh[1]+pos[1]] += value * rDiag[pos[1]]
					return true
				})
				result = SparseCoo(leftSh[0], rightSh[1])
				for idx, v := range resArr {
					if v != 0 {
						result.FlatItemSet(v, idx)
					}
				}
			} else {
				result = Dense(leftSh[0], rightSh[1]).M()
				resArr := result.Array()
				lArr := left.Array()
				for i := 0; i < leftSh[0]; i++ {
					for j := 0; j < rightSh[1]; j++ {
						resArr[i*rightSh[1]+j] = lArr[i*leftSh[1]+j] * rDiag[j]
					}
				}
			}

		} else {
			result = Dense(leftSh[0], rightSh[1]).M()
			resArr := result.Array()
			lArr := left.Array()
			rArr := right.Array()
			for i := 0; i < leftSh[0]; i++ {
				for j := 0; j < rightSh[1]; j++ {
					value := 0.0
					for k := 0; k < leftSh[1]; k++ {
						value += lArr[i*leftSh[1]+k] * rArr[k*rightSh[1]+j]
					}
					resArr[i*rightSh[1]+j] = value
				}
			}
		}

		left = result
		leftSh = result.Shape()
		leftSp = result.Sparsity()
	}
	return result
}

// Get the value of the largest array element
func Max(array NDArray) float64 {
	max := math.Inf(-1)
	counted := 0
	array.VisitNonzero(func(pos []int, value float64) bool {
		counted++
		if value > max {
			max = value
		}
		return true
	})
	if max < 0 && counted < array.Size() {
		max = 0
	}
	return max
}

// Get the value of the smallest array element
func Min(array NDArray) float64 {
	min := math.Inf(+1)
	counted := 0
	array.VisitNonzero(func(pos []int, value float64) bool {
		counted++
		if value < min {
			min = value
		}
		return true
	})
	if min > 0 && counted < array.Size() {
		min = 0
	}
	return min
}

// Return a copy of the array, normalized to sum to 1
func Normalize(array NDArray) NDArray {
	s := array.Sum()
	if s != 0 && s != 1 {
		return array.ItemDiv(s)
	} else {
		return array.Copy()
	}
}

// Return the element-wise product of this array and one or more others
func Prod(array NDArray, others ...NDArray) NDArray {
	sh := array.Shape()
	for _, o := range others {
		sh2 := o.Shape()
		if len(sh2) != len(sh) {
			panic(fmt.Sprintf("Can't multiply arrays with shapes %v and %v", sh, sh2))
		}
		for i := range sh {
			if sh[i] != sh2[i] {
				panic(fmt.Sprintf("Can't multiply arrays with shapes %v and %v", sh, sh2))
			}
		}
	}

	result := array.Copy()
	for _, o := range others {
		result.VisitNonzero(func(pos []int, value float64) bool {
			result.ItemSet(value*o.Item(pos...), pos...)
			return true
		})
	}
	return result
}

// Get a 1D copy of the array, in 'C' order: rightmost axes change fastest
func Ravel(array NDArray) NDArray {
	result := Dense(array.Size())
	shape := array.Shape()
	array.VisitNonzero(func(pos []int, value float64) bool {
		result.ItemSet(value, ndToFlat(shape, pos))
		return true
	})
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
	var result NDArray
	sp := array.Sparsity()
	sh := array.Shape()
	for _, o := range others {
		switch o.Sparsity() {
		case DenseArray:
			sp = DenseArray
		case SparseCooMatrix:
			if sp == SparseDiagMatrix {
				sp = SparseCooMatrix
			}
		}
		sh2 := o.Shape()
		if len(sh2) != len(sh) {
			panic(fmt.Sprintf("Can't add arrays with shapes %v and %v", sh, sh2))
		}
		for i := range sh {
			if sh[i] != sh2[i] {
				panic(fmt.Sprintf("Can't add arrays with shapes %v and %v", sh, sh2))
			}
		}
	}

	switch sp {
	case array.Sparsity():
		result = array.Copy()
	case DenseArray:
		result = array.Dense()
	case SparseCooMatrix:
		result = SparseCoo(sh[0], sh[1])
		array.VisitNonzero(func(pos []int, value float64) bool {
			result.ItemSet(value, pos...)
			return true
		})
	}
	for _, o := range others {
		o.VisitNonzero(func(pos []int, value float64) bool {
			result.ItemSet(result.Item(pos...)-value, pos...)
			return true
		})
	}
	return result
}

// Return the sum of all array elements
func Sum(array NDArray) float64 {
	var result float64
	array.VisitNonzero(func(pos []int, value float64) bool {
		result += value
		return true
	})
	return result
}
