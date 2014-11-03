package matrix

import (
	"fmt"
	"github.com/gonum/matrix/mat64"
)

// A two dimensional array with some special functionality
type Matrix interface {
	NDArray

	// Get the matrix data as a 1D array
	Array() []float64

	// Set the values of the items on a given column
	ColSet(col int, values []float64)

	// Get the number of columns
	Cols() int

	// Get the matrix inverse
	Inverse() Matrix

	// Solve for x, where ax = b and a is `this`.
	LDivide(b Matrix) Matrix

	// Get the result of matrix multiplication between this and some other
	// matrices. Matrix dimensions must be aligned correctly for multiplication.
	// If A is m x p and B is p x n, then C = A.MProd(B) is the m x n matrix
	// with C[i, j] = \sum_{k=1}^p A[i,k] * B[k,j].
	MProd(others ...Matrix) Matrix

	// Get the matrix norm of the specified ordinality (1, 2, infinity, ...)
	Norm(ord float64) float64

	// Set the values of the items on a given row
	RowSet(row int, values []float64)

	// Get the number of rows
	Rows() int
}

// Create a matrix from literal data
func M(shape []int, array ...float64) Matrix {
	if len(shape) != 2 {
		panic(fmt.Sprintf("A matrix should be 2D, not %dD", len(shape)))
	}
	return A(shape, array...).ToMatrix()
}

// Convert our matrix type to mat64's matrix type
func toMat64(m Matrix) *mat64.Dense {
	return mat64.NewDense(m.Rows(), m.Cols(), m.Array())
}

// Convert mat64's matrix type to our matrix type
func toMatrix(m mat64.Matrix) Matrix {
	rows, cols := m.Dims()
	array := &DenseF64Array{
		shape: []int{rows, cols},
		array: make([]float64, rows*cols),
	}
	for i0 := 0; i0 < rows; i0++ {
		for i1 := 0; i1 < cols; i1++ {
			array.ItemSet(m.At(i0, i1), i0, i1)
		}
	}
	return array
}

// Get the matrix inverse
func Inverse(a Matrix) Matrix {
	inv := mat64.Inverse(toMat64(a))
	return toMatrix(inv)
}

// Solve for x, where ax = b.
func LDivide(a, b Matrix) Matrix {
	x := mat64.Solve(toMat64(a), toMat64(b))
	return toMatrix(x)
}

// Get the matrix norm of the specified ordinality (1, 2, infinity, ...)
func Norm(m Matrix, ord float64) float64 {
	return toMat64(m).Norm(ord)
}

// Solve is an alias for LDivide
func Solve(a, b Matrix) Matrix {
	return LDivide(a, b)
}
