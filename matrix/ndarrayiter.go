package matrix

// An iterator over populated matrix entries. Useful for generalizing operations
// over various dense and sparse representations. Any skipped entries can be
// assumed to be zero, but entries which are not skipped may also be zero.
type CoordNDArrayIterator interface {

	// Ask whether there are more values to iterate over.
	HasNext() bool

	// Return the value and coordinates of the next entry
	Next() (float64, []int)
}

// An iterator over populated matrix entries. Useful for generalizing operations
// over various dense and sparse representations. Any skipped entries can be
// assumed to be zero, but entries which are not skipped may also be zero.
type FlatNDArrayIterator interface {

	// Ask whether there are more values to iterate over.
	HasNext() bool

	// Return the value and flat index of the next entry
	FlatNext() (float64, int)
}
