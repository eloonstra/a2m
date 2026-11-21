package launcher

import "testing"

const (
	seedContextLarge  = 262144
	seedReserveNormal = 32768
	seedZero          = 0
	seedNegative      = -1
	seedNegativeBig   = -100
)

func FuzzCompactWindow(f *testing.F) {
	f.Add(seedContextLarge, seedReserveNormal)
	f.Add(seedZero, seedZero)
	f.Add(seedNegative, seedNegativeBig)
	f.Fuzz(func(t *testing.T, contextWindow, outputReserve int) {
		got := compactWindow(contextWindow, outputReserve)
		inRange := got >= minAutoCompactWindow && got <= maxAutoCompactWindow
		if !inRange {
			t.Fatalf("compactWindow(%d, %d) = %d, want a value in [%d, %d]",
				contextWindow, outputReserve, got, minAutoCompactWindow, maxAutoCompactWindow)
		}
	})
}
