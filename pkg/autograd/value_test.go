package autograd

import (
	"math"
	"testing"
)

const eps = 1e-7
const tolerance = 1e-5

// almostEqual checks if two floats are approximately equal within tolerance.
func almostEqual(a, b, tol float64) bool {
	return math.Abs(a-b) < tol
}

// TestBasicAdd tests that Add computes correct values and gradients.
func TestBasicAdd(t *testing.T) {
	a := NewValue(2.0)
	b := NewValue(3.0)
	c := a.Add(b)

	if c.Data != 5.0 {
		t.Errorf("Add: expected 5.0, got %v", c.Data)
	}

	c.Backward()

	if a.Grad != 1.0 {
		t.Errorf("Add gradient for a: expected 1.0, got %v", a.Grad)
	}
	if b.Grad != 1.0 {
		t.Errorf("Add gradient for b: expected 1.0, got %v", b.Grad)
	}
}

// TestBasicMul tests that Mul computes correct values and gradients.
func TestBasicMul(t *testing.T) {
	a := NewValue(2.0)
	b := NewValue(3.0)
	c := a.Mul(b)

	if c.Data != 6.0 {
		t.Errorf("Mul: expected 6.0, got %v", c.Data)
	}

	c.Backward()

	// d(a*b)/da = b = 3
	if a.Grad != 3.0 {
		t.Errorf("Mul gradient for a: expected 3.0, got %v", a.Grad)
	}
	// d(a*b)/db = a = 2
	if b.Grad != 2.0 {
		t.Errorf("Mul gradient for b: expected 2.0, got %v", b.Grad)
	}
}

// TestBasicSub tests that Sub computes correct values and gradients.
func TestBasicSub(t *testing.T) {
	a := NewValue(5.0)
	b := NewValue(3.0)
	c := a.Sub(b)

	if c.Data != 2.0 {
		t.Errorf("Sub: expected 2.0, got %v", c.Data)
	}

	c.Backward()

	// d(a-b)/da = 1
	if a.Grad != 1.0 {
		t.Errorf("Sub gradient for a: expected 1.0, got %v", a.Grad)
	}
	// d(a-b)/db = -1
	if b.Grad != -1.0 {
		t.Errorf("Sub gradient for b: expected -1.0, got %v", b.Grad)
	}
}

// TestBasicDiv tests that Div computes correct values and gradients.
func TestBasicDiv(t *testing.T) {
	a := NewValue(6.0)
	b := NewValue(2.0)
	c := a.Div(b)

	if c.Data != 3.0 {
		t.Errorf("Div: expected 3.0, got %v", c.Data)
	}

	c.Backward()

	// d(a/b)/da = 1/b = 0.5
	if !almostEqual(a.Grad, 0.5, tolerance) {
		t.Errorf("Div gradient for a: expected 0.5, got %v", a.Grad)
	}
	// d(a/b)/db = -a/b^2 = -6/4 = -1.5
	if !almostEqual(b.Grad, -1.5, tolerance) {
		t.Errorf("Div gradient for b: expected -1.5, got %v", b.Grad)
	}
}

// TestPow tests that Pow computes correct values and gradients.
func TestPow(t *testing.T) {
	a := NewValue(2.0)
	b := a.Pow(3.0) // 2^3 = 8

	if b.Data != 8.0 {
		t.Errorf("Pow: expected 8.0, got %v", b.Data)
	}

	b.Backward()

	// d(x^3)/dx = 3*x^2 = 3*4 = 12
	if !almostEqual(a.Grad, 12.0, tolerance) {
		t.Errorf("Pow gradient: expected 12.0, got %v", a.Grad)
	}
}

// TestExp tests that Exp computes correct values and gradients.
func TestExp(t *testing.T) {
	a := NewValue(2.0)
	b := a.Exp()

	expected := math.Exp(2.0)
	if !almostEqual(b.Data, expected, tolerance) {
		t.Errorf("Exp: expected %v, got %v", expected, b.Data)
	}

	b.Backward()

	// d(e^x)/dx = e^x
	if !almostEqual(a.Grad, expected, tolerance) {
		t.Errorf("Exp gradient: expected %v, got %v", expected, a.Grad)
	}
}

// TestLog tests that Log computes correct values and gradients.
func TestLog(t *testing.T) {
	a := NewValue(2.0)
	b := a.Log()

	expected := math.Log(2.0)
	if !almostEqual(b.Data, expected, tolerance) {
		t.Errorf("Log: expected %v, got %v", expected, b.Data)
	}

	b.Backward()

	// d(ln(x))/dx = 1/x = 0.5
	if !almostEqual(a.Grad, 0.5, tolerance) {
		t.Errorf("Log gradient: expected 0.5, got %v", a.Grad)
	}
}

// TestReLUPositive tests ReLU with positive input.
func TestReLUPositive(t *testing.T) {
	a := NewValue(3.0)
	b := a.ReLU()

	if b.Data != 3.0 {
		t.Errorf("ReLU positive: expected 3.0, got %v", b.Data)
	}

	b.Backward()

	if a.Grad != 1.0 {
		t.Errorf("ReLU positive gradient: expected 1.0, got %v", a.Grad)
	}
}

// TestReLUNegative tests ReLU with negative input.
func TestReLUNegative(t *testing.T) {
	a := NewValue(-3.0)
	b := a.ReLU()

	if b.Data != 0.0 {
		t.Errorf("ReLU negative: expected 0.0, got %v", b.Data)
	}

	b.Backward()

	if a.Grad != 0.0 {
		t.Errorf("ReLU negative gradient: expected 0.0, got %v", a.Grad)
	}
}

// TestBackwardSimpleExpression tests Backward on (a*b + c).
func TestBackwardSimpleExpression(t *testing.T) {
	a := NewValue(2.0)
	b := NewValue(3.0)
	c := NewValue(4.0)

	// L = a*b + c = 2*3 + 4 = 10
	ab := a.Mul(b)
	L := ab.Add(c)

	if L.Data != 10.0 {
		t.Errorf("Expression: expected 10.0, got %v", L.Data)
	}

	L.Backward()

	// dL/da = b = 3
	if a.Grad != 3.0 {
		t.Errorf("Gradient for a: expected 3.0, got %v", a.Grad)
	}
	// dL/db = a = 2
	if b.Grad != 2.0 {
		t.Errorf("Gradient for b: expected 2.0, got %v", b.Grad)
	}
	// dL/dc = 1
	if c.Grad != 1.0 {
		t.Errorf("Gradient for c: expected 1.0, got %v", c.Grad)
	}
}

// TestBackwardComplexExpression tests a more complex expression.
func TestBackwardComplexExpression(t *testing.T) {
	// f = (a + b) * (b + 1)
	// where a=2, b=3
	// f = (2+3) * (3+1) = 5 * 4 = 20
	a := NewValue(2.0)
	b := NewValue(3.0)
	one := Scalar(1.0)

	apb := a.Add(b)   // a + b = 5
	bp1 := b.Add(one) // b + 1 = 4
	f := apb.Mul(bp1) // (a+b) * (b+1) = 20

	if f.Data != 20.0 {
		t.Errorf("Complex expression: expected 20.0, got %v", f.Data)
	}

	f.Backward()

	// df/da = (b + 1) = 4
	if !almostEqual(a.Grad, 4.0, tolerance) {
		t.Errorf("Gradient for a: expected 4.0, got %v", a.Grad)
	}
	// df/db = (b + 1) + (a + b) = 4 + 5 = 9
	// (b appears in both apb and bp1)
	if !almostEqual(b.Grad, 9.0, tolerance) {
		t.Errorf("Gradient for b: expected 9.0, got %v", b.Grad)
	}
}

// TestNeg tests negation.
func TestNeg(t *testing.T) {
	a := NewValue(5.0)
	b := a.Neg()

	if b.Data != -5.0 {
		t.Errorf("Neg: expected -5.0, got %v", b.Data)
	}

	b.Backward()

	// d(-a)/da = -1
	if a.Grad != -1.0 {
		t.Errorf("Neg gradient: expected -1.0, got %v", a.Grad)
	}
}

// TestZeroGrad tests that ZeroGrad resets gradient.
func TestZeroGrad(t *testing.T) {
	a := NewValue(2.0)
	b := NewValue(3.0)
	c := a.Mul(b)
	c.Backward()

	if a.Grad != 3.0 {
		t.Errorf("Before ZeroGrad: expected 3.0, got %v", a.Grad)
	}

	a.ZeroGrad()

	if a.Grad != 0.0 {
		t.Errorf("After ZeroGrad: expected 0.0, got %v", a.Grad)
	}
}

// numericalGradient computes the numerical gradient of f at x using central difference.
func numericalGradient(f func(float64) float64, x float64) float64 {
	return (f(x+eps) - f(x-eps)) / (2 * eps)
}

// TestNumericalGradientMul checks analytical vs numerical gradient for multiplication.
func TestNumericalGradientMul(t *testing.T) {
	aVal := 2.0
	bVal := 3.0

	// Analytical gradient via autograd
	a := NewValue(aVal)
	b := NewValue(bVal)
	c := a.Mul(b)
	c.Backward()

	// Numerical gradient for a
	fA := func(x float64) float64 {
		return x * bVal
	}
	numGradA := numericalGradient(fA, aVal)

	if !almostEqual(a.Grad, numGradA, tolerance) {
		t.Errorf("Numerical gradient check for a: analytical=%v, numerical=%v", a.Grad, numGradA)
	}

	// Numerical gradient for b
	fB := func(x float64) float64 {
		return aVal * x
	}
	numGradB := numericalGradient(fB, bVal)

	if !almostEqual(b.Grad, numGradB, tolerance) {
		t.Errorf("Numerical gradient check for b: analytical=%v, numerical=%v", b.Grad, numGradB)
	}
}

// TestNumericalGradientComplex checks analytical vs numerical gradient for complex expression.
func TestNumericalGradientComplex(t *testing.T) {
	// f(a, b, c) = (a * b + c)^2
	aVal, bVal, cVal := 2.0, 3.0, 4.0

	// Analytical gradient via autograd
	a := NewValue(aVal)
	b := NewValue(bVal)
	c := NewValue(cVal)
	ab := a.Mul(b)
	abc := ab.Add(c)
	f := abc.Pow(2)
	f.Backward()

	// Numerical gradient for a
	fA := func(x float64) float64 {
		return math.Pow(x*bVal+cVal, 2)
	}
	numGradA := numericalGradient(fA, aVal)

	if !almostEqual(a.Grad, numGradA, tolerance) {
		t.Errorf("Numerical gradient for a: analytical=%v, numerical=%v", a.Grad, numGradA)
	}

	// Numerical gradient for b
	fB := func(x float64) float64 {
		return math.Pow(aVal*x+cVal, 2)
	}
	numGradB := numericalGradient(fB, bVal)

	if !almostEqual(b.Grad, numGradB, tolerance) {
		t.Errorf("Numerical gradient for b: analytical=%v, numerical=%v", b.Grad, numGradB)
	}

	// Numerical gradient for c
	fC := func(x float64) float64 {
		return math.Pow(aVal*bVal+x, 2)
	}
	numGradC := numericalGradient(fC, cVal)

	if !almostEqual(c.Grad, numGradC, tolerance) {
		t.Errorf("Numerical gradient for c: analytical=%v, numerical=%v", c.Grad, numGradC)
	}
}

// TestNumericalGradientExp checks analytical vs numerical gradient for Exp.
func TestNumericalGradientExp(t *testing.T) {
	xVal := 1.5

	x := NewValue(xVal)
	y := x.Exp()
	y.Backward()

	fX := func(v float64) float64 {
		return math.Exp(v)
	}
	numGrad := numericalGradient(fX, xVal)

	if !almostEqual(x.Grad, numGrad, tolerance) {
		t.Errorf("Numerical gradient for Exp: analytical=%v, numerical=%v", x.Grad, numGrad)
	}
}

// TestNumericalGradientLog checks analytical vs numerical gradient for Log.
func TestNumericalGradientLog(t *testing.T) {
	xVal := 2.5

	x := NewValue(xVal)
	y := x.Log()
	y.Backward()

	fX := func(v float64) float64 {
		return math.Log(v)
	}
	numGrad := numericalGradient(fX, xVal)

	if !almostEqual(x.Grad, numGrad, tolerance) {
		t.Errorf("Numerical gradient for Log: analytical=%v, numerical=%v", x.Grad, numGrad)
	}
}

// TestNumericalGradientReLU checks analytical vs numerical gradient for ReLU.
func TestNumericalGradientReLU(t *testing.T) {
	// Test positive
	xVal := 2.0

	x := NewValue(xVal)
	y := x.ReLU()
	y.Backward()

	fX := func(v float64) float64 {
		if v > 0 {
			return v
		}
		return 0
	}
	numGrad := numericalGradient(fX, xVal)

	if !almostEqual(x.Grad, numGrad, tolerance) {
		t.Errorf("Numerical gradient for ReLU (positive): analytical=%v, numerical=%v", x.Grad, numGrad)
	}

	// Test negative
	xVal = -2.0
	x = NewValue(xVal)
	y = x.ReLU()
	y.Backward()

	numGrad = numericalGradient(fX, xVal)

	if !almostEqual(x.Grad, numGrad, tolerance) {
		t.Errorf("Numerical gradient for ReLU (negative): analytical=%v, numerical=%v", x.Grad, numGrad)
	}
}

// TestReusedVariable tests that a variable used multiple times accumulates gradients correctly.
func TestReusedVariable(t *testing.T) {
	// f = a * a = a^2
	a := NewValue(3.0)
	f := a.Mul(a)

	if f.Data != 9.0 {
		t.Errorf("a*a: expected 9.0, got %v", f.Data)
	}

	f.Backward()

	// df/da = 2a = 6
	if !almostEqual(a.Grad, 6.0, tolerance) {
		t.Errorf("Gradient for a*a: expected 6.0, got %v", a.Grad)
	}
}

// TestNeuronSimulation simulates a simple neuron: y = relu(w*x + b)
func TestNeuronSimulation(t *testing.T) {
	// y = relu(w*x + b)
	// w=0.5, x=2, b=-0.5
	// wx + b = 1 - 0.5 = 0.5
	// relu(0.5) = 0.5
	w := NewValue(0.5)
	x := NewValue(2.0)
	b := NewValue(-0.5)

	wx := w.Mul(x)
	wxb := wx.Add(b)
	y := wxb.ReLU()

	if !almostEqual(y.Data, 0.5, tolerance) {
		t.Errorf("Neuron output: expected 0.5, got %v", y.Data)
	}

	y.Backward()

	// dy/dw = x * (1 if wx+b > 0 else 0) = 2 * 1 = 2
	if !almostEqual(w.Grad, 2.0, tolerance) {
		t.Errorf("Gradient for w: expected 2.0, got %v", w.Grad)
	}
	// dy/dx = w * 1 = 0.5
	if !almostEqual(x.Grad, 0.5, tolerance) {
		t.Errorf("Gradient for x: expected 0.5, got %v", x.Grad)
	}
	// dy/db = 1
	if !almostEqual(b.Grad, 1.0, tolerance) {
		t.Errorf("Gradient for b: expected 1.0, got %v", b.Grad)
	}
}
