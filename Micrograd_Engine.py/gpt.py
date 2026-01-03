import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._children = _children
        self._op = _op

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = 1.0  # d(output)/d(output)

        for v in reversed(topo):
            v._backward()

# inputs
x1 = Value(2.0)
x2 = Value(-1.0)

# weights and bias
w1 = Value(-3.0)
w2 = Value(1.0)
b  = Value(0.5)

# neuron computation
z = x1*w1 + x2*w2 + b
out = z.__mul__

print("Neuron output:", out.data)
out.backward()

print("dw1:", w1.grad)
print("dw2:", w2.grad)
print("db :", b.grad)
print("dx1:", x1.grad)
print("dx2:", x2.grad)
