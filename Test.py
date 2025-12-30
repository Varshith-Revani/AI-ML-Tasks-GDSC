from engine import Value

x = Value(2.0)
y = Value(3.0)
z = x * y + x + 2
z.backward()

print(x)  # grad should be y + 1 = 4
print(y)  # grad should be x = 2
