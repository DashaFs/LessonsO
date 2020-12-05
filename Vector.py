class Vector:
    def __init__(self, V):
        self.V = V

    def vector_addition(self, vector):
        new = []
        for i in range(len(vector)):
            new.append(vector.V[i] + self.V[i])
        return Vector(new)


v1 = [2, 3]
v2 = [7, 8]
v1.vector_addition(v2)
