def create_empty_matrix(n, m):
    result = []
    for i in range(n):
        result.append([0] * m)
    return result


class Matrix:
    def __init__(self, X):
        self.X = X

    def __repr__(self):
        return "<Matrix ({}, {}) Ts={}>".format(self.get_rows(), self.get_columns(), self.totalsum())

    def get_columns(self):
        return len(self.X[0])

    def get_rows(self):
        return len(self.X)

    def __getitem__(self, key):
        return Vector(self.X[key])

    def __add__(self, matrix):
        if self.get_rows() != matrix.get_rows() or self.get_columns() != matrix.get_columns():
            raise ValueError('The dimensions is not correct')
        empty = []
        for i in range(self.get_rows()):
            arr = [self.X[i][h] + matrix.X[i][h] for h in range(self.get_columns())]
            empty.append(arr)
        return Matrix(empty)

    def __sub__(self, other):
        return other * -1 + self



    def totalsum(self):
        tsum = 0
        for i in range(len(self.X)):
            for h in range(len(self.X[0])):
                tsum += self.X[i][h]
        return tsum

    def multiplied_by_num(self, num):
        multyplied = []
        for i in range(self.get_rows()):
            arr2 = [self.X[i][j] * num for j in range(self.get_columns())]
            multyplied.append(arr2)
        return Matrix(multyplied)

    def transpone(self):
        empty = []
        for i in range(self.get_columns()):
            arr = [self.X[j][i] for j in range(self.get_rows())]
            empty.append(arr)
        return Matrix(empty)

    def can_we_multiply(self, matrix):
        if self.get_columns() == matrix.get_rows():
            print('Yes')
        else:
            print('No')

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, Matrix):
            return self.multiplied_by_matrix(other)
        if isinstance(other, (int, float)):
            return self.multiplied_by_num(other)
        else:
            raise NotImplementedError

    def multiplied_by_matrix(self, matrix):
        if self.get_columns() != matrix.get_rows():
            raise ValueError(
                'Matrix with the dimensions ({} {}) and ({},{}) can not be multiplied'.format(self.get_rows(),
                                                                                              self.get_columns(),
                                                                                              matrix.get_rows(),
                                                                                              matrix.get_columns()))
        rows = self.get_rows()
        columns = matrix.get_columns()
        result = create_empty_matrix(rows, columns)
        for i in range(rows):
            for j in range(columns):
                row_x_column_result = 0
                for k in range(len(matrix.X)):
                    row_x_column_result = row_x_column_result + self.X[i][k] * matrix.X[k][j]
                    result[i][j] = row_x_column_result
        return Matrix(result)


class Vector:
    def __init__(self, V):
        self.V = V

    def size(self):
        return len(self.V)

    def __repr__(self):
        return "<Vector {} Dp={}>".format(self.size(), self.dot_product(self))

    def __getitem__(self, key):
        return self.V[key]

    def __add__(self, other):
        if isinstance(other, Vector):
            return self.vector_addition(other)
        raise NotImplementedError

    def vector_addition(self, vector):
        if self.size() != vector.size():
            raise ValueError("Can not sum")
        return Vector([self[i] + vector[i] for i in range(self.size())])

    def __rmul__(self, other):
        if isinstance(other, (int, float, Vector)):
            return self.__mul__(other)
        raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, Vector):
            return self.dot_product(other)
        if isinstance(other, (int, float)):
            return self.multiplied_by_num(other)
        if isinstance(other, Matrix):
            return self.vector_by_matrix(other)
        raise NotImplementedError

    def __sub__(self, other):
        return other * -1 + self

    def multiplied_by_num(self, num):
        # return Vector([self.V[i] * num for i in range(len(self.V))])
        return Vector([x * num for x in self.V])

    def dot_product(self, vector):
        arr = [self[i] * vector[i] for i in range(self.size())]
        sum = 0
        for j in arr:
            sum += j
        return sum

    def vector_by_matrix(self, matrix):
        if self.size() != matrix.get_rows():
            raise ValueError(
                'Matrix with the size {} cant be muliplied by vector {}'.format(matrix.get_rows(), self.size()))
        vector = []
        for i in range(matrix.get_columns()):
            vector.append(sum([matrix[j][i] * self[j] for j in range(self.size())]))
        return Vector(vector)

    def transpone(self):
        return Matrix([self.V]).transpone()


class LinearRegression:
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        y = y.transpone()
        w = Vector([1] * X.get_columns()).transpone()
        for i in range(1000):
            w = X.transpone().multiplied(y.multiplied_by_num(-1).addition(X.multiplied(w))).multiplied_by_num(
                -0.001).addition(w)
        self.w = w

    def predict(self, v):
        pred = v.vector_by_matrix(self.w)
        return pred.V[0]


# X1 = Matrix([[1, 2, 3],
#              [5, 7, 8]])
# y1 = Vector([2, 3])
# lr = LinearRegression()
# lr.fit(X1, y1)
# v1 = Vector([5, 7, 8])
# print(lr.predict(v1))

#
# m1 = Matrix([[1, 2, 3], [3, 4, 5]])
# m2 = Matrix([[1, 2], [1, 3]])
# m1.multiplied(m2)

v1 = Vector([3, 5, 4])

# print(v1.multiplied_by_num(3).V)
# print(v1.vector_addition(v1).V)
m1 = Matrix([[1, 2], [5, 3], [3, 4]])
m2 = Matrix([[1, 3], [7, 6], [3, 6]])

# print(m1.addition(m2).X)
# print(m1.transpone().X)
# print(m1.multiplied_by_num(2).X)
# print(m2.multiplied(m1).X)

print(m1 + m2)
print(m1 * m2.transpone())
print(m1)
print(m1 * 5)
print(5 * m1)
print(m1 - m2)
print(v1 + v1)
print('v1 * 2', v1 * 2)
print('2 * v1', 2 * v1)
print('v1 * v1', v1 * v1)
print('v1*m1', v1 * m1)
print('v1-v1', v1 - v1)
print('v1[0]', v1[0])
print('m1[0]', m1[0])
print('m1[0][0]', m1[0][0])
print('m1[0].dot_product', m1[0].dot_product(m1[0]))


def cross_validation_split(X, y, folds=3):
    X_split = []
    y_split = []
    X_copy = list(X)
    y_copy = list(y)
    fold_size = int(len(X) / folds)
    for i in range(folds):
        fold = []
        y_fold = []
        while len(fold) < fold_size:
            index = randrange(len(X_copy))
            fold.append(X_copy.pop(index))
            y_fold.append(y_copy.pop(index))
        X_split.append(fold)
        y_split.append(y_fold)
    return X_split, y_split


seed(1)

cross_validation_split(X, y)
