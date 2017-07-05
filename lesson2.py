
import numpy
import pandas

myar = numpy.array([[1, 2, 3], [4, 5, 6]])
rownames = ['a', 'b']
colnames = ['one', 'two', 'three']

df = pandas.DataFrame(myar, index=rownames, columns=colnames)

print(df)
