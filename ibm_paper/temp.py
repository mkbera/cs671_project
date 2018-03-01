from preprocess import *

a = get_features()

q1 = a[:, 0]
q2 = a[:, 1]
label = a[:, 2]

print(q1)
print(q2)
print(label)
