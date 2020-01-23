import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()

graph1 = tf1.Graph()

with graph1.as_default():
    a = tf1.constant([2],name = 'constant_a')
    b = tf1.constant([3], name='constant_b')

sesh = tf1.Session(graph=graph1)
print(sesh.run(a))
sesh.close()

with graph1.as_default():
    a = tf1.constant([2], name='constant_a')
    b = tf1.constant([3], name='constant_b')
    c = a + b

sesh = tf1.Session(graph=graph1)
print(sesh.run(c))
sesh.close()

#to avoid closing sessions all the time, use with statement
with tf1.Session(graph=graph1) as sesh:
    print(sesh.run(a))
    print(sesh.run(c))

graph2 = tf1.Graph()
with graph2.as_default():
    Matrix_one = tf1.constant([[2, 3], [3, 4]])
    Matrix_two = tf1.constant([[2, 3], [3, 4]])

    mul_operation = tf1.matmul(Matrix_one, Matrix_two)

with tf1.Session(graph=graph2) as sesh:
    print(sesh.run(mul_operation))

graph3 = tf1.Graph()
with graph3.as_default():
    v = tf1.Variable(0)
    update = tf1.assign(v,v+1)
    init_var = tf1.global_variables_initializer()

with tf1.Session(graph=graph3) as sesh:
    sesh.run(init_var)
    print(sesh.run(v))
    for _ in range(3):
        sesh.run(update)
        print(sesh.run(v))

graphPlaceHolder = tf1.Graph()
with graphPlaceHolder.as_default():
    a = tf1.placeholder(tf1.float32)
    b = a * 2

dictionary={a: [ [ [1,2,3],[4,5,6],[7,8,9],[10,11,12] ] , [ [13,14,15],[16,17,18],[19,20,21],[22,23,24] ] ] }
with tf1.Session(graph=graphPlaceHolder) as sesh:
    print(sesh.run(b,feed_dict={a:3.5}))
    print(sesh.run(b, feed_dict=dictionary))

#operations examples
#tf1.constant, tf1.matmul, tf1.add, tf1.nn.sigmoid, tf1.nn.relu
tf1.enable_v2_behavior()
import tensorflow as tf

a = tf1.constant([2], name='constant_a')
b = tf1.constant([3], name='constant_b')
c = tf.convert_to_tensor([ [ [1,2,3],[4,5,6],[7,8,9],[10,11,12] ] , [ [13,14,15],[16,17,18],[19,20,21],[22,23,24] ] ],dtype=tf.float32)
Matrix_one = tf.constant([[2, 3], [3, 4]])
Matrix_two = tf.constant([[2, 3], [3, 4]])
v = tf.Variable(0)

@tf.function
def f1():
    return a

@tf.function
def f2():
    return a + b

@tf.function
def f3(a,b):
    return tf.matmul(a,b)

@tf.function
def f4(x):
    x.assign_add(1)
    return x

@tf.function
def f5(x):
    return 2.0*x

tf.print(f1())
tf.print(f2())
tf.print(f3(Matrix_one,Matrix_two))
tf.print(v)
for i in range(3):
    tf.print(f4(v))
tf.print(f5(3.5))
tf.print(f5(c))

@tf.function
def f6(x):
    return tf.square(x)

@tf.function
def add(a,b):
    return tf.square(a) + (2*a)

v = tf.Variable([1.0,2.0])
with tf.GradientTape() as tape:
    result = add(v,1.0)
print(tape.gradient(result,v))