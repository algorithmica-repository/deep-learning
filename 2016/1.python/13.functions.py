def abs(x):
    if x < 0: 
        return -x
    return x

abs(-5)
abs(5)


def add(x,y=1):
    return x + y

add(5)
add(5,5)


def test(x,y=1,z=0):
    return (x+z)*y

test(10, z=10)

#lambda  is an anonymous function
# (x,y) : return x+y
add1 = lambda x,y:x+y
add1(10,20)

#find squares of bunch of numbers
def square(x):
    return x*x
for i in range(1,10):
    print square(i)
    
map(square, range(1,10))

map(lambda x:x*x, range(1,10))


#comprehensions

#generate numbers and their squares map
squares1 = { }
for i in range(1,10):
    squares1[i] = i * i
    
squares2 = {x: x **2 for x in range(1,10)}

import functools
a = [10,20,30]
map(add, a)
map(add(y=2), a)
map(functools.partial(add,2), a)



