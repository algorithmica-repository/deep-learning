#tuple is an immutable list 

tuple1 = (10, 30, 20, 40)
type(tuple1)
print tuple1

tuple1[0]
tuple1[0:2]
tuple1[0:]
tuple1[:3]
tuple1[::2]

len(tuple1)

for x in tuple1:
    print x
    
tuple2 = (10,20)
x,y = tuple2
print x
print y


