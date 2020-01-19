#generator to produce batch values of size 1
def test_generator1():
    i = 0
    while True:
        i = i + 1
        yield i
  
#using next over generator      
gen1 = test_generator1()
print(next(gen1))
print(next(gen1))

#using generator directly
for item in test_generator1():
    print(item)
    if item >10:
        break

#generator to produce batch of size 3
def test_generator2():
    res = []
    i = 0
    while True:
        res.clear()
        for j in range(3):
            i = i + 1
            res.append(i)
        yield res
  
#using next over generator      
gen2 = test_generator2()
print(next(gen2))
print(next(gen2))

#using generator directly
for item in test_generator2():
    print(item)