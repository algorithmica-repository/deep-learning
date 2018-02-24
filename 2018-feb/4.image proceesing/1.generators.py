def test_generator():
    i = 0
    while True:
        i = i + 1
        yield i
  
#using next over generator      
gen1 = test_generator()
print(next(gen1))
print(next(gen1))

#using generator directly
for item in test_generator():
    print(item)
    if item > 4:
        break