a = 10
b = a
print(id(10))
print(id(a))
print(id(b))
# The results are same. 파이썬에서 변수를 할당하는 작업은 
# 해당 객체에 대한 참조만 한다는 것.

c = [1,2,3]
d = c
print(c,d,sep=' ')
c[2] = 10
print(c,d,sep= ',')
