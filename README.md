# Introduction

python cffi binding for Roaring bitmaps in C, what is RoaringBitmap? 
please visit: https://github.com/RoaringBitmap/CRoaring

# Install:
```
pip install git+https://github.com/sunzhaoping/python-croaring.git
```
or
```
pypy setup.py build && pypy setup.py install
```

# Demo:
```python
# import the croaring module
from croaring import Set

# create a bitmap like set
s = Set([1,2,3,4,5,6,7])

# count the elements
len(s)

# if a element in set
1 in s

# if another set in current set
other =  Set([1,2,3,4,5,6])
other in s

# iterate it
[ element for element in s]

```

