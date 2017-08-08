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

# create a set like bitmap
s = Set([1,2,3,4,5,6,7])

# count the elements
len(s)

#
```

