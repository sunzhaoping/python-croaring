# Introduction

python cffi binding for Roaring bitmaps in C, what is RoaringBitmap? 
please visit: https://github.com/RoaringBitmap/CRoaring
just use it as a Set of integer elements!

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
from croaring import BitSet

# create a bitmap like set
s = BitSet([1,2,3,4,5,6,7])

# count the elements
len(s)

# if a element in set
1 in s

# if another set in current set
other =  BitSet([1,2,3,4,5,6])
other in s

# iterate it
[ element for element in s]

# and or xor them
s & other
s | other
s ^ ohter

# add or sub them
s - other
s + other

# get the index of a element
s.index(7)

# get the element from index
s[0]

#serialize to a buffer
buf = s.dumps()

#deseialize from buffer
s.loads(buf)

# add element
s.add(8)

# remove element
s.remove(8)

# discard element
s.discard(8)

#union many sets
new = Set.union(s, other)
new = Set.union_heap(s, other)
```

