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
```
from croaring import Set
s = Set([1,2,3,4,5,6,7])
len(s)
```

