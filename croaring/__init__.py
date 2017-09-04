# -*- coding: utf8 -*-
from __future__ import absolute_import, division, print_function, with_statement
import sys
import os
import binascii
import threading
import collections
import array
from cffi import FFI
from cffi.verifier import Verifier
PY3 = sys.version_info >= (3,)
if PY3:
    xrange = range

include_dir = os.path.split(os.path.realpath(__file__))[0]
ffi = FFI()

def _create_modulename(cdef_sources, source, sys_version):
    """
        This is the same as CFFI's create modulename except we don't include the
        CFFI version.
        """
    key = '\x00'.join([sys_version[:3], source, cdef_sources])
    key = key.encode('utf-8')
    k1 = hex(binascii.crc32(key[0::2]) & 0xffffffff)
    k1 = k1.lstrip('0x').rstrip('L')
    k2 = hex(binascii.crc32(key[1::2]) & 0xffffffff)
    k2 = k2.lstrip('0').rstrip('L')
    return '_Croaring_cffi_{0}{1}'.format(k1, k2)

def _compile_module(*args, **kwargs):
    raise RuntimeError(
                       "Attempted implicit compile of a cffi module. All cffi modules should be pre-compiled at installation time."
                       )

class LazyLibrary(object):
    def __init__(self, ffi):
        self._ffi = ffi
        self._lib = None
        self._lock = threading.Lock()

    def __getattr__(self, name):
        if self._lib is None:
            with self._lock:
                if self._lib is None:
                    self._lib = self._ffi.verifier.load_library()

        return getattr(self._lib, name)

CDEF="""
typedef struct roaring_array_s {
    int32_t size;
    int32_t allocation_size;
    void **containers;
    uint16_t *keys;
    uint8_t *typecodes;
} roaring_array_t;

typedef struct roaring_bitmap_s {
    roaring_array_t high_low_container;
    bool copy_on_write;
} roaring_bitmap_t;

typedef bool (*roaring_iterator)(uint32_t value, void *param);
typedef bool (*roaring_iterator64)(uint64_t value, void *param);
typedef struct roaring_uint32_iterator_s roaring_uint32_iterator_t;
roaring_bitmap_t *roaring_bitmap_create();
roaring_bitmap_t *roaring_bitmap_from_range(uint32_t min, uint32_t max,uint32_t step);
roaring_bitmap_t *roaring_bitmap_create_with_capacity(uint32_t cap);
roaring_bitmap_t *roaring_bitmap_of_ptr(size_t n_args, const uint32_t *vals);
void roaring_bitmap_printf_describe(const roaring_bitmap_t *ra);
roaring_bitmap_t *roaring_bitmap_of(size_t n, ...);
roaring_bitmap_t *roaring_bitmap_copy(const roaring_bitmap_t *r);
void roaring_bitmap_printf(const roaring_bitmap_t *ra);
void roaring_bitmap_free(roaring_bitmap_t *r);
void roaring_bitmap_add(roaring_bitmap_t *r, uint32_t val);
uint64_t roaring_bitmap_get_cardinality(const roaring_bitmap_t *ra);
bool roaring_bitmap_contains(const roaring_bitmap_t *r, uint32_t val);
void roaring_bitmap_remove(roaring_bitmap_t *r, uint32_t val);
roaring_bitmap_t *roaring_bitmap_or(const roaring_bitmap_t *x1, const roaring_bitmap_t *x2);
void roaring_bitmap_or_inplace(roaring_bitmap_t *x1, const roaring_bitmap_t *x2);
roaring_bitmap_t *roaring_bitmap_and(const roaring_bitmap_t *x1, const roaring_bitmap_t *x2);
void roaring_bitmap_add_many(roaring_bitmap_t *r, size_t n_args, const uint32_t *vals);
void roaring_bitmap_and_inplace(roaring_bitmap_t *x1, const roaring_bitmap_t *x2);
roaring_bitmap_t *roaring_bitmap_xor(const roaring_bitmap_t *x1, const roaring_bitmap_t *x2);
void roaring_bitmap_xor_inplace(roaring_bitmap_t *x1, const roaring_bitmap_t *x2);
roaring_bitmap_t *roaring_bitmap_andnot(const roaring_bitmap_t *x1, const roaring_bitmap_t *x2);
void roaring_bitmap_andnot_inplace(roaring_bitmap_t *x1, const roaring_bitmap_t *x2);
roaring_bitmap_t *roaring_bitmap_or_many_heap(uint32_t number,const roaring_bitmap_t **x);
roaring_bitmap_t *roaring_bitmap_or_many(uint32_t number,const roaring_bitmap_t **x);
roaring_bitmap_t *roaring_bitmap_flip(const roaring_bitmap_t *x1, uint64_t range_start, uint64_t range_end);
void roaring_bitmap_flip_inplace(roaring_bitmap_t *x1, uint64_t range_start,uint64_t range_end);
bool roaring_bitmap_run_optimize(roaring_bitmap_t *r);
size_t roaring_bitmap_shrink_to_fit(roaring_bitmap_t *r);
roaring_bitmap_t *roaring_bitmap_deserialize(const void *buf);
bool roaring_bitmap_is_empty(const roaring_bitmap_t *ra);
size_t roaring_bitmap_serialize(const roaring_bitmap_t *ra, char *buf);
size_t roaring_bitmap_size_in_bytes(const roaring_bitmap_t *ra);
bool roaring_bitmap_equals(const roaring_bitmap_t *ra1, const roaring_bitmap_t *ra2);
uint64_t roaring_bitmap_rank(const roaring_bitmap_t *bm, uint32_t x);
uint32_t roaring_bitmap_minimum(const roaring_bitmap_t *bm);
uint32_t roaring_bitmap_maximum(const roaring_bitmap_t *bm);
bool roaring_bitmap_select(const roaring_bitmap_t *bm, uint32_t rank, uint32_t *element);
bool roaring_bitmap_intersect(const roaring_bitmap_t *x1, const roaring_bitmap_t *x2);
bool roaring_bitmap_is_subset(const roaring_bitmap_t *ra1,const roaring_bitmap_t *ra2);
bool roaring_bitmap_is_strict_subset(const roaring_bitmap_t *ra1,const roaring_bitmap_t *ra2);
roaring_uint32_iterator_t *roaring_create_iterator(const roaring_bitmap_t *ra);
bool roaring_advance_uint32_iterator(roaring_uint32_iterator_t *it);
void roaring_free_uint32_iterator(roaring_uint32_iterator_t *it);
bool iterator_has_value(const roaring_uint32_iterator_t *it);
uint32_t iterator_current_value(const roaring_uint32_iterator_t *it);
void roaring_bitmap_clear(roaring_bitmap_t *ra);
void roaring_bitmap_to_uint32_array(const roaring_bitmap_t *ra, uint32_t *ans);
roaring_bitmap_t *roaring_bitmap_and_many(size_t number, const roaring_bitmap_t **x);
"""

SOURCE = """
#include <roaring.c>
roaring_bitmap_t *roaring_bitmap_and_many(size_t number, const roaring_bitmap_t **x) {
    if (number == 0) {
        return roaring_bitmap_create();
    }
    if (number == 1) {
        return roaring_bitmap_copy(x[0]);
    }
    roaring_bitmap_t *answer = roaring_bitmap_copy(x[0]);
    for (size_t i = 2; i < number; i++) {
        roaring_bitmap_and_inplace(answer, x[i]);
    }
    return answer;
}
bool iterator_has_value(const roaring_uint32_iterator_t *it){
    return it->has_value;
}
uint32_t iterator_current_value(const roaring_uint32_iterator_t *it){
    return it->current_value;
}
"""

ffi.cdef(CDEF)
ffi.verifier = Verifier(ffi,
                        SOURCE ,
                        include_dirs=[include_dir],
                        modulename=_create_modulename(CDEF, SOURCE, sys.version),
                        extra_compile_args=['-march=native','-std=c99','-O3'])

ffi.verifier.compile_module = _compile_module
ffi.verifier._compile_module = _compile_module

lib = LazyLibrary(ffi)

class BitSet(collections.Set):
    def __init__(self, values=None, copy_on_write=False, croaring = None):
        if croaring:
            assert values is None and not copy_on_write
            self._croaring = croaring
            return
        elif values is None:
            self._croaring = lib.roaring_bitmap_create()
        elif isinstance(values, BitSet):
            self._croaring = lib.roaring_bitmap_copy(values._croaring)
        elif PY3 and isinstance(values, range):
            _, (start, stop, step) = values.__reduce__()
            if step < 0:
                values = range(min(values), max(values)+1, -step)
                _, (start, stop, step) = values.__reduce__()
            if start >= stop:
                self._croaring = lib.roaring_bitmap_create()
            else:
                self._croaring = lib.roaring_bitmap_from_range(start, stop, step)
        elif isinstance(values, array.array):
            buffer = ffi.cast("uint32_t*", ffi.from_buffer(values))
            self._croaring = lib.roaring_bitmap_of_ptr(len(values), buffer)
        else:
            self._croaring = lib.roaring_bitmap_create()
            self.update(values)
        if not isinstance(values, BitSet):
            self._croaring.copy_on_write = copy_on_write

    def update(self, *all_values):
        for values in all_values:
            if isinstance(values, BitSet):
                self |= values
            elif PY3 and isinstance(values, range):
                self |= BitSet(values, copy_on_write=self.copy_on_write)
            elif isinstance(values, array.array):
                buffer = ffi.cast("uint32_t*", ffi.from_buffer(values))
                lib.roaring_bitmap_add_many(self._croaring, len(values), buffer)
            else:
                lib.roaring_bitmap_add_many(self._croaring, len(values), values)

    def intersection_update(self, *all_values):
        for values in all_values:
            if isinstance(values, BitSet):
                self &= values
            else:
                self &= BitMap(values, copy_on_write=self.copy_on_write)
    @property
    def copy_on_write(self):
        return self._croaring.copy_on_write

    def __repr__(self):
        return str(self)
    
    def __str__(self):
        values = ', '.join([str(n) for n in self])
        return 'BitSet([%s])' % values

    def __bool__(self):
        return not bool(lib.roaring_bitmap_is_empty(self._croaring))

    def __contains__(self, value):
        if isinstance(value, BitSet):
            return bool(lib.roaring_bitmap_is_subset(value._croaring, self._croaring))
        return bool(lib.roaring_bitmap_contains(self._croaring, ffi.cast("uint32_t", value)))

    def __iter__(self):
        iter = lib.roaring_create_iterator(self._croaring)
        while lib.iterator_has_value(iter):
           yield lib.iterator_current_value(iter)
           lib.roaring_advance_uint32_iterator(iter)
        lib.roaring_free_uint32_iterator(iter)

    def __and__(self, other):
        _croaring =  lib.roaring_bitmap_and(self._croaring,  other._croaring)
        return BitSet(croaring = _croaring)

    def __iand__(self, other):
        lib.roaring_bitmap_and_inplace(self._croaring,  other._croaring)
        return self

    def __or__(self, other):
        _croaring =  lib.roaring_bitmap_or(self._croaring,  other._croaring)
        return BitSet(croaring = _croaring)

    def __ior__(self, other):
        lib.roaring_bitmap_or_inplace(self._croaring,  other._croaring)
        return self

    def __xor__(self, other):
        _croaring =  lib.roaring_bitmap_xor(self._croaring,  other._croaring)
        return BitSet(croaring = _croaring)

    def __ixor__(self, other):
        lib.roaring_bitmap_xor_inplace(self._croaring,  other._croaring)
        return self

    def __sub__(self, other):
        _croaring =  lib.roaring_bitmap_andnot(self._croaring,  other._croaring)
        return BitSet(croaring = _croaring)

    def __isub__(self, other):
        lib.roaring_bitmap_andnot_inplace(self._croaring,  other._croaring)
        return self

    def __add__(self, other):
        return self.__or__(other)

    def __iadd__(self, other):
        return self.__ior__(other)

    def __del__(self):
        if hasattr(self, '_croaring') and self._croaring is not None:
            lib.roaring_bitmap_free(self._croaring)
            del self._croaring

    def _get_elt(self, index):
        if not self:
            raise IndexError("BitSet index out of range")
        elif index == 0:
            return self.minimum()
        elif index == -1:
            return self.maximum()
        elif index < 0:
            if abs(index) <= len(self):
                return self[len(self) + index]
            raise IndexError("BitSet index out of range")

        elem = ffi.new('uint32_t *')
        found = lib.roaring_bitmap_select(self._croaring, index, elem)
        out = ffi.new('uint32_t *')
        if lib.roaring_bitmap_select(self._croaring, ffi.cast("uint32_t", index), out):
            return out[0]
        else:
            raise IndexError("BitSet index out of range")
    
    def _get_slice(self, sl):
        start, stop, step = sl.indices(len(self))
        sign = 1 if step > 0 else -1
        if (sign > 0 and start >= stop) or (sign < 0 and start <= stop):
            return self.__class__()
        if abs(step) == 1:
            first_elt = self._get_elt(start)
            last_elt  = self._get_elt(stop-sign)
            values = range(first_elt, last_elt+sign, step)
            result = self.__class__(values, copy_on_write=self.copy_on_write)
            result &= self
            return result
        else:
            return self.__class__(list(self)[sl])    

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._get_elt(index)
        elif isinstance(index, slice):
            return self._get_slice(index)
        else:
            return TypeError('Indices must be integers or slices, not %s' % type(index)) 

    def __richcmp__(self, other, op):
        if op == 0: # <
            return bool(lib.roaring_bitmap_is_strict_subset(self._croaring, other._croaring))
        elif op == 1: # <=
            return bool(lib.roaring_bitmap_is_subset(self._croaring, other._croaring))
        elif op == 2: # ==
            return bool(lib.roaring_bitmap_equals(self._croaring, other._croaring))
        elif op == 3: # !=
            return not (self == other)
        elif op == 4: # >
            return bool(lib.roaring_bitmap_is_strict_subset(self._croaring, other._croaring))
        else:         # >=
            assert op == 5
            return bool(lib.roaring_bitmap_is_subset(self._croaring, other._croaring))
 
    def __len__(self):
        return lib.roaring_bitmap_get_cardinality(self._croaring)

    def __eq__(self, other):
        return bool(lib.roaring_bitmap_equals(self._croaring, other._croaring))
    
    def __lt__(self, other):
        return bool(lib.roaring_bitmap_is_strict_subset(self._croaring, other._croaring))

    def __gt__(self, other):
        return other < self

    def __le__(self, other):
        return bool(lib.roaring_bitmap_is_subset(self._croaring, other._croaring))

    def __ge__(self, other):
        return other <= self

    def __copy__(self):
        return self.copy()

    def copy(self):
        _croaring =  lib.roaring_bitmap_copy(self._croaring)
        return BitSet(croaring = _croaring)

    def flip(self, start, end):
        _croaring = lib.roaring_bitmap_flip(self._croaring, ffi.cast("uint64_t",start), ffi.cast("uint64_t", end))
        return BitSet(croaring = _croaring)

    def flip_inplace(self, start, end):
        lib.roaring_bitmap_flip(self._croaring, ffi.cast("uint64_t",start), ffi.cast("uint64_t", end))
        return self

    def __getstate__(self):
        return self.dumps()

    def __setstate__(self, value):
        inbuf = ffi.new('char[%d]'%(len(value)), value)
        self._croaring = lib.roaring_bitmap_deserialize(inbuf)

    @classmethod
    def union(cls, *bitsets):
        if size <= 1:
            return cls(*bitsets)
        elif size == 2:
            return bitsets[0] | bitsets[1]
        else:
            _croaring = lib.roaring_bitmap_or_many(len(bitsets), [b._croaring for b in bitsets])
            return cls(croaring = _croaring)

    @classmethod
    def union_heap(cls, *bitsets):
        if size <= 1:
            return cls(*bitsets)
        elif size == 2:
            return bitsets[0] | bitsets[1]
        else:
            _croaring = lib.roaring_bitmap_or_many_heap(len(bitsets), [b._croaring for b in bitsets])
            return cls(croaring = _croaring)

    @classmethod
    def intersection(cls, *bitsets):
        size = len(bitsets)
        if size <= 1:
            return cls(*bitsets)
        elif size == 2:
            return bitsets[0] | bitsets[1]
        else:
            _croaring = lib.roaring_bitmap_and_many(len(bitsets), [b._croaring for b in bitsets])
            return cls(croaring = _croaring)
    
    def union_cardinality(self, other):
        return lib.roaring_bitmap_or_cardinality(self._croaring, other._croaring)

    def intersection_cardinality(self, other):
        return lib.roaring_bitmap_and_cardinality(self._croaring, other._croaring)

    def difference_cardinality(self, other):
        return lib.roaring_bitmap_andnot_cardinality(self._croaring, other._croaring)

    def symmetric_difference_cardinality(self, other):
        return lib.roaring_bitmap_xor_cardinality(self._croaring, other._croaring)

    def add(self, value):
        lib.roaring_bitmap_add(self._croaring, ffi.cast("uint32_t", value))

    def discard(self, value):
        lib.roaring_bitmap_remove(self._croaring, ffi.cast("uint32_t", value))

    def pop(self, maxvalue = False):
        result = self.maximum() if maxvalue else self.minimum()
        self.remove(result)
        return result

    def remove(self, value):
        if value in self:
            lib.roaring_bitmap_remove(self._croaring, ffi.cast("uint32_t", value))
        else:
            raise KeyError(value)

    def dumps(self):
        buf_size = lib.roaring_bitmap_size_in_bytes(self._croaring)
        out = ffi.new('char[%d]' % (buf_size))
        size = lib.roaring_bitmap_serialize(self._croaring, out)
        if size < 0:
            return None
        return ffi.buffer(out)[:size]

    def clear(self):
        lib.roaring_bitmap_clear(self._croaring)

    @classmethod
    def loads(cls, buf):
        inbuf = ffi.new('char[%d]'%(len(buf)), buf)
        _croaring = lib.roaring_bitmap_deserialize(inbuf)
        return cls(croaring = _croaring)

    def minimum(self):
        return lib.roaring_bitmap_minimum(self._croaring)

    def maximum(self):
        return lib.roaring_bitmap_maximum(self._croaring)

    def min(self):
        return lib.roaring_bitmap_minimum(self._croaring)

    def max(self):
        return lib.roaring_bitmap_maximum(self._croaring)

    def bytes_size(self):
        return lib.roaring_bitmap_size_in_bytes(self._croaring)

    def run_optimize(self):
        return lib.roaring_bitmap_run_optimize(self._croaring)

    def is_empty(self):
        return bool(lib.roaring_bitmap_is_empty(self._croaring))

    def shrink(self):
        return lib.roaring_bitmap_shrink_to_fit(self._croaring)

    def intersect(self, ohter):
        return bool(lib.roaring_bitmap_intersect(self._croaring, other._croaring))

    def index(self, value):
        return lib.roaring_bitmap_rank(self._croaring, ffi.cast("uint32_t", value))

    def to_array(self):
        size = len(self)
        out = ffi.new('uint32_t[%d]' % (size)) 
        lib.roaring_bitmap_to_uint32_array(self._croaring, out)
        ar = array.array('I', out)
        return ar

def load_from_file(file_name):
    result = None
    try:
        with open(file_name, 'fb') as f:
            result = BitSet.loads(f.read())
    except Exception as e:
        import logging
        logging.error(str(e))
    return result

def load_from_s3(file_name):
    result = None
    try:
        import s3fs
        s3 = s3fs.S3FileSystem(anon=False)
        result = BitSet.loads(s3.cat(file_name))
    except Exception as e:
        import logging
        logging.error(str(e))
    return result

def validate_bitset(bitset, filetype):
    if isinstance(bitset, BitSet):
        bs = bitset
    elif filetype == "s3":
        bs= load_from_s3(bitset)
    else:
        bs = load_from_file(bitset)
        
    if not bs:
        bs = BitSet()
    return bs
        
def calculate_len(bitset, filetype=""):
    bs = validate_bitset(bitset, filetype)
    return len(bs)
        
def calculate_and(bitset1, bitset2, filetype1="", filetype2=""):
    bs1 = validate_bitset(bitset1, filetype1)
    bs2 = validate_bitset(bitset2, filetype2)
    return bs1 & bs2

def calculate_or(bitset1, bitset2, filetype1="", filetype2=""):
    bs1 = validate_bitset(bitset1, filetype1)
    bs2 = validate_bitset(bitset2, filetype2)
    return bs1 | bs2

def calculate_xor(bitset1, bitset2, filetype1="", filetype2=""):
    bs1 = validate_bitset(bitset1, filetype1)
    bs2 = validate_bitset(bitset2, filetype2)
    return bs1 ^ bs2

def calculate_sub(bitset1, bitset2, filetype1="", filetype2=""):
    bs1 = validate_bitset(bitset1, filetype1)
    bs2 = validate_bitset(bitset2, filetype2)
    return bs1 - bs2

def calculate_add(bitset1, bitset2, filetype1="", filetype2=""):
    bs1 = validate_bitset(bitset1, filetype1)
    bs2 = validate_bitset(bitset2, filetype2)
    return bs1 + bs2
