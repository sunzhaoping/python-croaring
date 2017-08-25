# -*- coding: utf8 -*-
from __future__ import absolute_import, division, print_function, with_statement
import sys
import os
import binascii
import threading
import collections
from cffi import FFI
from cffi.verifier import Verifier

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
typedef bool (*roaring_iterator)(uint32_t value, void *param);
typedef bool (*roaring_iterator64)(uint64_t value, void *param);
typedef struct roaring_bitmap_s roaring_bitmap_t;
typedef struct roaring_uint32_iterator_s roaring_uint32_iterator_t;
roaring_bitmap_t *roaring_bitmap_create();
void roaring_bitmap_free(roaring_bitmap_t *r);
void roaring_bitmap_add(roaring_bitmap_t *r, uint32_t val);
uint64_t roaring_bitmap_get_cardinality(const roaring_bitmap_t *ra);
bool roaring_bitmap_contains(const roaring_bitmap_t *r, uint32_t val);
void roaring_bitmap_remove(roaring_bitmap_t *r, uint32_t val);
roaring_bitmap_t *roaring_bitmap_or(const roaring_bitmap_t *x1, const roaring_bitmap_t *x2);
void roaring_bitmap_or_inplace(roaring_bitmap_t *x1, const roaring_bitmap_t *x2);
roaring_bitmap_t *roaring_bitmap_and(const roaring_bitmap_t *x1, const roaring_bitmap_t *x2);
void roaring_bitmap_and_inplace(roaring_bitmap_t *x1, const roaring_bitmap_t *x2);
roaring_bitmap_t *roaring_bitmap_xor(const roaring_bitmap_t *x1, const roaring_bitmap_t *x2);
void roaring_bitmap_xor_inplace(roaring_bitmap_t *x1, const roaring_bitmap_t *x2);
roaring_bitmap_t *roaring_bitmap_andnot(const roaring_bitmap_t *x1, const roaring_bitmap_t *x2);
void roaring_bitmap_andnot_inplace(roaring_bitmap_t *x1, const roaring_bitmap_t *x2);
roaring_bitmap_t *roaring_bitmap_copy(const roaring_bitmap_t *r);
roaring_bitmap_t *roaring_bitmap_or_many_heap(uint32_t number,const roaring_bitmap_t **x);
roaring_bitmap_t *roaring_bitmap_or_many(uint32_t number,const roaring_bitmap_t **x);
bool roaring_bitmap_run_optimize(roaring_bitmap_t *r);
size_t roaring_bitmap_shrink_to_fit(roaring_bitmap_t *r);
roaring_bitmap_t *roaring_bitmap_deserialize(const void *buf);
bool roaring_bitmap_is_empty(const roaring_bitmap_t *ra);
size_t roaring_bitmap_serialize(const roaring_bitmap_t *ra, char *buf);
size_t roaring_bitmap_size_in_bytes(const roaring_bitmap_t *ra);
bool roaring_bitmap_equals(const roaring_bitmap_t *ra1, const roaring_bitmap_t *ra2);
bool roaring_bitmap_is_subset(const roaring_bitmap_t *ra1, const roaring_bitmap_t *ra2);
uint64_t roaring_bitmap_rank(const roaring_bitmap_t *bm, uint32_t x);
uint32_t roaring_bitmap_minimum(const roaring_bitmap_t *bm);
uint32_t roaring_bitmap_maximum(const roaring_bitmap_t *bm);
bool roaring_bitmap_select(const roaring_bitmap_t *bm, uint32_t rank, uint32_t *element);
bool roaring_bitmap_intersect(const roaring_bitmap_t *x1, const roaring_bitmap_t *x2);
roaring_uint32_iterator_t *roaring_create_iterator(const roaring_bitmap_t *ra);
bool roaring_advance_uint32_iterator(roaring_uint32_iterator_t *it);
void roaring_free_uint32_iterator(roaring_uint32_iterator_t *it);
bool iterator_has_value(const roaring_uint32_iterator_t *it);
uint32_t iterator_current_value(const roaring_uint32_iterator_t *it);
"""

SOURCE = """
#include <roaring.c>
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
    def __init__(self, iterable=(), croaring=None):
        self._croaring = croaring or lib.roaring_bitmap_create()
        if iterable:
            for item in iterable:
                self.add(item)

    def __contains__(self, value):
        if isinstance(value, Roaring):
            return lib.roaring_bitmap_is_subset(value._croaring, self._croaring)
        return lib.roaring_bitmap_contains(self._croaring, ffi.cast("uint32_t", value))

    def __iter__(self):
        iter = lib.roaring_create_iterator(self._croaring)
        while lib.iterator_has_value(iter):
           yield lib.iterator_current_value(iter)
           lib.roaring_advance_uint32_iterator(iter)
        lib.roaring_free_uint32_iterator(iter)

    def __and__(self, other):
        _croaring =  lib.roaring_bitmap_and(self._croaring,  other._croaring)
        return Roaring(croaring = _croaring)

    def __iand__(self, other):
        lib.roaring_bitmap_and_inplace(self._croaring,  other._croaring)
        return self

    def __or__(self, other):
        _croaring =  lib.roaring_bitmap_or(self._croaring,  other._croaring)
        return Roaring(croaring = _croaring)

    def __ior__(self, other):
        lib.roaring_bitmap_or_inplace(self._croaring,  other._croaring)
        return self

    def __xor__(self, other):
        _croaring =  lib.roaring_bitmap_xor(self._croaring,  other._croaring)
        return Roaring(croaring = _croaring)

    def __ixor__(self, other):
        lib.roaring_bitmap_xor_inplace(self._croaring,  other._croaring)
        return self

    def __sub__(self, other):
        _croaring =  lib.roaring_bitmap_andnot(self._croaring,  other._croaring)
        return Roaring(croaring = _croaring)

    def __isub__(self, other):
        lib.roaring_bitmap_andnot_inplace(self._croaring,  other._croaring)
        return self

    def __add__(self, other):
        return self.__or__(other)

    def __iadd__(self, other):
        return self.__ior__(other)

    def __del__(self):
        lib.roaring_bitmap_free(self._croaring)

    def __getitem__(self, index):
        out = ffi.new('uint32_t *')
        if lib.roaring_bitmap_select(self._croaring, ffi.cast("uint32_t", index), out):
            return out[0]
        else:
            raise IndexError("bitset index out of range")

    def __len__(self):
        return lib.roaring_bitmap_get_cardinality(self._croaring)

    def __eq__(self, bitmap):
        return lib.roaring_bitmap_equals(self._croaring, bitmap._croaring)

    def __copy__(self):
        return self.copy()

    def copy(self):
        _croaring =  lib.roaring_bitmap_copy(self._croaring)
        return Roaring(croaring = _croaring)

    def __getstate__(self):
        return self.dumps()

    def __setstate__(self, value):
        inbuf = ffi.new('char[%d]'%(len(value)), value)
        self._croaring = lib.roaring_bitmap_deserialize(inbuf)

    @classmethod
    def union(cls, *bitmaps):
        _croaring =  lib.roaring_bitmap_or_many(len(bitmaps),  [b._croaring for b in bitmaps])
        return Roaring(croaring = _croaring)

    @classmethod
    def union_heap(cls, *bitmaps):
        _croaring =  lib.roaring_bitmap_or_many_heap(len(bitmaps),  [b._croaring for b in bitmaps])
        return Roaring(croaring = _croaring)

    def add(self, value):
        lib.roaring_bitmap_add(self._croaring, ffi.cast("uint32_t", value))

    def discard(self, value):
        try:
            self.remove(value)
        except KeyError:
            pass

    def pop(self, maxvalue = False):
        result = self.maximum() if maxvalue else self.minimum()
        self.remove(result)
        return result

    def remove(self, value):
        try:
            lib.roaring_bitmap_remove(self._croaring, ffi.cast("uint32_t", value))
        except:
            raise ValueError("Bitset.remove(): %s is not in bitset" % (value))

    def dumps(self):
        buf_size = lib.roaring_bitmap_size_in_bytes(self._croaring)
        out = ffi.new('char[%d]' % (buf_size))
        size = lib.roaring_bitmap_serialize(self._croaring, out)
        if size < 0:
            return None
        return ffi.buffer(out)[:size]

    @classmethod
    def loads(cls, buf):
        inbuf = ffi.new('char[%d]'%(len(buf)), buf)
        _croaring = lib.roaring_bitmap_deserialize(inbuf)
        return cls(croaring = _croaring)

    def minimum(self):
        return lib.roaring_bitmap_minimum(self._croaring)

    def maximum(self):
        return lib.roaring_bitmap_minimum(self._croaring)

    def bytes_size(self):
        return lib.roaring_bitmap_size_in_bytes(self._croaring)

    def run_optimize(self):
        return lib.roaring_bitmap_run_optimize(self._croaring)

    def is_empty(self):
        return lib.roaring_bitmap_is_empty(self._croaring)

    def shrink(self):
        return lib.roaring_bitmap_shrink_to_fit(self._croaring)

    def intersect(self, ohter):
        return lib.roaring_bitmap_intersect(self._croaring, other._croaring)

    def index(self, value):
        return lib.roaring_bitmap_rank(self._croaring, ffi.cast("uint32_t", value))

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

def calculate_len(bitset, filetype=""):
    if isinstance(bitset, BitSet):
        bs = bitset
    elif filetype == "s3":
        bs= load_from_s3(bitset)
    else:
        bs = load_from_file(bitset)
    return len(bs) if bs else 0
