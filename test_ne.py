from __future__ import division

import numexpr as ne
from numexpr import NumExpr, E
import numpy as np
from numpy.testing import *
from timeit import repeat
from numba import autojit, __version__
import meta
import ast

print "using numba", __version__
print "using numexpr", ne.__version__

def timefunc(correct, func, *args, **kwargs):
    print func.__name__.ljust(20),
    # Warming up
    res = func(*args, **kwargs)
    failed = False
    if correct is not None:
        if not np.all((np.isnan(res) & np.isnan(correct)) | (res == correct)):
        # if not np.allclose(res, correct):
            failed = True
            print
            print "correct"
            print correct
            print correct.dtype
            print "res"
            print res
            print res.dtype
            print "diff"
            print res - correct
    # time it
    if failed:
        print "---- ms"
    else:
        print '{:>5.0f} ms'.format(min(repeat(lambda: func(*args, **kwargs),
                                            number=5, repeat=2)) * 1000)
                                          
                                            
    return res

where = np.where
log = np.log
exp = np.exp
abs = np.absolute
sum = np.sum

# mod is slow
# pow is slow
# abs is slow

# def __expr_func__(a, b, c):
    # length = len(a)
    # result = np.empty(length, dtype=np.double)
    # for i in range(length):
        # result[i] = abs(a[i] - c[i] + b[i])
    # return result

# abs_ast = meta.decompiler.decompile_func(__expr_func__)
# print ast.dump(abs_ast, annotate_fields=False)

# abs_nb = autojit(__expr_func__)

# expr = 'where((a <= 5) & (~(c > 2)), -b + 3 * b ** 60 + b ** 0 - 4.3 * c ** 3, 4 * log(b) * b - 3.3 * (c % 5) * exp(c) * c)'
# expr = 'where((a <= 5) & (~(c > 2)), -b + 3 * b + b ** 0 - 4.3 * c ** 3, 4 * log(b) * b - 3.3 * (c % 5) * exp(c) * c)'
# expr = 'abs(a - c + b)'
# expr = '2.1 * a + 3.2 * b * b + 4.3 * c * c * c'
# expr = 'sum(a + b + c, axis=0)'
# expr = 'sum(a)'
# expr = 'a & b & c' # works
# expr = 'sum(a & b & c)' # fails
# expr = 'sum(b)'
# expr = '(a + 1) % (b + 3)'
# expr = 'a >= b'
# expr = 'a**2 + b**2 + 2*a*b'
# expr = 'a*a + b*b + 2*a*b'
# expr = 'a / b'
# expr = 'a != a'
expr = 'a / 2'

def func_np(a):
# def func_np(a, b):
# def func_np(a, b, c):
    return eval(expr)

def func_ne(a):
# def func_ne(a, b):
# def func_ne(a, b, c):
    # res = ne.evaluate(expr, out=r)
    # assert res is r
    # return res
    return ne.evaluate(expr)

# @autojit    
# def func_nb(a, b):
    # return a*a + b*b + 2*a*b
#    return a**2 + b**2 + 2*a*b

# BLOCK_SIZE1 = 128
# BLOCK_SIZE2 = 8
# str_list1 = [b'foo', b'bar', b'', b'  ']
# str_list2 = [b'foo', b'', b'x', b' ']
# str_nloops = len(str_list1) * (BLOCK_SIZE1 + BLOCK_SIZE2 + 1)
# a = np.array(str_list1 * str_nloops)
# b = np.array(str_list2 * str_nloops)
# expr = 'a >= b'
    
# array_size = 100
# a = np.arange(array_size, dtype=np.int) #[::2]
# b = np.arange(array_size, dtype=np.float32) #/ array_size
    
# a = np.arange(100.0)

# size = (1000, 1000)
size = (10,)

# - withnan: sum([1.0, nan, 2.0], skip_na=False)
# - qshow(withnan)
# - assertTrue(withnan != withnan)
# a = np.array([1.0, np.nan, 2.0])

# print "nan?", a != a
# y = ne.evaluate("a != a")
# print "nan in numba?", y


# a = np.random.randint(10, size=size)
# a = np.arange(5)
# a = np.array([0, 1, 0, 3, 2]) 
a = np.random.rand(*size)
# a = a < 5

# b = np.random.randint(10, size=size)
# b = np.array([0, 0, 1, 2, 3])
# b = np.arange(1000, dtype=np.float64)
# b = np.random.rand(*size)
# b = b < 0.5
# c = np.random.randint(10, size=size)
# c = np.random.rand(*size)
# c = c < 5

# r = np.empty(size, dtype=np.float64)

# r = a & b & c
def info(a):
    print
    print "shape", a.shape, "dtype", a.dtype
    print "flags"
    print a.flags
    print a
    print "sum", np.sum(a)

# print "numpy"
# info(r)
# info(r.astype(np.uint8, copy=False))

# r = np.empty(size, dtype=float)
#r = np.empty(1000, dtype=float)

# r = abs_nb(a, b, c)
# r = abs_nb(a, b, c)
# r = abs_nb(a, b, c)

# a = np.random.rand(size)
# b = np.random.rand(size)
# c = np.random.rand(size)


correct = timefunc(None, func_np, a)
timefunc(correct, func_ne, a)
print "with one thread"
ne.set_num_threads(1)
timefunc(correct, func_ne, a)

# correct = timefunc(None, func_np, a, b)
# timefunc(correct, func_ne, a, b)
# print "with one thread"
# ne.set_num_threads(1)
# timefunc(correct, func_ne, a, b)

# correct = timefunc(None, func_np, a, b, c)
# timefunc(correct, func_ne, a, b, c)
# print "with one thread"
# ne.set_num_threads(1)
# timefunc(correct, func_ne, a, b, c)

# func = NumExpr(E.a)
# x = np.arange(100.0)
# y = func(x)
# assert_array_equal(x, y)
