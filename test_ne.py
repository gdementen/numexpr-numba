import numexpr as ne
import numpy as np
from timeit import repeat
from numba import autojit, __version__
import meta
import ast

print "using numba", __version__

size = (1000, 1000)

def timefunc(correct, func, *args, **kwargs):
    print func.__name__.ljust(20),
    # Warming up
    res = func(*args, **kwargs)
    failed = False
    if correct is not None:
        if not np.allclose(res, correct):
            failed = True
            print "correct", correct, correct.dtype
            print "res", res, res.dtype
            print "diff", res - correct
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
expr = 'where((a <= 5) & (~(c > 2)), -b + 3 * b + b ** 0 - 4.3 * c ** 3, 4 * log(b) * b - 3.3 * (c % 5) * exp(c) * c)'
# expr = 'abs(a - c + b)'
# expr = '2.1 * a + 3.2 * b * b + 4.3 * c * c * c'
# expr = 'sum(a + b + c, axis=0)'
# expr = 'a & b & c' # works
# expr = 'sum(a & b & c)' # fails
# expr = 'sum(b)'

def np_nopow(a, b, c):
    return eval(expr)

def ne_nopow(a, b, c):
    return ne.evaluate(expr) #, out=r)
    
a = np.random.randint(10, size=size)
# a = a < 5
b = np.random.rand(*size)
# b = b < 0.5
c = np.random.randint(10, size=size)
# c = c < 5

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


correct = timefunc(None, np_nopow, a, b, c)
timefunc(correct, ne_nopow, a, b, c)
ne.set_num_threads(1)
timefunc(correct, ne_nopow, a, b, c)
