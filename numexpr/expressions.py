###################################################################
#  Numexpr - Fast numerical array expression evaluator for NumPy.
#
#      License: MIT
#      Author:  See AUTHORS.txt
#
#  See LICENSE.txt and LICENSES/*.txt for details about copyright and
#  rights to use.
####################################################################

__all__ = ['E']

import ast
import operator
import sys
import threading

import numpy

# Declare a double type that does not exist in Python space
double = numpy.double

# The default kind for undeclared variables
default_kind = 'double'
if sys.version_info[0] < 3:
    int_ = int
    long_ = long
else:
    int_ = numpy.int32
    long_ = numpy.int64

type_to_kind = {bool: 'bool', int_: 'int', long_: 'long', float: 'float',
                double: 'double', complex: 'complex', bytes: 'bytes'}
kind_to_type = {'bool': bool, 'int': int_, 'long': long_, 'float': float,
                'double': double, 'complex': complex, 'bytes': bytes}
kind_rank = ['bool', 'int', 'long', 'float', 'double', 'complex', 'none']
scalar_constant_types = [bool, int_, long, float, double, complex, bytes]

# Final corrections for Python 3 (mainly for PyTables needs)
if sys.version_info[0] > 2:
    type_to_kind[str] = 'str'
    kind_to_type['str'] = str
    scalar_constant_types.append(str)
scalar_constant_types = tuple(scalar_constant_types)


class Expression(object):
    def __init__(self):
        object.__init__(self)

    def __getattr__(self, name):
        if name.startswith('_'):
            return self.__dict__[name]
        else:
            return VariableNode(name, default_kind)

E = Expression()

class Context(threading.local):
    initialized = False
    def __init__(self, dict_):
        if self.initialized:
            raise SystemError('__init__ called too many times')
        self.initialized = True
        self.__dict__.update(dict_)
    def get(self, value, default):
        return self.__dict__.get(value, default)
    def get_current_context(self):
        return self.__dict__
    def set_new_context(self, dict_):
        self.__dict__.update(dict_)

# This will be called each time the local object is used in a separate thread
_context = Context({})

def get_optimization():
    return _context.get('optimization', 'none')

# helper functions for creating __magic__ methods
def ophelper(f):
    def func(*args):
        args = list(args)
        for i, x in enumerate(args):
            if isConstant(x):
                args[i] = x = ConstantNode(x)
            if not isinstance(x, ExpressionNode):
                raise TypeError("unsupported object type: %s" % type(x))
        return f(*args)
    func.__name__ = f.__name__
    func.__doc__ = f.__doc__
    func.__dict__.update(f.__dict__)
    return func

def allConstantNodes(args):
    "returns True if args are all ConstantNodes."
    for x in args:
        if not isinstance(x, ConstantNode):
            return False
    return True

def isConstant(ex):
    "Returns True if ex is a constant scalar of an allowed type."
    return isinstance(ex, scalar_constant_types)

def commonKind(nodes):
    node_kinds = [node.astKind for node in nodes]
    str_count = node_kinds.count('bytes') + node_kinds.count('str')
    if 0 < str_count < len(node_kinds):  # some args are strings, but not all
        raise TypeError("strings can only be operated with strings")
    if str_count > 0:  # if there are some, all of them must be
        return 'bytes'
    n = -1
    for x in nodes:
        n = max(n, kind_rank.index(x.astKind))
    return kind_rank[n]

max_int32 = 2147483647
min_int32 = -max_int32 - 1

def bestConstantType(x):
    # ``numpy.string_`` is a subclass of ``bytes``
    if isinstance(x, (bytes, str)):
        return bytes
    # Numeric conversion to boolean values is not tried because
    # ``bool(1) == True`` (same for 0 and False), so 0 and 1 would be
    # interpreted as booleans when ``False`` and ``True`` are already
    # supported.
    if isinstance(x, (bool, numpy.bool_)):
        return bool
    # ``long`` objects are kept as is to allow the user to force
    # promotion of results by using long constants, e.g. by operating
    # a 32-bit array with a long (64-bit) constant.
    if isinstance(x, (long_, numpy.int64)):
        return long_
    # ``double`` objects are kept as is to allow the user to force
    # promotion of results by using double constants, e.g. by operating
    # a float (32-bit) array with a double (64-bit) constant.
    if isinstance(x, double):
        return double
    if isinstance(x, (int, numpy.integer)):
        # Constants needing more than 32 bits are always
        # considered ``long``, *regardless of the platform*, so we
        # can clearly tell 32- and 64-bit constants apart.
        if not (min_int32 <= x <= max_int32):
            return long_
        return int_
    # The duality of float and double in Python avoids that we have to list
    # ``double`` too.
    for converter in float, complex:
        try:
            y = converter(x)
        except StandardError, err:
            continue
        if y == x:
            return converter


def getKind(x):
    converter = bestConstantType(x)
    return type_to_kind[converter]

def binop(opname, reversed=False, kind=None):
    # Getting the named method from self (after reversal) does not
    # always work (e.g. int constants do not have a __lt__ method).
    opfunc = getattr(operator, "__%s__" % opname)
    @ophelper
    def operation(self, other):
        if reversed:
            self, other = other, self
        if allConstantNodes([self, other]):
            return ConstantNode(opfunc(self.value, other.value))
        else:
            return OpNode(opname, (self, other), kind=kind)
    return operation

def func(func, minkind=None, maxkind=None):
    @ophelper
    def function(*args):
        if allConstantNodes(args):
            return ConstantNode(func(*[x.value for x in args]))
        kind = commonKind(args)
        if kind in ('int', 'long'):
            # Exception for following NumPy casting rules
            #FIXME: this is not always desirable. The following
            # functions which return ints (for int inputs) on numpy
            # but not on numexpr: copy, abs, fmod, ones_like
            kind = 'double'
        else:
            # Apply regular casting rules
            if minkind and kind_rank.index(minkind) > kind_rank.index(kind):
                kind = minkind
            if maxkind and kind_rank.index(maxkind) < kind_rank.index(kind):
                kind = maxkind
        name = func.__name__
        if name == "absolute":
            name = "abs"
        return FuncNode(name, args, kind)
    return function

@ophelper
def where_func(a, b, c):
    if isinstance(a, ConstantNode):
        #FIXME: This prevents where(True, a, b)
        raise ValueError("too many dimensions")
    if allConstantNodes([a,b,c]):
        return ConstantNode(numpy.where(a, b, c))
    return FuncNode('where', [a,b,c])

def encode_axis(axis):
    if isinstance(axis, ConstantNode):
        axis = axis.value
    if axis is not None:
        if axis < 0:
            raise ValueError("negative axis are not supported")
        if axis > 254:
            raise ValueError("cannot encode axis")
    return RawNode(axis)

def sum_func(a, axis=None):
    axis = encode_axis(axis)
    #XXX: I suspect these two if are swapped (see prod_func below)
    if isinstance(a, ConstantNode):
        return a
    if isinstance(a, (bool, int_, long_, float, double, complex)):
        a = ConstantNode(a)
    return FuncNode('sum', [a, axis], kind=a.astKind)

def prod_func(a, axis=None):
    axis = encode_axis(axis)
    if isinstance(a, (bool, int_, long_, float, double, complex)):
        a = ConstantNode(a)
    if isinstance(a, ConstantNode):
        return a
    return FuncNode('prod', [a, axis], kind=a.astKind)

@ophelper
def div_op(a, b):
    if get_optimization() in ('moderate', 'aggressive'):
        if (isinstance(b, ConstantNode) and
            (a.astKind == b.astKind) and
            a.astKind in ('float', 'double', 'complex')):
            return OpNode('mul', [a, ConstantNode(1./b.value)])
    return OpNode('div', [a,b])

@ophelper
def truediv_op(a, b):
    if get_optimization() in ('moderate', 'aggressive'):
        if (isinstance(b, ConstantNode) and
            (a.astKind == b.astKind) and
            a.astKind in ('float', 'double', 'complex')):
            return OpNode('mul', [a, ConstantNode(1./b.value)])
    kind = commonKind([a, b])
    if kind in ('bool', 'int', 'long'):
        kind = 'double'
    return OpNode('div', [a, b], kind=kind)

@ophelper
def rtruediv_op(a, b):
    return truediv_op(b, a)

@ophelper
def pow_op(a, b):
    if allConstantNodes([a, b]):
        return ConstantNode(a**b)
    if isinstance(b, ConstantNode):
        x = b.value
        if get_optimization() == 'aggressive':
            RANGE = 50 # Approximate break even point with pow(x,y)
            # Optimize all integral and half integral powers in [-RANGE, RANGE]
            # Note: for complex numbers RANGE could be larger.
            if (int(2*x) == 2*x) and (-RANGE <= abs(x) <= RANGE):
                n = int_(abs(x))
                ishalfpower = int_(abs(2*x)) % 2
                def multiply(x, y):
                    if x is None: return y
                    return OpNode('mul', [x, y])
                r = None
                p = a
                mask = 1
                while True:
                    if (n & mask):
                        r = multiply(r, p)
                    mask <<= 1
                    if mask > n:
                        break
                    p = OpNode('mul', [p,p])
                if ishalfpower:
                    kind = commonKind([a])
                    if kind in ('int', 'long'):
                        kind = 'double'
                    r = multiply(r, OpNode('sqrt', [a], kind))
                if r is None:
                    r = OpNode('ones_like', [a])
                if x < 0:
                    r = OpNode('div', [ConstantNode(1), r])
                return r
        if get_optimization() in ('moderate', 'aggressive'):
            if x == -1:
                return OpNode('div', [ConstantNode(1),a])
            if x == 0:
                return OpNode('ones_like', [a])
            if x == 0.5:
                kind = a.astKind
                if kind in ('int', 'long'): kind = 'double'
                return FuncNode('sqrt', [a], kind=kind)
            if x == 1:
                return a
            if x == 2:
                return OpNode('mul', [a,a])
    return OpNode('pow', [a,b])

# The functions and the minimum and maximum types accepted
func_defs = {
    'copy': (numpy.copy,),
    'ones_like': (numpy.ones_like,),
    'sqrt': (numpy.sqrt, 'float'),

    'sin': (numpy.sin, 'float'),
    'cos': (numpy.cos, 'float'),
    'tan': (numpy.tan, 'float'),
    'arcsin': (numpy.arcsin, 'float'),
    'arccos': (numpy.arccos, 'float'),
    'arctan': (numpy.arctan, 'float'),

    'sinh': (numpy.sinh, 'float'),
    'cosh': (numpy.cosh, 'float'),
    'tanh': (numpy.tanh, 'float'),
    'arcsinh': (numpy.arcsinh, 'float'),
    'arccosh': (numpy.arccosh, 'float'),
    'arctanh': (numpy.arctanh, 'float'),

    'fmod': (numpy.fmod, 'float'),
    'arctan2': (numpy.arctan2, 'float'),

    'log': (numpy.log, 'float'),
    'log1p': (numpy.log1p, 'float'),
    'log10': (numpy.log10, 'float'),
    'exp': (numpy.exp, 'float'),
    'expm1': (numpy.expm1, 'float'),

    'abs': (numpy.absolute, 'float'),

    'complex': (complex, 'complex'),
}

functions = dict((k, func(*v)) for k, v in func_defs.items())
functions.update({
    'real': (numpy.real, 'double', 'double'),
    'imag': (numpy.imag, 'double', 'double'),
    'where' : where_func,
    'sum' : sum_func,
    'prod' : prod_func,
})

class ExpressionNode(object):
    """An object that represents a generic number object.

    This implements the number special methods so that we can keep
    track of how this object has been used.
    """
    astType = 'generic'

    def __init__(self, value=None, kind=None, children=None):
        object.__init__(self)
        self.value = value
        if kind is None:
            kind = 'none'
        self.astKind = kind
        if children is None:
            self.children = ()
        else:
            self.children = tuple(children)

    def get_real(self):
        if self.astType == 'constant':
            return ConstantNode(complex(self.value).real)
        return OpNode('real', (self,), 'double')
    real = property(get_real)

    def get_imag(self):
        if self.astType == 'constant':
            return ConstantNode(complex(self.value).imag)
        return OpNode('imag', (self,), 'double')
    imag = property(get_imag)

    def __str__(self):
        return '%s(%s, %s, %s)' % (self.__class__.__name__, self.value,
                                   self.astKind, self.children)
    def __repr__(self):
        return self.__str__()

    def __neg__(self):
        return OpNode('neg', (self,))
    def __invert__(self):
        return OpNode('invert', (self,))
    def __pos__(self):
        return self

    # The next check is commented out. See #24 for more info.

    def __nonzero__(self):
         raise TypeError("You can't use Python's standard boolean operators in "
                         "NumExpr expressions. You should use their bitwise "
                         "counterparts instead: '&' instead of 'and', "
                         "'|' instead of 'or', and '~' instead of 'not'.")

    __add__ = __radd__ = binop('add')
    __sub__ = binop('sub')
    __rsub__ = binop('sub', reversed=True)
    __mul__ = __rmul__ = binop('mul')
    if sys.version_info[0] < 3:
        __div__ = div_op
        __rdiv__ = binop('div', reversed=True)
    __truediv__ = truediv_op
    __rtruediv__ = rtruediv_op
    __pow__ = pow_op
    __rpow__ = binop('pow', reversed=True)
    __mod__ = binop('mod')
    __rmod__ = binop('mod', reversed=True)

    # boolean operations

    __and__ = binop('and', kind='bool')
    __or__ = binop('or', kind='bool')

    __gt__ = binop('gt', kind='bool')
    __ge__ = binop('ge', kind='bool')
    __eq__ = binop('eq', kind='bool')
    __ne__ = binop('ne', kind='bool')
    __lt__ = binop('gt', reversed=True, kind='bool')
    __le__ = binop('ge', reversed=True, kind='bool')



class LeafNode(ExpressionNode):
    leafNode = True

class VariableNode(LeafNode):
    astType = 'variable'
    def __init__(self, value=None, kind=None, children=None):
        LeafNode.__init__(self, value=value, kind=kind)
    
    def toPython(self):
        return ast.Name(self.value, ast.Load())
        

class RawNode(object):
    """Used to pass raw integers to interpreter.
    For instance, for selecting what function to use in func1.
    Purposely don't inherit from ExpressionNode, since we don't wan't
    this to be used for anything but being walked.
    """
    astType = 'raw'
    astKind = 'none'
    def __init__(self, value):
        self.value = value
        self.children = ()
    def __str__(self):
        return 'RawNode(%s)' % (self.value,)
    __repr__ = __str__


class ConstantNode(LeafNode):
    astType = 'constant'
    def __init__(self, value=None, children=None):
        kind = getKind(value)
        # Python float constants are double precision by default
        if kind == 'float':
            kind = 'double'
        LeafNode.__init__(self, value=value, kind=kind)
    def __neg__(self):
        return ConstantNode(-self.value)
    def __invert__(self):
        return ConstantNode(~self.value)
    def toPython(self):
        return ast.Num(self.value)

        
class OpNode(ExpressionNode):
    astType = 'op'
    def __init__(self, opcode=None, args=None, kind=None):
        if (kind is None) and (args is not None):
            kind = commonKind(args)
        ExpressionNode.__init__(self, value=opcode, kind=kind, children=args)
    
    def toPython(self):
        # Invert | Not | UAdd | USub
        unaryops = {
            'invert': ast.Not,
            # 'invert': ast.Invert, # use bitwise op, not logical op
            'neg': ast.USub
        }
        # And | Or 
        boolops = {
            'and': ast.And,
            'or': ast.Or 
        }
        # Eq | NotEq | Lt | LtE | Gt | GtE # Is | IsNot | In | NotIn
        cmpops = {
            'gt': ast.Gt,
            'ge': ast.GtE,
            'eq': ast.Eq,
            'ne': ast.NotEq
            # lt and le are handled by their reverse operation
        }   
        binops = {
            'add': ast.Add,
            'sub': ast.Sub,
            'mul': ast.Mult,
            'div': ast.Div,
            'mod': ast.Mod,
            'pow': ast.Pow,
            # translate to bit ops not "logical" ops
            # 'and': ast.BitAnd,
            # 'or': ast.BitOr 
        }
        op = self.value
        args = [c.toPython() for c in self.children]
        if op in unaryops:
            arg = args[0]
            # force conversion to int to workaround numba #238
            # it fixes age * ~gender but breaks count(~gender) :(
            # if op == 'invert':
                # arg = ast.BinOp(arg, ast.Add(), ast.Num(0))
            return ast.UnaryOp(unaryops[op](), arg)
        elif op in boolops:
            return ast.BoolOp(boolops[op](), args)
        elif op in cmpops:
            return ast.Compare(args[0], [cmpops[op]()], [args[1]])
        elif op == 'ones_like':
            return ast.Num(1)
        else:
            binop = ast.BinOp(args[0], binops[op](), args[1])
            # shield against integer division by 0
            if op == 'div' and self.children[1].astKind in ('int', 'long'):
                return ast.IfExp(args[1], binop, ast.Num(0))
            return binop

class FuncNode(OpNode):
    def __init__(self, opcode=None, args=None, kind=None):
        if (kind is None) and (args is not None):
            kind = commonKind(args)
        OpNode.__init__(self, opcode, args, kind)

    def toPython(self):
        args = [c.toPython() if c is not None else None for c in self.children]
        if self.value == "where":
            return ast.IfExp(*args)
        elif self.value == "real":
            return ast.Attribute(args[0], 'real', ast.Load())
        elif self.value == "imag":
            return ast.Attribute(args[0], 'imag', ast.Load())
        else:
            return ast.Call(ast.Name(self.value, ast.Load()), args, [],
                            None, None)
        