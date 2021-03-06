###################################################################
#  Numexpr - Fast numerical array expression evaluator for NumPy.
#
#      License: MIT
#      Author:  See AUTHORS.txt
#
#  See LICENSE.txt and LICENSES/*.txt for details about copyright and
#  rights to use.
####################################################################

import __future__
import sys
import ast
from ctypes import pythonapi, c_void_p
import threading

import numpy as np
from numba import jit, void, b1, i4, i8, f4, f8, c16, string_
import meta

# is_cpu_amd_intel is imported *from here* by pytables :(
from numexpr import expressions, use_vml, is_cpu_amd_intel

from numexpr.utils import CacheDict
from numexpr import utils

# Declare a double type that does not exist in Python space
double = np.double
if sys.version_info[0] < 3:
    int_ = int
    long_ = long
else:
    int_ = np.int32
    long_ = np.int64

typecode_to_kind = {'b': 'bool', 'i': 'int', 'l': 'long', 'f': 'float',
                    'd': 'double', 'c': 'complex', 's': 'bytes', 'n' : 'none'}
kind_to_typecode = {'bool': 'b', 'int': 'i', 'long': 'l', 'float': 'f',
                    'double': 'd', 'complex': 'c', 'bytes': 's', 'none' : 'n'}
type_to_numba = {bool: b1, int_: i4, long_: i8, float: f4,
                 double: f8, complex: c16, bytes: string_}
type_to_typecode = {bool: 'b', int_: 'i', long_:'l', float:'f',
                    double: 'd', complex: 'c', bytes: 's'}
type_to_kind = expressions.type_to_kind
kind_to_type = expressions.kind_to_type
default_type = kind_to_type[expressions.default_kind]

# Final addtions for Python 3 (mainly for PyTables needs)
if sys.version_info[0] > 2:
    typecode_to_kind['s'] = 'str'
    kind_to_typecode['str'] = 's'
    type_to_typecode[str] = 's'

scalar_constant_kinds = kind_to_typecode.keys()


class ASTNode(object):
    def __init__(self, astType='generic', astKind='unknown',
                 value=None, children=()):
        raise NotImplementedError()

def expressionToAST(ex):
    raise NotImplementedError()

def typeCompileAst(ast):
    raise NotImplementedError()

def stringToExpression(s, types, context):
    """Given a string, convert it to a tree of ExpressionNode's.
    """
    old_ctx = expressions._context.get_current_context()
    try:
        expressions._context.set_new_context(context)
        # first compile to a code object to determine the names
        if context.get('truediv', False):
            flags = __future__.division.compiler_flag
        else:
            flags = 0
        c = compile(s, '<expr>', 'eval', flags)
        # make VariableNode's for the names
        names = {}
        for name in c.co_names:
            if name == "None":
                names[name] = None
            elif name == "True":
                names[name] = True
            elif name == "False":
                names[name] = False
            else:
                t = types.get(name, default_type)
                names[name] = expressions.VariableNode(name, type_to_kind[t])
        names.update(expressions.functions)
        # now build the expression
        ex = eval(c, names)
        if expressions.isConstant(ex):
            ex = expressions.ConstantNode(ex, expressions.getKind(ex))
        elif not isinstance(ex, expressions.ExpressionNode):
            raise TypeError("unsupported expression type: %s" % type(ex))
    finally:
        expressions._context.set_new_context(old_ctx)
    return ex


def get_argnames(ex):
    return sorted({a.value for a in ex.allOf('variable')})
    

context_info = [
    ('optimization', ('none', 'moderate', 'aggressive'), 'aggressive'),
    ('truediv', (False, True, 'auto'), 'auto')
               ]

def getContext(kwargs, frame_depth=1):
    d = kwargs.copy()
    context = {}
    for name, allowed, default in context_info:
        value = d.pop(name, default)
        if value in allowed:
            context[name] = value
        else:
            raise ValueError("'%s' must be one of %s" % (name, allowed))

    if d:
        raise ValueError("Unknown keyword argument '%s'" % d.popitem()[0])
    if context['truediv'] == 'auto':
        caller_globals = sys._getframe(frame_depth + 1).f_globals
        context['truediv'] = \
            caller_globals.get('division', None) == __future__.division

    return context

class ArraySubscripter(ast.NodeTransformer):
    def visit_Name(self, node):
        if node.id not in py_funcs and node.id not in ('imag', 'real', 'abs', 'double'):
            # print node.id
            new_node = ast.Subscript(node,
                                     ast.Index(ast.Name('i', ast.Load())),
                                     ast.Load())
            return ast.copy_location(new_node, node)
        return node

class TemplateFiller(ast.NodeTransformer):
    def __init__(self, expr, argnames):
        self.expr = expr
        self.argnames = argnames

    def visit_Name(self, node):
        if node.id == '__expr_placeholder__':
            return ast.copy_location(self.expr, node)
        else:
            return node
    
    def visit_arguments(self, node):
        assert node.args[0].id == '__args_placeholder__'
        argnames = ['__result__'] + self.argnames
        node.args = [ast.Name(name, ast.Param()) for name in argnames]
        return node

savethread = pythonapi.PyEval_SaveThread
savethread.argtypes = []
savethread.restype = c_void_p

restorethread = pythonapi.PyEval_RestoreThread
restorethread.argtypes = [c_void_p]
restorethread.restype = None

def template_func(__args_placeholder__):
    _threadstate = savethread()
    for i in range(len(__result__)):
        __result__[i] = __expr_placeholder__
    restorethread(_threadstate)
        
template_ast = meta.decompiler.decompile_func(template_func)

from copy import deepcopy

def ast_expr_to_ast_func(ast_expr, arg_names):
    # subscripted_expr = ArraySubscripter().visit(ast_expr.body[0].value)
    subscripted_expr = ArraySubscripter().visit(ast_expr)
    # print ast.dump(subscripted_expr, annotate_fields=False)
    template_filler = TemplateFiller(subscripted_expr, arg_names)
    ast_func = template_filler.visit(deepcopy(template_ast))
    ast_func.name = '__expr_func__'
    # print ast.dump(ast_func, annotate_fields=False)
    ast_module = ast.Module([ast_func])
    return ast.fix_missing_locations(ast_module)
 
import math

py_funcs = {
     'savethread': savethread,
     'restorethread': restorethread,

#    'abs': abs,
#    'absolute': abs,
    'complex': complex,

    'sqrt': math.sqrt,

    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'arcsin': math.asin,
    'arccos': math.acos,
    'arctan': math.atan,

    'sinh': math.sinh,
    'cosh': math.cosh,
    'tanh': math.tanh,
    'arcsinh': math.asinh,
    'arccosh': math.acosh,
    'arctanh': math.atanh,

    'fmod': math.fmod,
    'arctan2': math.atan2,

    'log': math.log,
    'log1p': math.log1p,
    'log10': math.log10,
    'exp': math.exp,
    'expm1': math.expm1,

    'copy': np.copy,
    'ones_like': np.ones_like,
    'double': f8,
}
    

def ast_func_to_func(ast_func):
    code = compile(ast_func, '<expr>', 'exec')
    context = {'np': np}
    context.update(py_funcs)
    exec code in context
    return context['__expr_func__']


def precompile(ex, signature=(), context={}):
    """Compile the expression to an intermediate form.
    """
    if isinstance(ex, (str, unicode)):
        #XXX: we might want to work directly with (python's) AST
        # and do numexpr transformations directly at that level instead of going
        # str -> Expression -> ast -> ...
        types = dict(signature)
        ex = stringToExpression(ex, types, context)

    if signature:
        argnames = [name for (name, type_) in signature]
    else:
        # this can only occur when NumExpr() is called directly without
        # signature, and in that case we have no other choice than use
        # the default type for all arguments (double)
        argnames = get_argnames(ex)
        signature = [(name, double) for name in argnames]

    if ex.value in ('sum', 'prod'):
        reduction_func = getattr(np, ex.value)
        args = ex.children
        # axis is second arg
        assert len(args) == 2
        ex, axis = args
        axis = axis.value
    else:
        reduction_func = None
        axis = None

    ast_expr = ex.toPython()
    # print ast.dump(ast_expr, annotate_fields=False)
    ast_func = ast_expr_to_ast_func(ast_expr, argnames)
    inner_func = ast_func_to_func(ast_func)
    # print ast.dump(ast_func, annotate_fields=False)
    res_type = kind_to_type[ex.astKind]
    full_sig = [('__result__', res_type)] + signature
    arg_types = [type_to_numba[type_] for name, type_ in full_sig]
    jit_signature = void(*[t[:] for t in arg_types])
    inner_func_nb = jit(jit_signature, nopython=True)(inner_func)

    if reduction_func is not None:
        # this is a hack. To do it (more) correctly (with multithreading),
        # I would need to use a different template_func:

        # for i in range(len(__result__)):
            # __result__[0] += __expr_placeholder__
          
        def func(*args, **kwargs):
            # order, casting, ex_uses_vml
            out = kwargs.pop('out', None)
            if out is not None:
                raise NotImplementedError()

            shape = args[0].shape
            args = [a.ravel() for a in args]
            tmp_out = np.empty(shape, dtype=res_type)
            inner_func_nb(tmp_out.ravel(), *args)
            return reduction_func(tmp_out, axis=axis)
    else:
        def func(*args, **kwargs):
            numthreads = utils.num_threads
            # only scalars
            if any(isinstance(arg, np.ndarray) for arg in args):
                shape = args[0].shape
                if any(arg.shape != shape for arg in args[1:]):
                    args = np.broadcast_arrays(*args) 
                    shape = args[0].shape
            
                out = kwargs.pop('out', None)
                if out is None:
                    out = np.empty(shape, dtype=res_type)

                # "flatten" arguments

                # we cannot use order="K" which is most efficient, in case arguments
                # have not the same in-memory layout, because we need the same
                # target memory layout for all arguments.
                #XXX: can't we test for that and convert only if really necessary?
                args = [out.ravel()] + [a.ravel() for a in args]
                length = len(args[0])
                # TODO: it might be better to make sure the starting bounds
                #       are aligned to X bytes
                # TODO: it might be better to not multithread at all if
                #       length < THRESHOLD
                chunklen = (length + numthreads - 1) // numthreads
                bounds = [(i * chunklen, min((i + 1) * chunklen, length))
                          for i in range(numthreads)]
                assert bounds[-1][1] == length
                chunks = [[arg[start:stop] for arg in args]
                           for start, stop in bounds]
                threads = [threading.Thread(target=inner_func_nb, args=chunk)
                           for chunk in chunks[:-1]]
                for thread in threads:
                    thread.start()

                # the main thread handles the last chunk
                inner_func_nb(*chunks[-1])

                for thread in threads:
                    thread.join()
                return out
            else:
                # all arguments are scalar
                out = np.empty(1, dtype=res_type)
                args = [out] + [np.array([a]) for a in args]
                inner_func_nb(*args)
                return out[0]

    def run(*args, **kwargs):
        return func(*args, **kwargs)
    func.run = run
    return func
    


def NumExpr(ex, signature=(), copy_args=(), **kwargs):
    """
    Compile an expression built using E.<variable> variables to a function.

    ex can also be specified as a string "2*a+3*b".

    The order of the input variables and their types can be specified using the
    signature parameter, which is a list of (name, type) pairs.

    Returns a `NumExpr` object containing the compiled function.
    """
    # NumExpr can be called either directly by the end-user, in which case
    # kwargs need to be sanitized by getContext, or by evaluate,
    # in which case kwargs are in already sanitized.
    # In that case frame_depth is wrong (it should be 2) but it doesn't matter
    # since it will not be used (because truediv='auto' has already been
    # translated to either True or False).

    # NOTE: `copy_args` is not necessary from 2.0 on.  It remains here
    # basically because PyTables trusted on it for certain operations.
    # I have filed a ticket for PyTables asking for its removal:
    # https://github.com/PyTables/PyTables/issues/117

    context = getContext(kwargs, frame_depth=1)
    return precompile(ex, signature, context)


def disassemble(nex):
    """
    Given a NumExpr object, return a list which is the program disassembled.
    """
    raise NotImplementedError()


def getType(a):
    kind = a.dtype.kind
    if kind == 'b':
        return bool
    if kind in 'iu':
        if a.dtype.itemsize > 4:
            return long_  # ``long`` is for integers of more than 32 bits
        if kind == 'u' and a.dtype.itemsize == 4:
            return long_  # use ``long`` here as an ``int`` is not enough
        return int_
    if kind == 'f':
        if a.dtype.itemsize > 4:
            return double  # ``double`` is for floats of more than 32 bits
        return float
    if kind == 'c':
        return complex
    if kind == 'S':
        return bytes
    raise ValueError("unkown type %s" % a.dtype.name)


#XXX: no tan? log10? fmod?
#XXX: what is "inv"?
vml_funcs = set([
    'sqrt',
    'sin', 'cos',
    'arcsin', 'arccos', 'arctan',
    'sinh', 'cosh', 'tanh',
    'arcsinh', 'arccosh', 'arctanh',
    'arctan2',
    'log', 'log1p',
    'exp', 'expm1',
    'abs',
    'pow', 'div',
    'inv',
])
    
def getExprNames(text, context):
    ex = stringToExpression(text, {}, context)
    argnames = get_argnames(ex)
    if not use_vml:
        ex_uses_vml = False
        return argnames, ex_uses_vml
    else:
        # try to figure out if vml operations are used by expression
        ex_uses_vml = any(node.value in vml_funcs for node in ex.allOf('op'))
        return argnames, ex_uses_vml


# Dictionaries for caching variable names and compiled expressions
_names_cache = CacheDict(256)
_numexpr_cache = CacheDict(256)

def evaluate(ex, local_dict=None, global_dict=None,
             out=None, order='K', casting='safe', **kwargs):
    """Evaluate a simple array expression element-wise, using the new iterator.

    ex is a string forming an expression, like "2*a+3*b". The values for "a"
    and "b" will by default be taken from the calling function's frame
    (through use of sys._getframe()). Alternatively, they can be specifed
    using the 'local_dict' or 'global_dict' arguments.

    Parameters
    ----------

    local_dict : dictionary, optional
        A dictionary that replaces the local operands in current frame.

    global_dict : dictionary, optional
        A dictionary that replaces the global operands in current frame.

    out : NumPy array, optional
        An existing array where the outcome is going to be stored.  Care is
        required so that this array has the same shape and type than the
        actual outcome of the computation.  Useful for avoiding unnecessary
        new array allocations.

    order : {'C', 'F', 'A', or 'K'}, optional
        Controls the iteration order for operands. 'C' means C order, 'F'
        means Fortran order, 'A' means 'F' order if all the arrays are
        Fortran contiguous, 'C' order otherwise, and 'K' means as close to
        the order the array elements appear in memory as possible.  For
        efficient computations, typically 'K'eep order (the default) is
        desired.

    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur when making a copy or
        buffering.  Setting this to 'unsafe' is not recommended, as it can
        adversely affect accumulations.

          * 'no' means the data types should not be cast at all.
          * 'equiv' means only byte-order changes are allowed.
          * 'safe' means only casts which can preserve values are allowed.
          * 'same_kind' means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
          * 'unsafe' means any data conversions may be done.
    """
    if not isinstance(ex, (str, unicode)):
        raise ValueError("must specify expression as a string")
    # Get the names for this expression
    context = getContext(kwargs, frame_depth=1)
    expr_key = (ex, tuple(sorted(context.items())))
    if expr_key not in _names_cache:
        _names_cache[expr_key] = getExprNames(ex, context)
    names, ex_uses_vml = _names_cache[expr_key]
    # Get the arguments based on the names.
    call_frame = sys._getframe(1)
    if local_dict is None:
        local_dict = call_frame.f_locals
    if global_dict is None:
        global_dict = call_frame.f_globals

    arguments = []
    for name in names:
        try:
            a = local_dict[name]
        except KeyError:
            a = global_dict[name]
        arguments.append(np.asarray(a))

    # Create a signature
    signature = [(name, getType(arg)) for (name, arg) in zip(names, arguments)]

    # Look up numexpr if possible.
    numexpr_key = expr_key + (tuple(signature),)
    try:
        compiled_ex = _numexpr_cache[numexpr_key]
    except KeyError:
        compiled_ex = _numexpr_cache[numexpr_key] = \
                      NumExpr(ex, signature, **context)
    kwargs = {'out': out, 'order': order, 'casting': casting,
              'ex_uses_vml': ex_uses_vml}
    return compiled_ex(*arguments, **kwargs)
