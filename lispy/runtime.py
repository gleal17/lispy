import math
import operator as op
from collections import ChainMap
from types import MappingProxyType

from .symbol import Symbol

class Env(dict):
    "An environment: a dict of {'var':val} pairs, with an outer Env."
    def __init__(self, parms=(), args=(), outer=None):
        # Bind parm list to corresponding args, or single parm to list of args
        self.outer = outer
        if isinstance(parms, Symbol): 
            self.update({parms:list(args)})
        else: 
            if len(args) != len(parms):
                raise TypeError('expected %s, given %s, ' 
                                % (to_string(parms), to_string(args)))
            self.update(zip(parms,args))
    def find(self, var):
        "Find the innermost Env where var appears."
        if var in self: return self
        elif self.outer is None: raise LookupError(var)
        else: return self.outer.find(var)

def to_string(x):
    "Convert a Python object back into a Lisp-readable string."
    if x is True: return "#t"
    elif x is False: return "#f"
    elif isinstance(x, Symbol): return x
    elif isinstance(x, str): return '"%s"' % x.encode('string_escape').replace('"',r'\"')
    elif isinstance(x, list): return '('+' '.join(map(to_string, x))+')'
    elif isinstance(x, complex): return str(x).replace('j', 'i')
    else: return str(x)
class Procedure(object):
    "A user-defined Scheme procedure."
    def __init__(self, parms, exp, env):
        self.parms, self.exp, self.env = parms, exp, env
    def __call__(self, *args): 
        return eval(self.exp, Env(self.parms, args, self.env))




def eval(x, env=None):
    """
    Avalia expressão no ambiente de execução dado.
    """
    Number = (int, float) 
    #bool , str
    # Cria ambiente padrão, caso o usuário não passe o argumento opcional "env"
    if env is None:
        env = ChainMap({}, global_env)
    
     # Avalia tipos atômicos
    if isinstance(x, Symbol):
        return env[x]
    elif isinstance(x, (Number)):
        #int, float, bool, str
        return x

    # Avalia formas especiais e listas
    head, *args = x
    
    # Comando (if <test> <then> <other>)
    # Ex: (if (even? x) (quotient x 2) x)
    if head == Symbol.IF:
        (test,conseq, alt) = args
        return eval(conseq if eval(test, env) else alt)

    # Comando (define <symbol> <expression>)
    # Ex: (define x (+ 40 2))
    elif head == Symbol.DEFINE:
        (_, var, exp) = x
        env[var] = eval(exp, env)
        return None

    # Comando (quote <expression>)
    # (quote (1 2 3))
    elif head == Symbol.QUOTE:
        (_, exp) = x
        return exp

    # Comando (let <expression> <expression>)
    # (let ((x 1) (y 2)) (+ x y))
    elif head == Symbol.LET:
        return NotImplemented

    # Comando (lambda <vars> <body>)
    # (lambda (x) (+ x 1))
    elif head == Symbol.LAMBDA:
        (_, vars, exp) = x
        return Procedure(vars, exp, env)

    # Lista/chamada de funções
    # (sqrt 4)
    else:
        arguments = None
        proc = eval(head,env)
        arguments = [eval(arg, env) for arg in args]
        return proc(*arguments)


#
# Cria ambiente de execução.
#
def env(*args, **kwargs):
    """
    Retorna um ambiente de execução que pode ser aproveitado pela função
    eval().
    Aceita um dicionário como argumento posicional opcional. Argumentos nomeados
    são salvos como atribuições de variáveis.
    Ambiente padrão
    >>> env()
    {...}
        
    Acrescenta algumas variáveis explicitamente
    >>> env(x=1, y=2)
    {x: 1, y: 2, ...}
        
    Passa um dicionário com variáveis adicionais
    >>> d = {Symbol('x'): 1, Symbol('y'): 2}
    >>> env(d)
    {x: 1, y: 2, ...}
    """

    kwargs = {Symbol(k): v for k, v in kwargs.items()}
    if len(args) > 1:
        raise TypeError('accepts zero or one positional arguments')
    elif len(args):
        if any(not isinstance(x, Symbol) for x in args[0]):
            raise ValueError('keys in a environment must be Symbols')
        args[0].update(kwargs)
        return ChainMap(args[0], global_env)
    return ChainMap(kwargs, global_env)


def _make_global_env():
    """
    Retorna dicionário fechado para escrita relacionando o nome das variáveis aos
    respectivos valores.
    """

    dic = {
        **vars(math), # sin, cos, sqrt, pi, ...
        '+':op.add, '-':op.sub, '*':op.mul, '/':op.truediv, 
        '>':op.gt, '<':op.lt, '>=':op.ge, '<=':op.le, '=':op.eq, 
        'abs':     abs,
        'append':  op.add,  
        'apply':   lambda proc, args: proc(*args),
        'begin':   lambda *x: x[-1],
        'boolean?': lambda x: isinstance(x, bool),
        'car':     lambda x: head,
        'cdr':     lambda x: x[1:], 
        'cons':    lambda x,y: [x] + y,
        'eq?':     op.is_, 
        'expt':    pow,
        'equal?':  op.eq,
        'even?':   lambda x: x % 2 == 0,
        'length':  len, 
        'list':    lambda *x: list(x), 
        'list?':   lambda x: isinstance(x, list), 
        'map':     map,
        'max':     max,
        'min':     min,
        'not':     op.not_,
        'null?':   lambda x: x == [], 
        'number?': lambda x: isinstance(x, (float, int)),  
		'odd?':   lambda x: x % 2 == 1,
        'print':   print,
        'procedure?': callable,
        'quotient': op.floordiv,
        'round':   round,
        'symbol?': lambda x: isinstance(x, Symbol),
    }
    return MappingProxyType({Symbol(k): v for k, v in dic.items()})

global_env = _make_global_env() 