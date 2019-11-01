#!/usr/bin/env python
# coding: utf-8

# # Lispy
# 
# ![Lisp Cycles](https://imgs.xkcd.com/comics/lisp_cycles.png)
# 
# Neste exercício, seguiremos o tutorial do [Peter Norvig](https://en.wikipedia.org/wiki/Peter_Norvig), [Como escrever um interpretador LISP em Python](http://norvig.com/lispy.html). O objetivo é criar um interpretador simples de um sub-conjunto da linguagem Scheme que rode em Python. Nosso interpretador não será completo e terá uma performance sofrível. Mas, ainda assim, vale a pena a jornada para entendermos melhor como que um interpretador funciona.
# 
# Esse guia acompanha o tutorial do Peter Norvig e você deve lê-lo para entender os próximos passos. Um pequeno spoiler, caso você não conheça LISP: vamos implementar uma linguagem que entende códigos indecifráveis e infestados de parênteses como este:
# 
# ```scheme
# (define fib (lambda (n) (if (< n 2) 
#     1 
#     (+ (fib (- n 1)) (fib (- n 2))))))
# ```
# 
# Agora vou me despedir temporariamente: leia até a seção "What A Language Interpreter Does" e depois volte para este guia tutorial

# ## Exercício 1 - Entendendo o parser (1 pt)
# 
# Agora que você está começando a entender o funcionamento do LISP e como as funções eval e parse devem funcionar, escreva o resultado esperado da análise (*parsing*) do programa abaixo na variável ``ast``:
# 
# ```scheme
# (lambda (n) (if (< n 1) 
#         1 
#         (* n (fat (- n 1))))) 
# ```
# 
# **Dica:** você não precisa implementar a função parse ainda. Basta escrever o resultado esperado na variável. O resultado consiste em uma lista de listas onde os símbolos como "lambda", "n", "<", etc são representados por strings.

# In[1]:


# ast = abstract syntax tree (árvore sintática abstrata, do inglês)

ast = [ 
    ### BEGIN SOLUTION
    'lambda', ['n'], ['if', ['<', 'n', 1], 
             1, 
             ['*', 'n', ['fat', ['-', 'n', 1]]]]
    ### END SOLUTION
]

print(ast)


# Este notebook conseguie corrigir este resultado automaticamente. Para isto, é necessário importar o módulo "maestro" como abaixo e depois rodar a célula com os "testes para questão 1"

# In[3]:


"testes para questão 1"

from hashlib import md5
find_secret = lambda x: md5(repr(x).encode('utf8')).hexdigest()

print('Hash da AST:', find_secret(ast))
assert find_secret(ast) == 'fa82facf358cb9e0a22f264e5f3715f3'


# ## Exercício 2 - Analisador léxico (1 pt)
# 
# Agora que você sabe que o parser deve fazer, vamos começar a implementá-lo. Leia as seções "Type Definitions" e "Tokenize". Vamos começar implementando o tokenizador (ou analisador léxico).

# In[5]:


def tokenize(chars: str) -> list:
    """
    Converte uma string de caracteres em uma lista de tokens.
    """
    ### BEGIN SOLUTION
    return chars.replace('(', ' ( ').replace(')', ' ) ').split()
    ### END SOLUTION


# In[7]:
# Teste seu tokenizador aqui e verifique o resultado executando a próxima célula!

program1 = '(begin (define r 10) (* pi (* r r)))'
program2 = '(+ (* 4 10) 2)'
print('Programa 1:', tokenize(program1))
print('Programa 2:', tokenize(program2))


# In[9]:
"tokeniza corretamente"
assert tokenize('(+ (* 4 10) 2)') == ['(', '+', '(', '*', '4', '10', ')', '2', ')']
### BEGIN HIDDEN TESTS
assert tokenize('(begin (define r 10) (* pi (* r r)))') == [
    '(', 'begin', '(', 'define', 'r', '10', ')', '(', '*', 'pi', '(', '*', 'r', 'r', ')', ')', ')']
### END HIDDEN TESTS


# ## Exercício 3 - Análise sintática (3 pts)
# 
# O próximo passo consiste na análise sintática que converte uma lista de tokens em uma árvore sintática. Quando compomos os dois passos de 1) tokenizar a string de entrada e 2) converter a lista de tokens em uma árvore sintática, podemos criar uma função que realiza a análise sintática completa.
# 
# O primeiro passo é entender os tipos de saída das nossas árvores sintáticas e como eles serão representados em Python. Isto é feito na célula abaixo:

# In[10]:
Symbol = str              # Um símbolo Scheme, implementado como uma string Python
Number = (int, float)     # Um símbolo Scheme, implementado como int ou float
Atom   = (Symbol, Number) # Um átomo pode ser um símbolo ou um número
List   = list             # Uma lista Scheme representada como uma lista Python
Exp    = (Atom, List)     # Expressão Scheme, que pode ser um átomo ou uma lista


# Depois, criamos a função `read_from_tokens()` que cria uma árvore sintática a partir de uma sequência de tokens e depois montamos a função `parse()` juntando `tokenize()` com `read_from_tokens()`:

# In[11]:
def parse(program: str) -> Exp:
    """
    Lê uma expressão Scheme de uma string e retorna a árvore sintática correspondente.
    """
    return read_from_tokens(tokenize(program))


def read_from_tokens(tokens: list) -> Exp:
    """
    Monta árvore sintática a partir de uma lista de tokens.
    """
    ### BEGIN SOLUTION
    if len(tokens) == 0:
        raise SyntaxError('unexpected EOF')
    token = tokens.pop(0)
    if token == '(':
        L = []
        while tokens[0] != ')':
            L.append(read_from_tokens(tokens))
        tokens.pop(0) # pop off ')'
        return L
    elif token == ')':
        raise SyntaxError('unexpected )')
    else:
        try: 
            return int(token)
        except ValueError:
            try: 
                return float(token)
            except ValueError:
                return Symbol(token)
    ### END SOLUTION


# In[12]:
"testa parser"

program = "(begin (define r 10) (* pi (* r r)))"
assert parse(program) ==  ['begin', ['define', 'r', 10], ['*', 'pi', ['*', 'r', 'r']]]
### BEGIN HIDDEN TESTS
program = "(begin (+ (fib 5) (fat 5)))"
assert parse(program) == ['begin', ['+', ['fib', 5], ['fat', 5]]]
### END HIDDEN TESTS


# ## Exercício 4 - Ambiente de execução (2 pts)
# 
# Vamos agora definir um ambiente de execução padrão. Leia a seção "Enviroment" e depois volte para cá.
# 
# Peter Norvig propôs utilizar um dicionário que guarda as variáveis do escopo atual de execução com seus respectivos valores. Implemente o dicionário global_env que guarda estas implementações para as funções abaixo:
# 
# - **abs, sin, cos, sqrt, pi, ...:** funções matemáticas
# - **+, -, *, /, >, <, >=, <=, =:** operadores matemáticos
# - **append:** junta duas listas
# - **apply**: aplica lista de arguments em função (apply f args)
# - **begin**: avalia sequência de comandos e retora o último
# - **car**: primeiro elemento da lista
# - **cdr**: resto da lista (pula 1o elemento)
# - **cons**: construtor; (car x lst) retorna lista que adiciona x ao início de lst
# - **eq?**: testa se dois argumentos são idênticos 
# - **equal?**: testa se dois argumentos são iguais
# - **length**: retorna tamanho de uma lista 
# - **list**: cria uma lista a partir dos argumentos
# - **list?**: verifica se argumento é uma lista 
# - **map**: (map f lst) aplica função em cada elemento de uma lista e retorna lista com resultado
# - **max**: maior entre dois argumentos
# - **min**: menor entre dois argumentos
# - **not**: negação booleana
# - **null?**: verifica se o argumento é uma lista vazia
# - **number?**: verifica se o argumento é um número   
# - **procedure?**: verifica se o argumento é uma função
# - **round**: arredonda valor float para inteiro
# - **symbol?**: verifica se valor é um símbolo

# In[13]:


import math
import operator as op


def standard_env():
    """
    Retorna ambiente de execução (dicionário) que mapeia os nomes com
    as implementações das principais funções do Scheme.
    """
    env = {}
    env.update(vars(math)) # sin, cos, sqrt, pi, ...
    env.update({
        '+':op.add, '-':op.sub, '*':op.mul, '/':op.truediv, 
        ### BEGIN SOLUTION
        '>':op.gt, '<':op.lt, '>=':op.ge, '<=':op.le, '=':op.eq, 
        'abs':     abs,
        'append':  op.add,  
        'apply':   lambda proc, args: proc(*args),
        'begin':   lambda *x: x[-1],
        'car':     lambda x: x[0],
        'cdr':     lambda x: x[1:], 
        'cons':    lambda x, y: [x, *y],
        'eq?':     op.is_, 
        'equal?':  op.eq, 
        'length':  len, 
        'list':    lambda *x: list(x), 
        'list?':   lambda x: isinstance(x,list), 
        'map':     lambda *args: list(map(*args)),
        'max':     max,
        'min':     min,
        'not':     op.not_,
        'null?':   lambda x: x == [], 
        'number?': lambda x: isinstance(x, Number),   
        'procedure?': callable,
        'round':   round,
        ### END SOLUTION
        'symbol?': lambda x: isinstance(x, Symbol),
    })
    return env
    
    
global_env = standard_env()


# In[14]:
"testa presença de funções global"

functions = (
    'abs, sin, cos, sqrt, pi, tan, log, +, -, *, /, >, <, >=, <=, =, append, apply, begin, car, cdr, '
    'cons, eq?, equal?, length, list, list?, map, max, min, not, null?, number?, procedure?, '
    'round, symbol?').split(', ')

assert all(f in global_env for f in functions)


# In[15]:
"testa implementações"

e = global_env
_sqrt, _cons, _append = e['sqrt'], e['cons'], e['append']
_car, _cdr = e['car'], e['cdr']
_list = e['list']

assert _sqrt(4) == 2
assert _cons(1, [2, 3]) == [1, 2, 3]
assert _append([1, 2], [3, 4]) == [1, 2, 3, 4]
assert _car([1, 2, 3]) == 1
assert _cdr([1, 2, 3]) == [2, 3]
assert _list(1, 2, 3) == [1, 2, 3]

### BEGIN HIDDEN TESTS
_map = e['map']
assert _map(_sqrt, [1, 4, 9, 16]) == [1, 2, 3, 4]
### END HIDDEN TESTS


# ## Exercício 5 - eval (3 pts)
# 
# O próximo passo é a parte mais importante de um interpretador. Vamos implementar a função `eval(ast)`, que recebe uma representação de um programa (no nosso caso a árvore sintática) e a executa. Leia a seção "Evaluation: eval" e implemente a função `eval(ast)`.
# In[16]:

def eval(x: Exp, env = None) -> Exp:
    """
    Avalia expressão no ambiente dado.
    """
    if env is None:
        env = global_env.copy()
        
    ### BEGIN SOLUTION
    if isinstance(x, Symbol):
        return env[x]

    elif isinstance(x, Number):
        return x
    
    # Expressões
    cmd = x[0]
    
    if cmd == 'if':
        (_, test, then, alt) = x
        exp = (then if eval(test, env) else alt)
        return eval(exp, env)
    
    elif cmd == 'define':
        (_, symbol, exp) = x
        env[symbol] = result = eval(exp, env)
        return result
        
    # Chamada de função
    else:
        proc = eval(cmd, env)
        args = (eval(arg, env) for arg in x[1:])
        return proc(*args)
    ### END SOLUTION


# In[17]:
# Teste seu código aqui

eval(parse(input('Digite uma expressão scheme [ex. (+ 1 2)]: ')))


# In[18]:
"checa funcao eval"

assert eval(parse('(+ 40 2)')) == 42
assert eval(parse('42')) == 42
assert eval(parse('sqrt'))(4) == 2
assert eval(parse('(sqrt 4)'))  == 2
assert eval(parse('(if (< 1 2) 42 0)')) == 42

ns = {}
assert eval(parse('(define x 42)'), ns) == 42
assert ns == {'x': 42}

### BEGIN HIDDEN TESTS
ns = global_env.copy()
assert eval(parse('(define x (if (< 1 2) 42 0))'), ns) == 42
assert ns['x'] == 42
### END HIDDEN TESTS


# ## Exercício 6 - repl (2 pts)
# 
# Toda linguagem interpretada que se preze possui um shell iterativo. Vamos implementar o nosso a partir das sugetões em "Interaction: A REPL".

# In[19]:


def repl(prompt='lis.py> '):
    """
    Executa o console no modo "read-eval-print loop"
    
    Deve interromper a execução se o usuário digitar "exit"
    """
    ### BEGIN SOLUTION
    ns = global_env.copy()
    
    while True:
        program = input(prompt).strip()
        if program == 'exit':
            break
        
        val = eval(parse(program), ns)
        if val is not None: 
            print(show_scheme(val))
    ### END SOLUTION


def show_scheme(exp):
    """
    Converte expressão para sua representação em Scheme.
    """
    if isinstance(exp, List):
        return '(' + ' '.join(map(schemestr, exp)) + ')' 
    else:
        return str(exp)


# In[21]:


# Teste o seu repl aqui

repl()


# In[22]:


# Função que testa iteração: não edite
# IGNORE ESTA CÉLULA ;)
from collections import deque
from functools import wraps
from contextlib import contextmanager
from types import SimpleNamespace as record
import builtins
import sys
import io

_input = input
_print = print

def check_interaction(*args, **kwargs):
    func, mapping, *args = args

    spec = deque()
    for k, v in mapping.items():
        spec.extend([k, ignore_ws, v])
        
    interaction = deque()
    
    @wraps(_input)
    def input_(msg=None):
        if msg is not None:
            print_(msg, end='')
        if not spec:
            raise AssertionError('Unexpected input')

        res = spec.popleft()
        if not _is_input(res):
            raise AssertionError('Expects output, but got an input command')

        # Extract input from singleton list or set
        try:
            value, = res
            interaction.append([value])
            return value
        except TypeError:
            raise ValueError('Expected input list must have a single value')

    @wraps(_print)
    def print_(*args_, file=None, **kwargs_):
        if file in (None, sys.stdout, sys.stderr):
            fd = io.StringIO()
            # noinspection PyTypeChecker
            _print(*args_, file=fd, **kwargs_)
            output = fd.getvalue()
            interaction.append(output)
            for line in output.splitlines():
                _consume_output(line, spec)
        else:
            _print(*args_, file=file, **kwargs_)

    with update_builtins(input=input_, print=print_):
        try:
            func(*args, **kwargs)
        except Exception as ex:
            print(interaction)
            raise

    assert spec == deque([]), f'Outputs/inputs were not exhausted: {spec}'
    
    
def _is_input(obj):
    return not isinstance(obj, str) and not callable(obj)


def _consume_output(printed, spec: deque):
    """
    Helper function: consume the given output from io spec.

    Raises AssertionErrors when it encounter problems.
    """

    if not printed:
        return
    elif not spec:
        raise AssertionError('Asking to consume output, but expects no interaction')
    elif _is_input(spec[0]):
        raise AssertionError('Expects input, but trying to print a value')
    elif printed == spec[0]:
        spec.popleft()
    elif callable(spec[0]):
        spec.popleft()(printed, spec)
    elif spec[0].startswith(printed):
        spec[0] = spec[0][len(printed):]
    elif printed.startswith(spec[0]):
        n = len(spec.popleft())
        _consume_output(printed[n:], spec)
    else:
        raise AssertionError(f'Printed wrong value:\n'
                             f'    print: {printed!r}\n'
                             f'    got:   {spec[0]!r}')
    
    
def ignore_ws(received, spec):
    """
    Consume all whitespace in the beginning of the spec.

    No-op if first element does not start with whitespace.
    """
    if spec and isinstance(spec[0], str):
        spec[0] = spec[0].popleft().lstrip()
        _consume_output(received, spec)
        

@contextmanager
def update_builtins(**kwargs):
    """
    Context manager that temporarily sets the specified builtins to the given
    values.

    Examples:
        >>> with update_builtins(print=lambda *args: None) as orig:
        ...     print('Hello!')          # print is shadowed here
        ...     orig.print('Hello!')  # Calls real print
    """

    undefined = object()
    revert = {k: getattr(builtins, k, undefined) for k in kwargs}
    try:
        for k, v in kwargs.items():
            setattr(builtins, k, v)
        yield record(**{k: v for k, v in revert.items() if v is not undefined})
    finally:
        for k, v in revert.items():
            if v is not undefined:
                setattr(builtins, k, v)


# In[23]:


"checa repl"

check_interaction(repl, {
    'lis.py>': {'(+ 1 2)'},
    'lis.py>': '3',
    'lis.py>': {'(list 1 2 3)'},
    'lis.py>': '(1 2 3)',
    'lis.py>': {'exit'},
})


# ## Exercício 7 - quote (1 pt)
# 
# Leia a seção "Language 2: Full Lispy", para entender o que ainda falta na nossa implementação. Neste exercício vamos implementar a forma especial de "quotation" (que em português podemos traduzir para "citação"). Esta é talvez tarefa mais simples, pois a função do "quotation" é simplesmente retornar o argumento sem fazer nada com ele.
# 
# A existência de uma função que faz isto é um pouco intrigante, mas na verdade está na raiz do poder do LISP como linguagem. Com o "quote", podemos criar facilmente estruturas de dados que representam programa, manipulá-las e eventualmente executá-las utilizando a função `eval()`. 
# 
# O fato do Scheme (e quase todas formas de LISP) exporem as entranhas do interpretador desta maneira a torna uma linguagem extremamente poderosa. A comunidade LISP muitas vezes chama estas técnicas de metaprogramação de "code as data" (código como dados) e são consideradas como aspectos essenciais do LISP que a diferenciam de outras linguagens.

# In[24]:


def eval(x: Exp, env = None) -> Exp:
    """
    Avalia expressão no ambiente dado.
    """
    if env is None:
        env = global_env.copy()
        
    ### BEGIN SOLUTION
    if isinstance(x, Symbol):
        return env[x]

    elif isinstance(x, Number):
        return x
    
    # Expressões
    cmd = x[0]
    
    if cmd == 'if':
        (_, test, then, alt) = x
        exp = (then if eval(test, env) else alt)
        return eval(exp, env)
    
    elif cmd == 'define':
        (_, symbol, exp) = x
        env[symbol] = result = eval(exp, env)
        return result
    
    elif cmd == 'quote':
        (_, expr) = x
        return expr
    
    # Chamada de função
    else:
        proc = eval(cmd, env)
        args = [eval(arg, env) for arg in x[1:]]
        return proc(*args)
    ### END SOLUTION


# In[98]:


# Teste aqui!

eval(parse('(quote (sin x))'))


# In[25]:


"implementa o comando quote"

assert eval(parse('(quote (sin x))')) == ['sin', 'x']
### BEGIN HIDDEN TESTS
assert eval(parse('(quote (quote (1 2 3)))')) == ['quote', [1, 2, 3]]
assert eval(parse('(quote 42)')) == 42
assert eval(parse('(quote sqrt)')) == 'sqrt'
### END HIDDEN TESTS


# ## Exercício 8 - lambda (3 pts)
# 
# Nosso interpretador está tomando forma! Agora vamos abordar um aspecto mais complicado que é a implementação de procedimentos. A grande dificuldade aqui é criar ambientes de execução aninhados, já que a função pode criar variáveis locais, mas também herda as variáveis definidas no escopo global.
# 
# Norvig sugere utilizar o ChainMap, que realmente é perfeito para a nossa situação. Leia a seção "Nested Environments" para entender o que ele propõe e implemente "lambdas" na função `eval()`.

# In[27]:


from collections import ChainMap


def eval(x: Exp, env = None) -> Exp:
    """
    Avalia expressão no ambiente dado.
    """
    if env is None:
        env = global_env.copy()
        
    ### BEGIN SOLUTION
    if isinstance(x, Symbol):
        return env[x]

    elif isinstance(x, Number):
        return x
    
    # Expressões
    cmd = x[0]
    
    if cmd == 'if':
        (_, test, then, alt) = x
        exp = (then if eval(test, env) else alt)
        return eval(exp, env)
    
    elif cmd == 'define':
        (_, symbol, exp) = x
        env[symbol] = result = eval(exp, env)
        return result
    
    elif cmd == 'quote':
        (_, expr) = x
        return expr
    
    elif cmd == 'lambda':
        (_, names, body) = x
        
        def proc(*args):
            local = dict(zip(names, args))
            return eval(body, ChainMap(local, env))
        
        return proc
    
    # Chamada de função
    else:
        proc = eval(cmd, env)
        args = [eval(arg, env) for arg in x[1:]]
        return proc(*args)
    ### END SOLUTION


# In[28]:


# Teste aqui!

inc = eval(parse('(lambda (x) (+ x 1))'))
inc(41)


# In[29]:


# Função incremento
assert eval(parse('(lambda (x) (+ x 1))'))(41) == 42

# Função fatorial
assert eval(parse('''
(begin
    (define fat (lambda (n)
        (if (< n 1)
            1
            (* n (fat (- n 1))))))
    
    (fat 5))''')) == 120
### BEGIN HIDDEN TESTS
_double = eval(parse('(lambda (x) (* 2 x))'))
assert all(_double(i) == 2 * i for i in range(50)), 'Função double implementada incorretamente'
### END HIDDEN TESTS


# ## Exercício 9 - Fibonacci em LISP (2pts)
# 
# Temos agora um interpretador completo o suficiente para implementar os clássicos da computação, como fibonacci, fatorial, etc. Vamos aproveitar!
# 
# O próximo exercício consiste em implementar uma função que calcula o n-ésimo termo da sequência de Fibonacci até o número n. Lembre-se que o Scheme não possui laços, então devemos implementar o Fibonacci de forma recursiva.

# In[30]:


cmd = '''
(define fib (lambda (n) 
    (* n n)))
'''
### BEGIN SOLUTION
cmd = '''
(begin
    (define fib (lambda (n) (fib-acc n 1 1)))
    
    (define fib-acc (lambda (n x y)
        (if (= n 0) 
            x 
            (fib-acc (- n 1) y (+ x y))))))
'''
### END SOLUTION


# In[31]:


# Teste sua função aqui!
ns = global_env.copy()
eval(parse(cmd), ns)
fib = ns['fib']

[fib(n) for n in range(10)]


# In[32]:


ns = global_env.copy()
eval(parse(cmd), ns)
fib = ns['fib']

assert [fib(n) for n in range(10)] == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
### BEGIN HIDDEN TESTS
_fib = lambda n, x=1, y=1: x if n == 0 else _fib(n - 1, y, x + y)
assert all(fib(n) == _fib(n) for n in range(100))
### END HIDDEN TESTS


# ## Questão 10 - extendendo as funções builtin (3pts)
# 
# Agora vamos acrescentar algumas funções adicionais no nosso ambiente. O ambiente sugerido pelo Dr. Norvig é bem minimalista e não implementa várias funções disponíveis por padrão no Scheme. Obviamente não vamos implementar todas estas funções, mas apenas acrescentar algumas. 
# 
# Escolhi as funções `even?/odd?/quotient/modulo/remainder` para este exercício. Queremos implementar estas funções com base no comportamento esperado para a linguagem. Para fazer isto, vamos ler a documentação do Scheme e entender exatamente o que deve ser implementado. Siga para [even?](https://docs.racket-lang.org/reference/number-types.html?q=even%3F#%28def._%28%28quote._~23~25kernel%29._even~3f%29%29) para começar os serviços.

# In[33]:


def standard_env_ext():
    env = standard_env()
    ### BEGIN SOLUTION
    env.update({
        'even?': lambda x: x % 2 == 0,
        'odd?': lambda x: x % 2 == 1,
        'quotient': lambda x, y: int(x / y),
        'remainder': lambda x, y: int(math.remainder(x, y)),
        'modulo': op.mod,
    })
    ### END SOLUTION
    return env

global_env = standard_env_ext()


# In[34]:


"even? e odd?"

_is_even = global_env['even?']
_is_odd = global_env['odd?']
assert _is_even(42) and not _is_even(21)
assert not _is_odd(42) and _is_odd(21)
### BEGIN HIDDEN TESTS
for i in range(10):
    assert _is_even(2 * i) and not _is_even(2 * i + 1)
### END HIDDEN TESTS


# In[35]:


"divisão inteira"

_quotient = global_env['quotient']
assert _quotient(10, 3) == 3
assert _quotient(-10.0, 3) == -3.0
assert _quotient(13, -5) == -2
assert _quotient(-13, -5) == 2
### BEGIN HIDDEN TESTS
assert _quotient(11, 4) == 2
assert _quotient(-11, 4) == -2
assert _quotient(11, -4) == -2
assert _quotient(-11, -4) == 2
### END HIDDEN TESTS


# In[36]:


"resto da divisão"

_modulo = global_env['modulo']
assert _modulo(10, 3) == 1
assert _modulo(-10.0, 3) == 2.0
assert _modulo(10.0, -3) == -2.0
assert _modulo(-10, -3) == -1

_remainder = global_env['remainder']
assert _remainder(10, 3) == 1
assert _remainder(-10.0, 3) == -1.0
assert _remainder(10.0, -3) == 1.0
assert _remainder(-10.0, -3) == -1.0
assert isinstance(_remainder(13, 4), int)

### BEGIN HIDDEN TESTS
assert _modulo(11, 4) == 3
assert _modulo(-11, 4) == 1
assert _remainder(13, 4) == 1
assert _remainder(-13, 4) == -1
### END HIDDEN TESTS


# ### Exercício 11 - Collatz em LISP (3pts)
# 
# 
# Agora que sabemos calcular restos da divisão e checar a paridade de números, vamos ao último desafio. Imlemente a sequência de [Collatz](https://www.youtube.com/watch?v=5mFpVDpKX70)
# 
# Você deve definir uma função `(collatz n)` que recebe um
# número n qualquer e retorna uma lista com a sequência de Collatz até convergir para 1 (inclusive).

# In[37]:


cmd = '''
(begin 
    (define collatz (lambda (n) 
        (cons 1 (list 2 3))))
    
    (define auxiliary-function (lambda (n) n)))
'''
### BEGIN SOLUTION
cmd = '''
(begin
    (define collatz (lambda (n) 
        (if (= n 1) 
            (list 1) 
            (cons n (collatz (collatz-next n))))))
    
    (define collatz-next (lambda (n) 
        (if (even? n)
            (quotient n 2)
            (+ (* 3 n) 1)))))
'''
### END SOLUTION


# In[38]:


# Teste sua função aqui! 
ns = standard_env_ext()
eval(parse(cmd), ns)
collatz = ns['collatz']

collatz(13)


# In[41]:


ns = standard_env_ext()
eval(parse(cmd), ns)
collatz = ns['collatz']

assert collatz(13) == [13, 40, 20, 10, 5, 16, 8, 4, 2, 1]
assert collatz(5) == [5, 16, 8, 4, 2, 1]

### BEGIN HIDDEN TESTS
assert collatz(7) == [7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1]
### END HIDDEN TESTS

