from lark import Lark, InlineTransformer
from pathlib import Path

from .runtime import Symbol
from .symbol import Symbol
import six

def psym(s, symbol_table={}):
        "Find or create unique Symbol entry for str s in symbol table."
        if s not in symbol_table:
            symbol_table[s] = Symbol(s)
        return symbol_table[s]
EOF_OBJECT = Symbol('#<eof-object>')  # Note: uninterned; can't be read

class LispTransformer(InlineTransformer):

    def start(self, expr): 
        return [Symbol.BEGIN, expr]

    def list(self, *expr):
        return list(expr)

    def atom(self, expr): 
        """
        Numeros em numeros; #t e #f são booleans; "..." string;
        outro Symbol.
        """
        if expr == '#t':
            return True
        elif expr == '#f':
            return False
        elif expr[0] == '"':
            if six.PY3:
                return expr[1:-1]
        try:
            return int(expr)
        except ValueError:
            try:
                return float(expr)
            except ValueError:
                try:
                    return complex(expr.replace('i', 'j', 1))
                except ValueError:
                    return psym(expr)
         
def parse(src: str):
        """
        Compila string de entrada e retorna a S-expression equivalente.
        """
        return parser.parse(src)



def _make_grammar():
    """
    Retorna uma gramática do Lark inicializada.
    """
    path = Path(__file__).parent / 'grammar.lark'
    with open(path) as fd:
        grammar = Lark(fd, parser='lalr', transformer=LispTransformer())
    return grammar

parser = _make_grammar()
