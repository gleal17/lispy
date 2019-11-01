from lark import Lark, InlineTransformer
from pathlib import Path

from .runtime import Symbol
from .symbol import Symbol
import six

class LispTransformer(InlineTransformer):
    def start(self, expr): 
        return [Symbol.BEGIN, expr]

    def list(self,expr):
        return list(expr)

    def Sym(self, symbol_table={}):
        "Find or create unique Symbol entry for str s in symbol table."
        if self not in symbol_table:
            symbol_table[self] = Symbol(self)
        return symbol_table[self]
    EOF_OBJECT = Symbol('#<eof-object>')  # Note: uninterned; can't be read
    
    def atom(self, token):
        """
        Numeros em numeros; #t e #f são booleans; "..." string;
        outro Symbol.
        """
        if token == '#t':
            return True
        elif token == '#f':
            return False
        elif token[0] == '"':
            if six.PY3:
                return token[1:-1]
            else:
                return token[1:-1].decode('string_escape')
        try:
            return int(token)
        except ValueError:
            try:
                return float(token)
            except ValueError:
                try:
                    return complex(token.replace('i', 'j', 1))
                except ValueError:
                    return Sym(token)

def parse(src: str):
        """
        Compila string de entrada e retorna a S-expression equivalente.
        """
        if src is True:
            return "#t"
        elif src is False:
            return "#f"
        elif isinstance(src, Symbol):
            return src
        elif isinstance(src, str):
            return '"%s"' % src.encode('string_escape').replace('"', r'\"')
        elif isinstance(src, list):
            return '('+' '.join(list(map(parse, src)))+')'
        elif isinstance(src, complex):
            return str(src).replace('j', 'i')
        else:
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
