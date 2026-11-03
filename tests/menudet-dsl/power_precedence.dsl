# me:compiler=tcc
# Python precedence: -x ** 2 is -(x ** 2), ** is right-associative.
def power_precedence(x, y):
    return -x ** 2 + 2.0 ** y ** 2.0 - 2.0 ** -y
