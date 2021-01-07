bad_tree = [['func', 'char**', '_main_', [['var', 'int', 'var0']], [['var', 'bool*', 'var1'], ['var', 'int', 'var2'], ['var', 'char*', 'var3'], ['var', 'int', 'var4'], ['var', 'int', 'var5'], ['var', 'int', 'var6'], ['var', 'char**', 'var7']], [['assign', 'char**', ['var', 'char**', 'var7'], ['invoke', 'char**', '_ctor', []]], ['assign', 'bool*', ['var', 'bool*', 'var1'], ['invoke', 'bool*', '_ctor', [['invoke', 'int', '+', [['var', 'int', 'var0'], ['val', 'int', 1]]]]]], ['assign', 'int', ['var', 'int', 'var2'], ['val', 'int', 0]], ['assign', 'char*', ['var', 'char*', 'var3'], ['val', 'char*', '']], ['assign', 'int', ['var', 'int', 'var4'], ['val', 'int', 2]], ['while', 'void', ['invoke', 'bool', '<=', [['var', 'int', 'var4'], ['var', 'int', 'var0']]], [['if', 'void', ['invoke', 'bool', 'array_index', [['var', 'bool*', 'var1'], ['var', 'int', 'var4']]], [['continue', 'void']], []], ['assign', 'int', ['var', 'int', 'var5'], ['invoke', 'int', '+', [['var', 'int', 'var4'], ['var', 'int', 'var4']]]], ['while', 'void', ['invoke', 'bool', '<=', [['var', 'int', 'var5'], ['var', 'int', 'var0']]], [['assign', 'bool', ['invoke', 'bool', 'array_index', [['var', 'bool*', 'var1'], ['var', 'int', 'var5']]], ['val', 'bool', True]]], [['assign', 'int', ['var', 'int', 'var5'], ['invoke', 'int', '+', [['var', 'int', 'var5'], ['var', 'int', 'var4']]]]]], ['assign', 'int', ['var', 'int', 'var6'], ['var', 'int', 'var4']], ['while', 'void', ['invoke', 'bool', '<=', [['var', 'int', 'var6'], ['var', 'int', 'var0']]], [['assign', 'int', ['var', 'int', 'var2'], ['invoke', 'int', '+', [['var', 'int', 'var2'], ['val', 'int', 1]]]], ['assign', 'char*', ['var', 'char*', 'var3'], ['invoke', 'char*', 'concat', [['var', 'char*', 'var3'], ['invoke', 'char*', 'concat', [['invoke', 'char*', 'str', [['var', 'int', 'var6']]], ['val', 'char*', ' ']]]]]]], [['assign', 'int', ['var', 'int', 'var6'], ['invoke', 'int', '', [['var', 'int', 'var6'], ['var', 'int', 'var4']]]]]]], [['assign', 'int', ['var', 'int', 'var4'], ['invoke', 'int', '+', [['var', 'int', 'var4'], ['val', 'int', 1]]]]]], ['invoke', 'void', 'array_push', [['var', 'char*', 'var7'], ['invoke', 'char*', 'str', [['var', 'int', 'var2']]]]], ['assign', 'char**', ['var', 'char**', 'var7'], ['invoke', 'char**', 'array_concat', [['var', 'char**', 'var7'], ['invoke', 'char**', 'string_split', [['var', 'char*', 'var3'], ['val', 'char*', ' \\t']]]]]], ['return', 'void', ['var', 'char**', 'var7']]]]]
# print(bad_tree)
# print("\n")
import numpy as np

def make_the_tree_good(bad_tree):
    # Zwracana zmienna
    good_tree = []

    # Warunek zakończenia rekurencji
    if type(bad_tree) is not list:
        good_tree = bad_tree

    # Warunek zaistnienia rekurencji
    else:
        for x in bad_tree:
            mttg = make_the_tree_good(x)

            # Dobranie sposobu dołączenia do wynikowej listy
            if type(mttg) is list:
                good_tree.extend(mttg)
            else:
                good_tree.append(str(mttg))

    return good_tree


# print(make_the_tree_good(bad_tree))
# print(type(str([1,2,3])))
