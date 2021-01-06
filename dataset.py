from naps.pipelines.read_naps import read_naps_dataset

from naps.uast import uast_pprint

import re

import numpy as np

def flatten(iterable):
    iterator, sentinel, stack = iter(iterable), object(), []
    while True:
        value = next(iterator, sentinel)
        if value is sentinel:
            if not stack:
                break
            iterator = stack.pop()
        elif isinstance(value, str):
            yield value
        else:
            try:
                new_iterator = iter(value)
            except TypeError:
                yield value
            else:
                stack.append(iterator)
                iterator = new_iterator


# trainA, trainB, test = read_naps_dataset()
# for name, ds in zip(("trainA", "trainB", "test"), read_naps_dataset()):
#     print("DATASET %s" % name)
def get_unique_tokens():
    ds, _, _ = read_naps_dataset()
    tokens_total = []
    with ds:
        
        for d in ds:
            if "is_partial" in d and d["is_partial"]:
                continue
            #print(' '.join(d["text"]))
            #uast_pprint.pprint(d["code_tree"])
            
            tokens = list(set(list(flatten(d["code_tree"]["funcs"]))))
            tokens_total += tokens

    tokens_total = list(set(tokens_total))
    print(tokens_total) 
    print(len(tokens_total)) #1771

    with open('tokens.txt', 'w') as f:
        for token in tokens_total:
            f.write("%s\n" % token)

def get_tokens():
    with open('tokens.txt', 'r') as f:
        tokens = f.read().splitlines()
        return tokens

def generate_output(tokens, unique):
    values = np.zeros((len(tokens), len(unique)))
    for i in range(len(tokens)):
        index = unique.index(str(tokens[i]))
        values[i][index] = 1
    return values



if __name__ == "__main__":
    #get_unique_tokens()
    ds, _, _ = read_naps_dataset()
    with ds:
        
        for d in ds:
            if "is_partial" in d and d["is_partial"]:
                continue
            print(' '.join(d["text"]))
            #uast_pprint.pprint(d["code_tree"])
            print(d["code_tree"]["funcs"])
            break
    #print(generate_output(['a','b'],['a','b','c','d','e','f']))