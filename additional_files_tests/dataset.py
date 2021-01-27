from naps.pipelines.read_naps import read_naps_dataset
from tree_transformation import make_the_tree_good

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
            yield str(value)
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
            
            tokens = list(set(list(make_the_tree_good(d["code_tree"]["funcs"]))))
            tokens_total += tokens

    tokens_total = list(set(tokens_total))
    print(tokens_total) 
    print(len(tokens_total)) #1762

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

def generate_tokens(output, unique):
    output = np.array(output)
    tokens = []
    for out in output:
        idx = np.argmax(out)
        tokens.append(unique[idx])
    return tokens

def even_embeddings(embed, n):
    if len(embed) < n:
        diff =  n - len(embed)
        embed = np.vstack([embed, np.zeros((diff,len(embed[0])))])
    return embed

def even_tokens(tokens, n):
    if len(tokens) < n:
        diff =  n - len(tokens)
        tokens = np.concatenate((tokens, np.full(diff, '')))
    return tokens



if __name__ == "__main__":
    #get_unique_tokens()
    # ds, _, _ = read_naps_dataset()
    # with ds:
        
    #     for d in ds:
    #         if "is_partial" in d and d["is_partial"]:
    #             continue
    #         print(' '.join(d["text"]))
    #         #uast_pprint.pprint(d["code_tree"])
    #         print(d["code_tree"]["funcs"])
    #         break
    # out = generate_output(['a','b'],['a','b','c','d','e','f'])
    # print(out)
    # print(generate_tokens(out, ['a','b','c','d','e','f']))
    # embed = np.array([[1,2,3],[1,2,3],[1,2,3],[1,2,3]])
    # embed = even_embeddings(embed, 8)
    # print(embed)
    print(even_tokens(['a','b','c','d','e','f'], 12))
    