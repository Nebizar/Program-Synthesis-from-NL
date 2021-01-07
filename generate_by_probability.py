from naps.pipelines.read_naps import read_naps_dataset
from tree_transformation import make_the_tree_good


def get_nonflatten_tokens():
    ds, _, _ = read_naps_dataset()
    tokens_total = []
    with ds:

        for d in ds:
            if "is_partial" in d and d["is_partial"]:
                continue
            # print(' '.join(d["text"]))
            # uast_pprint.pprint(d["code_tree"])

            tokens = list(d["code_tree"]["funcs"])
            tokens_total += tokens

    tokens_total = tokens_total
    # print(tokens_total)
    # print(len(tokens_total))  # 20.000 z grubsza

    '''with open('nonflatten_tokens.txt', 'w') as f:
        for token in tokens_total:
            f.write("%s\n" % token)'''

    return tokens_total


if __name__ == "__main__":

    print("Witamy w pliku testowym!")
    '''connections = {}
    connections["alfa"] = 4
    connections["beta"] = 5
    for x in connections:
        if x == "alfa":
            print(connections[x])'''


    a = ["a", "b", "v", "e", "r"]
    b = get_nonflatten_tokens()

    #data_size = 1000
    data = make_the_tree_good(b)  #[0:data_size]  # Przycięcie danych do mniejszego rozmiaru, ROZWAŻNIE Z TYM CIĘCIEM SIĘ OBCHODZIĆ
    print("Dane:")
    print(data)
    print(len(data))
    print("\n")



    connections = {}
    connection_length = 2  # że w sumie 2 wyrazy, czyli patrzymy na jedno wstecz

    for x in range(len(data) - connection_length + 1):

        # Dolicz wystąpienie znalezionego {prefixu}{suffixu} - suffix = ostatnia komenda
        prefix = "|".join(data[x: x + connection_length - 1])  # Musi być symbol, który nie istnieje w komendach
        suffix = data[x + connection_length - 1]

        if prefix in connections:
            if suffix in connections[prefix]:
                connections[prefix][suffix]+=1
            else:
                connections[prefix][suffix]=1
        else:
            connections[prefix] = {}
            connections[prefix][suffix] = 1

    # Zmiana liczebności na szansę danego suffixu na tle innych suffixów dla danego prefixu
    for prefix in connections:   # dla danego prefixu
        how_many = 0
        for suffix in connections[prefix]:
            how_many+=connections[prefix][suffix]

        for suffix in connections[prefix]:
            connections[prefix][suffix] = connections[prefix][suffix] / how_many

    print("Połączenia:")
    print(connections)


    # WYKORZYSTANIE
    # 1. Znając "aktualną" komendę i komendy ja poprzedzające, zbijamy komendy poprzedzające w prefix: "|".join(lista_komend)
    # *  W przypadku sprawdzania na jedną komendę wstecz wykorzystanie ".join" nie jest konieczne
    # 2. Suffix to nasza uzyskana komenda
    # 3. Prawdopodobieństwo uzyskujemy wywołując: connections[prefix][suffix]
    # 4. Strata (propozycja): <1 - uzyskana_szansa>   <- strata mniejsza dla bardziej prawidłowego tekstu?


    # PRZYKŁAD WYKORZYSTANIA:
    prefix = "var"
    suffix = "bool*"

    loss = 1
    if prefix in connections:
        if suffix in connections[prefix]:
            loss -= connections[prefix][suffix]

    print("\nSzansa:")
    print(connections[prefix][suffix])
    print("Strata (1 - szansa):")
    print(loss)
