from naps.pipelines.read_naps import read_naps_dataset
from tree_transformation import make_the_tree_good
from dataset import generate_tokens, get_tokens

global unique

def init_loss():
    global unique
    unique = get_tokens()


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


def loss_function(list_1, list_2):  # Funkcja straty - poprawność list_2 na bazie połączeń list_1
    global unique
    print('#################################')
    print(list_1)
    print(list_2)
    print('#################################')
    list_1 = generate_tokens(list_1, unique)
    list_2 = generate_tokens(list_2, unique)
    data = list_1


    # 1. TWORZENIE BAZY POŁĄCZEŃ
    connections = {}
    connection_length = 2  # Ile komend w sprawdzanym ciągu, 2 = patrzymy na JEDNĄ wstecz, 3 = DWIE, itd.

    for x in range(len(data) - connection_length + 1):

        # Dolicz wystąpienie znalezionego {prefixu}{suffixu}, gdzie suffix = ostatnia komenda
        prefix = "|".join(data[x: x + connection_length - 1])  # "|" - Musi to być symbol niespotykany w komendach
        suffix = data[x + connection_length - 1]

        if prefix in connections:
            if suffix in connections[prefix]:
                connections[prefix][suffix] += 1
            else:
                connections[prefix][suffix] = 1
        else:
            connections[prefix] = {}
            connections[prefix][suffix] = 1

    # Zmiana liczebności na szansę danego suffixu na tle innych suffixów dla danego prefixu
    for prefix in connections:
        how_many = 0
        for suffix in connections[prefix]:
            how_many += connections[prefix][suffix]

        for suffix in connections[prefix]:
            connections[prefix][suffix] = connections[prefix][suffix] / how_many

    #print("Połączenia:")
    #print(connections)


    # 2. LICZENIE CAŁKOWITEJ STRATY # list_2, connections
    total_loss = 0

    for x in range(connection_length - 1, len(list_2)):  # Pomijamy elementy dla których brakuje odpowiedniego prefixu
        prefix = "|".join(list_2[x - connection_length + 1: x])
        suffix = list_2[x]

        loss = 1
        if prefix in connections:
            if suffix in connections[prefix]:
                loss -= connections[prefix][suffix]
        total_loss+=loss

    return total_loss



if __name__ == "__main__":

    print("Przykładowe straty:")
    a = ['1','2','3','4','5']
    b = ['2','3','4','5','1']
    print(loss_function(a, b))

    a = ['1', '2', '3', '2', '1']
    b = ['2', '3', '2', '5', '1']
    print(loss_function(a, b))




    # PRZYKŁADOWE DANE (I ICH EWENTUALNE PRZYCIĘCIE):
    a = ["a", "b", "v", "e", "r"]
    b = get_nonflatten_tokens()

    #data_size = 1000
    data = make_the_tree_good(b)  #[0:data_size]  # Przycięcie danych do mniejszego rozmiaru, ROZWAŻNIE Z TYM CIĘCIEM!
    print("\n\nDane:")
    print(data)
    print(len(data))
    print("\n")


    # ALGORYTM WŁAŚCIWY
    connections = {}
    connection_length = 2  # Ile komend w sumie w ciągu, 2 = czyli patrzymy na JEDNO wstecz

    for x in range(len(data) - connection_length + 1):

        # Dolicz wystąpienie znalezionego {prefixu}{suffixu}, gdzie suffix = ostatnia komenda
        prefix = "|".join(data[x: x + connection_length - 1])  # "|" - Musi to być symbol niespotykany w komendach
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
    for prefix in connections:
        how_many = 0
        for suffix in connections[prefix]:
            how_many+=connections[prefix][suffix]

        for suffix in connections[prefix]:
            connections[prefix][suffix] = connections[prefix][suffix] / how_many

    print("Połączenia:")
    print(connections)


    # WYKORZYSTANIE
    # 1. Znając "aktualną" komendę i komendy ją poprzedzające, zbijamy komendy poprzedzające w prefix: "|".join(lista_komend)
    # *  W przypadku sprawdzania na jedną komendę wstecz wykorzystanie ".join" nie jest konieczne
    # 2. Suffix to nasza uzyskana komenda
    # 3. Prawdopodobieństwo uzyskujemy wywołując: connections[prefix][suffix]
    # 4. Strata (propozycja): <1 - uzyskana_szansa>   <- strata mniejsza dla bardziej prawidłowego tekstu?


    # PRZYKŁAD WYKORZYSTANIA (SPRAWNY TYLKO DLA connection_length == 2):
    if connection_length == 2:
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
    else:
        print(">> Przykład wykorzystania algorytmu możliwy do zaprezentowania tylko przy zmiennej connection_length == 2")
