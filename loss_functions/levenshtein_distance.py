# Duża część kodu, praktycznie całość dla podstawy od samych słów:
# https://blog.paperspace.com/implementing-levenshtein-distance-word-autocomplete-autocorrect/

import numpy



# Odległość Levenshteina - SŁOWA
def printDistances(distances, token1Length, token2Length):
    for t1 in range(token1Length + 1):
        for t2 in range(token2Length + 1):
            print(int(distances[t1][t2]), end=" ")
        print()

def levenshteinDistance(token1, token2):
    distances = numpy.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    a = 0
    b = 0
    c = 0

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1 - 1] == token2[t2 - 1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    # printDistances(distances, len(token1), len(token2))
    return distances[len(token1)][len(token2)]

# Znajdowanie najbliższych słów z podanego źródła słów
def calcDictDistance(word, numWords):
    file = open('1-1000.txt', 'r')
    lines = file.readlines()
    file.close()
    dictWordDist = []
    wordIdx = 0

    for line in lines:
        wordDistance = levenshteinDistance(word, line.strip())
        if wordDistance >= 10:
            wordDistance = 9
        dictWordDist.append(str(int(wordDistance)) + "-" + line.strip())
        wordIdx = wordIdx + 1

    closestWords = []
    wordDetails = []
    currWordDist = 0
    dictWordDist.sort()
    # print(dictWordDist)
    for i in range(numWords):
        currWordDist = dictWordDist[i]
        wordDetails = currWordDist.split("-")
        closestWords.append(wordDetails[1])
    return closestWords



# Odległość Levenshteina - ZDANIA
def levenshteinDistanceSentences(token1, token2):
    distances = numpy.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    a = 0
    b = 0
    c = 0

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1 - 1] == token2[t2 - 1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    # printDistances(distances, len(token1), len(token2))
    return distances[len(token1)][len(token2)]









if __name__ == "__main__":
    # Base levenshtein - for words
    print("BASE:")
    word1 = "kelm"
    word2 = "hello"
    distance = levenshteinDistance(word1, word2)
    print("Odległość Levenshteina dla słów \"" + str(word1)+"\" i \""+str(word2)+"\":")
    print(distance)

    similar = "pape"
    how_many = 3
    print(str(how_many) + " najbliższe słowa dla \""+str(similar)+"\":")
    print(calcDictDistance(similar, how_many))



    # Empowered levenshtein - for sentences
    print("\nUPGRADE:")
    sentence1 = ["def", "levenshtein", "distance"]
    sentence2 = ["hello", "levenshtein", "distance"]
    distance = levenshteinDistanceSentences(sentence1, sentence2)
    print("Odległość Levenshteina dla zdań " + str(sentence1) + " i " + str(sentence2) + ":")
    print(distance)





