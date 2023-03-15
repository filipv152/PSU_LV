lista = []

while True:
    unos = input("Unesite broj: ")
    if unos == 'Done':
        break
    lista.append(int(unos))
print("Uneseni brojevi")
for broj in lista:
    print(broj)

if len(lista) > 0:
    arsr = sum(lista)/len(lista)
    print("Aritmeticka sredina:", arsr)
    min = min(lista)
    print("Minimum:", min)
    max = max(lista)
    print("Maksimum:", max)
else:
    print("Niste unijeli dobre brojeve")

lista.sort()
print("Sortirani brojevi:")
print(lista)