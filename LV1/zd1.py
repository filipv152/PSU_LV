def total_euro(a, b):
    uk = a*b
    return uk

print("Upisite koliko sati ste odradili ")
sati = int(input())
print("Upisite kolika vam je satnica ")
placa = int(input())
print("Vasa ukupna placa je:", total_euro(sati,placa), "eura")