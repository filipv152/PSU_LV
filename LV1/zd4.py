print("Unesite ime tražene datoteke:")
imedatoteke = str(input())
spamdata = open(imedatoteke)
pouzdanost = 0 
ukpouzd = 0

try:
    for line in spamdata:
        line = line.rstrip()
        if line.startswith("X-DSPAM-Confidence:"):
            pouzdanost += float(line.split(":")[1])
            ukpouzd += 1
    spamdata.close()
    if ukpouzd > 0:
        srpouzd = ukpouzd / pouzdanost
        print("Srednja pouzdanost:", srpouzd)
    else:
        print("nema podataka u datoteci")
except FileExistsError:
    print("Datoteka ne postoji")