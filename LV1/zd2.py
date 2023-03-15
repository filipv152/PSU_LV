print("Unesite broj izmedu 0.0 i 1.0")
try:
    a = float(input())
except:
    print("niste unjeli broj")

if a < 0.6 and a >= 0.0:
    print("F")
elif a >= 0.6 and a < 0.7:
    print("D")
elif a >= 0.7 and a < 0.8:
    print("C")
elif a >= 0.8 and a < 0.9:
    print("B")
elif a >= 0.9 and a<1:
    print("A")
else:
    print("vas broj nije u zadanom intervalu")