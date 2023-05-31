import os
import shutil

testCSV = "Test.csv"
testDir = "Test_dir"

# kreiraj direktorij gdje ce se spremiti testne slike
os.makedirs(testDir, exist_ok=True)

# otvori CVS sa labelama i putanjama
rows = open(testCSV).read().strip().split("\n")[1:]


# prolazak kroz sve unose u CSV-u; kopiraj sliku u poddirektorij
for r in rows:

    (label, imagePath) = r.strip().split(",")[-2:]
    os.makedirs(os.path.join(testDir,label), exist_ok=True)
    shutil.copy(os.path.join("", imagePath), os.path.join(testDir,label))