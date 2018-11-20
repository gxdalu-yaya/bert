import sys

for line in sys.stdin:
    datas = line.strip().split("\t")
    if len(datas) != 2:
        continue
    print(datas[0])
    print(datas[1])
    print("")
