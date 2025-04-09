import numpy as np
import sys
import math

num = int(sys.argv[1]) #123
aio = int(sys.argv[2])

if aio != 0 and aio != 1:
    raise ValueError("irrelevant val")

if num < 1:
    raise ValueError("unacceptable val")

if aio == 0:
    batch = int(input("how many batch: ")) #8

    per_batch = float(num/batch)
    per_batch = math.floor(per_batch) #15

    print("num per batch: ",per_batch)

    result = []

    c=0

    while len(result) != num:
        batch_arr = []
        if c != batch:
            while len(batch_arr) != per_batch:
                batch_arr.append(np.random.randint(1,7))
            # print(len(batch_arr))
        else:
            left=num-len(result)
            while len(batch_arr) != left:
                batch_arr.append(np.random.randint(1,7))
            # print(len(batch_arr))
        # print(c+1,"# batch")
        c+=1
        one =""
        for i in batch_arr:
            result.append(i)
        for i in result:
            one+=str(i)
        if c > batch:
            print(one)
else:
    result = []
    while len(result)!=num:
        result.append(np.random.randint(1,7))
    one=""
    for i in result:
        one+=str(i)
    print(one)