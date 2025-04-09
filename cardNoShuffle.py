import numpy as np
import sys

num = sys.argv[1]

if int(num) > 52:
    raise ValueError("shut down; a poker is no more than 52")

variation = "A,2,3,4,5,6,7,8,9,T,J,Q,K".split(",")

style = "C,D,H,S".split(",")

ans = []

while len(ans)!=int(num):
    group_a,group_b = [],[]
    prob_a = round(np.random.randint(1300)/1300,5)
    if prob_a <= 0.20000:
        group_a = variation[:2]
    elif prob_a > 0.20000 and prob_a <= 0.60000:
        group_a = variation[2:9]
    else:
        group_a = variation[9:]
    prob_b = round(np.random.randint(4000)/4000,2)
    if prob_b <=0.20:
        group_b = style[0]
    elif prob_b > 0.20 and prob_b <=0.40:
        group_b = style[1]
    elif prob_b > 0.40 and prob_b <= 0.60:
        group_b = style[2]
    else:
        group_b = style[3]
    a_index = np.random.randint(len(group_a))
    b_index = np.random.randint(len(group_b))
    out = group_a[a_index]+group_b[b_index]
    if np.random.randint(100)/100 > 0.369:
        if out not in ans:
            ans.append(out)

ans = "".join(ans)

print(ans)
