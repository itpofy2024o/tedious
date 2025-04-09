import numpy as np
import sys

num = sys.argv[1]

if int(num) > 52:
    raise ValueError("shut down; a poker is no more than 52")

variation = "A,2,3,4,5,6,7,8,9,T,J,Q,K".split(",")

style = "C,D,H,S".split(",")

ans = []

while len(ans)!=int(num):
    shuffle_deck = []
    while len(shuffle_deck) != 52:
        card = variation[np.random.randint(len(variation))]+style[np.random.randint(len(style))]
        if card not in shuffle_deck:
            shuffle_deck.append(card)
    draw = shuffle_deck[np.random.randint(len(shuffle_deck))]
    if draw not in ans:
        ans.append(draw)

print(ans)

ans = "".join(ans)

print(ans)
