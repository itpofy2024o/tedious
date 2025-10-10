import secrets
import sys

num = int(sys.argv[1])

if num > 52:
    raise ValueError("shut down; a poker is no more than 52")

variation = "A,2,3,4,5,6,7,8,9,T,J,Q,K".split(",")
style = "C,D,H,S".split(",")

ans = []

while len(ans) != num:
    shuffle_deck = []
    while len(shuffle_deck) != 52:
        card = variation[secrets.randbelow(len(variation))] + style[secrets.randbelow(len(style))]
        if card not in shuffle_deck:
            shuffle_deck.append(card)
    draw = shuffle_deck[secrets.randbelow(len(shuffle_deck))]
    if draw not in ans:
        ans.append(draw)

print(ans)
print("".join(ans))
