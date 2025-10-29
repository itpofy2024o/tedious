// cardShuffle.go
package main

import (
    "fmt"
    "log"
    "math/rand"
    "os"
    "strconv"
    "time"
)

func main() {
    if len(os.Args) != 2 {
        log.Fatal("Usage: go run cardShuffle.go <num>")
    }
    num, err := strconv.Atoi(os.Args[1])
    if err != nil || num > 52 {
        log.Fatal("shut down; a poker is no more than 52")
    }

    rand.Seed(time.Now().UnixNano())

    variation := []string{"A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"}
    style := []string{"C", "D", "H", "S"}

    ans := []string{}
    for len(ans) < num {
        deck := buildShuffledDeck(variation, style)
        draw := deck[rand.Intn(52)]
        if !contains(ans, draw) {
            ans = append(ans, draw)
        }
    }

    fmt.Println(strings.Join(ans, ""))
}

func buildShuffledDeck(variation, style []string) []string {
    deck := []string{}
    seen := make(map[string]bool)

    for len(deck) < 52 {
        card := variation[rand.Intn(len(variation))] + style[rand.Intn(len(style))]
        if !seen[card] {
            seen[card] = true
            deck = append(deck, card)
        }
    }
    return deck
}

func contains(s []string, e string) bool {
    for _, a := range s {
        if a == e {
            return true
        }
    }
    return false
}
