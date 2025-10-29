// cardNoShuffle.go
package main

import (
    "fmt"
    "log"
    "math/rand"
    "os"
    "strconv"
    "strings"
    "time"
)

func main() {
    if len(os.Args) != 2 {
        log.Fatal("Usage: go run cardNoShuffle.go <num>")
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
        probA := rand.Float64()
        var groupA []string
        switch {
        case probA <= 0.2:
            groupA = variation[:2]
        case probA <= 0.6:
            groupA = variation[2:9]
        default:
            groupA = variation[9:]
        }

        probB := rand.Float64()
        var groupB string
        switch {
        case probB <= 0.2:
            groupB = style[0]
        case probB <= 0.4:
            groupB = style[1]
        case probB <= 0.6:
            groupB = style[2]
        default:
            groupB = style[3]
        }

        card := groupA[rand.Intn(len(groupA))] + groupB
        if rand.Float64() > 0.369 && !contains(ans, card) {
            ans = append(ans, card)
        }
    }

    fmt.Println(strings.Join(ans, ""))
}

func contains(s []string, e string) bool {
    for _, a := range s {
        if a == e {
            return true
        }
    }
    return false
}
