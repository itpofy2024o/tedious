// dice99.go
package main

import (
    "bufio"
    "fmt"
    "log"
    "math"
    "math/rand"
    "os"
    "strconv"
    "strings"
    "time"
)

func main() {
    if len(os.Args) != 3 {
        log.Fatal("Usage: go run dice99.go <num> <aio>")
    }

    num, _ := strconv.Atoi(os.Args[1])
    aio, _ := strconv.Atoi(os.Args[2])

    if num < 1 {
        log.Fatal("unacceptable val")
    }
    if aio != 0 && aio != 1 {
        log.Fatal("irrelevant val")
    }

    rand.Seed(time.Now().UnixNano())

    if aio == 0 {
        fmt.Print("how many batch: ")
        scanner := bufio.NewScanner(os.Stdin)
        scanner.Scan()
        batch, _ := strconv.Atoi(scanner.Text())

        perBatch := int(math.Floor(float64(num) / float64(batch)))
        fmt.Printf("num per batch: %d\n", perBatch)

        result := []int{}
        for i := 0; i < batch; i++ {
            thisBatch := perBatch
            if i == batch-1 {
                thisBatch = num - len(result)
            }
            for j := 0; j < thisBatch; j++ {
                result = append(result, rand.Intn(6)+1)
            }
        }
        fmt.Println(joinInts(result))
    } else {
        result := []int{}
        for i := 0; i < num; i++ {
            result = append(result, rand.Intn(6)+1)
        }
        fmt.Println(joinInts(result))
    }
}

func joinInts(nums []int) string {
    var b strings.Builder
    for _, n := range nums {
        b.WriteString(strconv.Itoa(n))
    }
    return b.String()
}
