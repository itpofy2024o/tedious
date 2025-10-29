package main

import (
    "fmt"
    "image"
    _ "image/jpeg"
    _ "image/png"
    "os"
    "path/filepath"
    "sort"
    "strconv"
    "strings"

    "github.com/disintegration/imaging"
)

type ImageInfo struct {
    Filename string
    Width    int
    Height   int
}

func main() {
    if len(os.Args) != 2 {
        fmt.Println("Usage: go run main.go <folder_path>")
        os.Exit(1)
    }

    folderPath := os.Args[1]
    if _, err := os.Stat(folderPath); os.IsNotExist(err) {
        fmt.Printf("Folder %s does not exist.\n", folderPath)
        os.Exit(1)
    }

    result := getFolderImageDimensions(folderPath)
    if result == nil {
        os.Exit(1)
    }

    totalWidth, totalHeight, count, extra := result[0], result[1], result[2], result[3]

    avgWidth := float64(totalWidth) / float64(count)
    avgHeight := float64(totalHeight) / float64(count)

    // Sort extra by filename
    sort.Slice(extra, func(i, j int) bool {
        return extra[i].Filename < extra[j].Filename
    })

    fmt.Printf("Average width: %.2f pixels\n", avgWidth)
    fmt.Printf("Average height: %.2f pixels\n", avgHeight)
    fmt.Printf("%v %d\n", extra, len(extra))
}

func getFolderImageDimensions(folderPath string) []interface{} {
    var totalWidth, totalHeight, validCount int
    var extra []ImageInfo

    extensions := map[string]bool{
        ".jpg":  true,
        ".jpeg": true,
        ".png":  true,
        ".bmp":  true,
        ".tiff": true,
    }

    err := filepath.Walk(folderPath, func(path string, info os.FileInfo, err error) error {
        if err != nil {
            return err
        }
        if info.IsDir() {
            return nil
        }

        ext := strings.ToLower(filepath.Ext(path))
        if !extensions[ext] {
            return nil
        }

        img, err := imaging.Open(path)
        if err != nil {
            fmt.Printf("Error opening %s: %v\n", filepath.Base(path), err)
            return nil
        }

        width, height := img.Bounds().Dx(), img.Bounds().Dy()
        filename := filepath.Base(path)

        widthEndsWithZero := strings.HasSuffix(strconv.Itoa(width), "0")
        heightEndsWithZero := strings.HasSuffix(strconv.Itoa(height), "0")

        fmt.Printf("Processed %s: %d x %d pixels\n", filename, width, height)

        if !widthEndsWithZero || !heightEndsWithZero {
            fmt.Printf("%t %d %t %d\n", widthEndsWithZero, width, heightEndsWithZero, height)
            extra = append(extra, ImageInfo{Filename: filename, Width: width, Height: height})
        }

        totalWidth += width
        totalHeight += height
        validCount++

        return nil
    })

    if err != nil {
        fmt.Printf("Error processing folder %s: %v\n", folderPath, err)
        return nil
    }

    if validCount == 0 {
        fmt.Printf("No valid image files found in %s\n", folderPath)
        return nil
    }

    fmt.Printf("\nTotal number of images processed: %d\n", validCount)
    fmt.Printf("Total width: %d pixels\n", totalWidth)
    fmt.Printf("Total height: %d pixels\n", totalHeight)

    return []interface{}{totalWidth, totalHeight, validCount, extra}
}
