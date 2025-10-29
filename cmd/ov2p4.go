package main

import (
    "fmt"
    "log"
    "os"
    "os/exec"
)

func convertMOVtoMP4(inputPath, outputPath string) error {
    // Check if input file exists
    if _, err := os.Stat(inputPath); os.IsNotExist(err) {
        return fmt.Errorf("input file not found: %s", inputPath)
    }

    // Build FFmpeg command
    cmd := exec.Command("ffmpeg",
        "-i", inputPath,           // Input file
        "-c:v", "libx264",         // Video codec
        "-preset", "fast",         // Encoding preset
        "-crf", "23",              // Constant Rate Factor (quality)
        "-c:a", "aac",             // Audio codec
        "-threads", "4",           // Use 4 threads
        "-y",                      // Overwrite output without asking
        outputPath,                // Output file
    )

    // Capture output for debugging
    cmd.Stdout = os.Stdout
    cmd.Stderr = os.Stderr

    fmt.Printf("Converting %s → %s...\n", inputPath, outputPath)

    // Run the command
    if err := cmd.Run(); err != nil {
        return fmt.Errorf("ffmpeg failed: %w", err)
    }

    fmt.Println("Conversion completed successfully!")
    return nil
}

func main() {
    inputPath := "test.mov"
    outputPath := "output.mp4"

    if err := convertMOVtoMP4(inputPath, outputPath); err != nil {
        log.Fatalf("Error: %v", err)
    }
}
