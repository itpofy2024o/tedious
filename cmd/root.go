// cmd/root.go
package cmd

import (
    "os"

    "github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
    Use:   "toolbox",
    Short: "A collection of video/audio tools",
    Long:  `toolbox - your personal media processing CLI`,
}

func Execute() {
    if err := rootCmd.Execute(); err != nil {
        os.Exit(1)
    }
}

func init() {
    // This will be called automatically
}
