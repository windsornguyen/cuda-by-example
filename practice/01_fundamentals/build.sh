#!/bin/bash

set -e  # Exit on error.

# Directories
SRC_DIR="./src"
TEST_DIR="./tests"
BUILD_DIR="./build"

# CUDA architecture
CUDA_ARCH="sm_80"

# Compiler flags
NVCC_FLAGS="-std=c++17 -O3 -G -g -Xcompiler -Wall -Xcompiler -Wextra"

# Create build directory if missing
mkdir -p $BUILD_DIR

# Print colored messages
print_color() {
    case $1 in
        "green") echo -e "\033[0;32m$2\033[0m" ;;
        "red") echo -e "\033[0;31m$2\033[0m" ;;
        "yellow") echo -e "\033[0;33m$2\033[0m" ;;
    esac
}

# Compile and run each test
for TEST_FILE in $TEST_DIR/*_test.cu; do
    if [ ! -f "$TEST_FILE" ]; then
        print_color "yellow" "No test files found."
        exit 0
    fi

    BASE_NAME=$(basename "$TEST_FILE" "_test.cu")
    SRC_FILE="$SRC_DIR/${BASE_NAME}.cu"
    OUTPUT_FILE="$BUILD_DIR/${BASE_NAME}_test"

    if [[ -f "$SRC_FILE" ]]; then
        print_color "green" "Compiling $TEST_FILE with $SRC_FILE"
        nvcc $NVCC_FLAGS -arch=$CUDA_ARCH -o "$OUTPUT_FILE" "$TEST_FILE" "$SRC_FILE"

        print_color "green" "Running $OUTPUT_FILE"
        "$OUTPUT_FILE" && print_color "green" "Test $BASE_NAME passed" || print_color "red" "Test $BASE_NAME failed"
    else
        print_color "yellow" "Source $SRC_FILE not found for $TEST_FILE, skipping."
    fi
done

print_color "green" "All tests completed"
