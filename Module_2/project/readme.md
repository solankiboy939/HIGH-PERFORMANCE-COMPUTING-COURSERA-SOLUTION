# Data Search Programming Assignment  
**Assignment for _Introduction to Parallel Programming with CUDA_**  
*GPU Programming Specialization — Johns Hopkins University | Coursera*

![CUDA](https://img.shields.io/badge/CUDA-NVIDIA-%2376B900?logo=nvidia)
![C++17](https://img.shields.io/badge/C++-17-blue?logo=cplusplus)
![Build](https://img.shields.io/badge/Makefile-Supported-green)

This project implements a **GPU-accelerated parallel search** that locates a target integer in a 1D array using CUDA threads. The solution supports random data generation, user-specified inputs, CSV file reading (extra credit), and both sorted/unsorted arrays.

> ⚠️ **Academic Use Only**: This code is shared for **learning and reference**. Please respect your course’s academic integrity policy.

---

## ✅ Features

- Searches for a value in an array using **1D CUDA kernel**
- Supports:
  - Randomly generated integer arrays
  - Sorted or unsorted data (`-s true|false`)
  - Custom search value (`-v`)
  - Input from CSV file (`-f`)
- Thread-safe result reporting using **atomic operations**
- Output saved to `output-<ID>.txt` for autograder compatibility

---

## 🛠️ Requirements

- **NVIDIA GPU** (compute capability ≥ 3.0)
- **CUDA Toolkit** (v11.0+)
- Linux or WSL (Windows Subsystem for Linux)
- `make`, `nvcc`, and C++17 support

---

## 📦 Files

| File | Description |
|------|-------------|
| `search.cu` | Main CUDA source (kernel + host logic) |
| `search.h` | Header with `__constant__ int d_v` and function declarations |
| `test_data.csv` | Sample input: `717899641,236192674,...` |
| `Makefile` | Build configuration |
| `run.sh` | Script to run all required test cases |

---

## ▶️ How to Build & Run

### 1. Build
```bash
make clean build
```
### 2. Run Examples
```bash
./search.exe -p TFpFX
```
### 3. View Result
```bash
cat output-IzONJ.txt
```
### Expected output:
``` bash
Data: 717899641 236192674 8063503 2024977982 453135755 604963279 93301624 2004344247 636111415 1270883455 
Searching for value: 93301624 
Found Index: 6
