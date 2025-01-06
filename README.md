## High-Throughput EdDSA Verification on Intel Processors with Advanced Vector Extensions

This repository contains the source code for the paper "High-Throughput EdDSA Verification on Intel Processors with Advanced Vector Extensions". The code provides the AVX2 implementation of the three methods discussed in the paper, enabling efficient EdDSA verification.
## Building and Testing
The source code contains a simple makefile for Clang. 
This makefile can be easily modified for other compilers (e.g. GCC), but the performance may be affected since our software was specifically "tuned" for Clang.

To compile the source code and run the speed test, use the make command:

```
$ make
$ ./test_bench
```

## Results
Refer to the paper for detailed results, comparisons, and insights on the performance of the implemented methods.
