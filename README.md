# Performance testing for isometric expansion

## Motivation

See [tatamic-inc/tatami#62](https://github.com/tatami-inc/tatami/pull/62) for the initial motivation.

The idea here is, when applying an operation that breaks sparsity, we could either:

1. "Dense": extract dense output from a sparse matrix and operate on each element in the dense array.
   This pays the cost of the operation on all entries, including zeros.
2. "Expanded": Extract dense output from a sparse matrix, special case the zeros, operate on each non-zero element in the dense array.
   This only pays the cost of the operation on non-zero entries but involves a branch condition at each loop iteration.
3. "Sparse": Extract sparse output, operate on each element, fill the dense array with the "zeroed" value, and then insert the sparse elements into the array.
   This only pays the cost of the operation on non-zero entries and avoids branching but involves the extra fill and indexing steps.

Which one is faster?

## Results

Testing with `std::exp()` with GCC 7.5.0 on Ubuntu Intel i7:

```console
$ ./build/expanded
Testing a 10000 x 10000 matrix with a density of 0.1
Dense time: 778 for 1.06488e+08 sum
Expanded time: 389 for 1.06488e+08 sum
Sparse time: 227 for 1.06488e+08 sum
```

Switching the `OPERATION` macro to some trivial arithmetic (e.g., `X + 23.0`):

```console
$ ./build/expanded 
Testing a 10000 x 10000 matrix with a density of 0.1
Dense time: 141 for -42157.7 sum
Expanded time: 265 for -42157.7 sum
Sparse time: 127 for -42157.7 sum
```

Doing a more expensive indexing in the sparse case, which involves a look-up on a second vector:

```console
$ ./build/indexed 
Testing a 10000 x 10000 matrix with a density of 0.1
Dense time: 267 for 2.1294e+07 sum
Expanded time: 179 for 2.1294e+07 sum
Sparse time: 149 for 2.1294e+07 sum
```

In all cases, the sparse algorithm wins.
It seems that the cost of extra indirection is less than that of both branching and operation calls.

# Build instructions

Just use the usual CMake process:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
