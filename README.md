# Performance testing for isometric expansion

## Motivation

See [tatamic-inc/tatami#62](https://github.com/tatami-inc/tatami/pull/62) for the initial motivation.

The idea here is, when applying an operation that breaks sparsity, we could either:

1. "Dense direct": extract dense output from a sparse matrix and operate on each element in the dense array.
   This pays the cost of the operation on all entries, including zeros.
2. "Dense conditional": compute the "zeroed" value once, extract dense output from a sparse matrix, and fill the buffer with either the precomputed zero (for zeros) or the operation result (otherwise).
   This only pays the cost of the operation on non-zero entries but involves a branch condition at each loop iteration.
3. "Sparse expanded": extract sparse output, operate on each element, fill the dense array with the "zeroed" value, and then insert the sparse elements into the array.
   This only pays the cost of the operation on non-zero entries but has less predictable memory accesses.
4. "Sparse indexed": extract sparse output, operate on each element, fill the dense array with the "zeroed" value, and then insert the sparse elements into the array according to a reverse-index map.
   The reverse index map represents the additional cost of indexed extraction in **tatami** and involves an extra lookup.

Which one is fastest?

## Results

Testing with `std::exp()` with GCC 7.5.0 on Ubuntu Intel i7:

```console
$ ./build/expanded
Testing a 10000 x 10000 matrix with a density of 0.1
Summation result should be 1.06488e+08

|               ns/op |                op/s |    err% |     total | benchmark
|--------------------:|--------------------:|--------:|----------:|:----------
|      606,371,030.00 |                1.65 |    1.7% |      6.64 | `dense direct`
|      319,603,239.00 |                3.13 |    0.4% |      3.51 | `direct conditional`
|      168,718,147.00 |                5.93 |    3.5% |      1.85 | `sparse expanded`
|      167,348,345.00 |                5.98 |    1.7% |      1.86 | `sparse indexed`
```

Everyone gets faster at lower densities:

```console
Testing a 10000 x 10000 matrix with a density of 0.01
Summation result should be 1.00646e+08

|               ns/op |                op/s |    err% |     total | benchmark
|--------------------:|--------------------:|--------:|----------:|:----------
|      503,186,090.00 |                1.99 |    0.4% |      5.60 | `dense direct`
|      176,230,405.00 |                5.67 |    1.3% |      1.94 | `dense conditional`
|      115,861,853.00 |                8.63 |    0.5% |      1.27 | `sparse expanded`
|      116,669,300.00 |                8.57 |    0.4% |      1.28 | `sparse indexed`
```

In all cases, the sparse algorithm wins, with only a minor penalty from reverse mapping the indices.
It seems that the cost of extra indirection is less than that of both branching and operation calls.

Switching the `OPERATION` macro to some trivial arithmetic (e.g., `X + 23.0`) and recompiling:

```console
$ ./build/expanded 
Testing a 10000 x 10000 matrix with a density of 0.1
Summation result should be 2.29998e+08

|               ns/op |                op/s |    err% |     total | benchmark
|--------------------:|--------------------:|--------:|----------:|:----------
|      144,486,460.00 |                6.92 |    0.8% |      1.59 | `dense direct`
|      277,319,906.00 |                3.61 |    0.6% |      3.08 | `dense conditional`
|      121,331,278.00 |                8.24 |    0.4% |      1.34 | `sparse expanded`
|      125,156,839.00 |                7.99 |    0.5% |      1.39 | `sparse indexed`
```

This demonstrates that the branching can be particularly expensive when the operation is cheap.
Interestingly, the direct dense method works pretty well, almost as fast as the sparse methods.
I guess there's enough low-level parallelization for contiguous access that offsets the cost of actually doing the operation.
Regardless, the sparse methods are still the best, so that's what we'll do in **tatami**.

# Build instructions

Just use the usual CMake process:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
