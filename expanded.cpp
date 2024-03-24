#include "CLI/App.hpp"
#include "CLI/Formatter.hpp"
#include "CLI/Config.hpp"

#include "nanobench.h"

#include <cmath>
#define OPERATION(X) std::exp(X)

#include <vector>
#include <random>

template<bool condition_>
double run_dense(size_t nr, const std::vector<std::vector<int> >& i, const std::vector<std::vector<double> >& x) {
    std::vector<double> buffer(nr);
    double sum = 0;

    for (int c = 0, nc = i.size(); c < nc; ++c) {
        const auto& curi = i[c];
        const auto& curx = x[c];
        std::fill(buffer.begin(), buffer.end(), 0);
        for (size_t i = 0, end = curi.size(); i < end; ++i) {
            buffer[curi[i]] = curx[i];
        }

        if constexpr(condition_) {
            const double constant = OPERATION(0.0);
            for (auto& b : buffer) {
                b = (b ? OPERATION(b) : constant);
            }
        } else {
            for (auto& b : buffer) {
                b = OPERATION(b);
            }
        }

        sum += std::accumulate(buffer.begin(), buffer.end(), 0.0);
    }

    return sum;
}

template<bool index_>
double run_sparse(size_t nr, const std::vector<std::vector<int> >& i, const std::vector<std::vector<double> >& x) {
    std::vector<double> buffer(nr);
    double sum = 0;

    typename std::conditional<index_, std::vector<size_t>, bool>::type remap;
    if constexpr(index_) {
        remap.resize(nr);
        std::iota(remap.begin(), remap.end(), 0);
    }

    for (int c = 0, nc = i.size(); c < nc; ++c) {
        const auto& curi = i[c];
        const auto& curx = x[c];
        const double constant = OPERATION(0.0);
        std::fill(buffer.begin(), buffer.end(), constant);
        for (size_t i = 0, end = curi.size(); i < end; ++i) {
            if constexpr(index_) {
                buffer[remap[curi[i]]] = OPERATION(curx[i]);
            } else {
                buffer[curi[i]] = OPERATION(curx[i]);
            }
        }
        sum += std::accumulate(buffer.begin(), buffer.end(), 0.0);
    }

    return sum;
}

int main(int argc, char* argv []) {
    CLI::App app{"Expanded testing checks"};
    double density;
    app.add_option("-d,--density", density, "Density of the expanded sparse matrix")->default_val(0.1);
    int nr;
    app.add_option("-r,--nrow", nr, "Number of rows")->default_val(10000);
    int nc;
    app.add_option("-c,--ncol", nc, "Number of columns")->default_val(10000);
    CLI11_PARSE(app, argc, argv);

    std::cout << "Testing a " << nr << " x " << nc << " matrix with a density of " << density << std::endl;

    // Simulating a sparse matrix, albeit not very efficiently, but whatever.
    std::vector<std::vector<int> > i(nc);
    std::vector<std::vector<double> > x(nc);

    std::mt19937_64 generator(1234567);
    std::uniform_real_distribution<double> distu;
    std::normal_distribution<double> distn;

    for (int c = 0; c < nc; ++c) {
        auto& curi = i[c];
        auto& curx = x[c];
        for (int r = 0; r < nr; ++r) {
            if (distu(generator) <= density) {
                curi.push_back(r);
                curx.push_back(distn(generator));
            }
        }
    }

    double expected = run_dense<false>(nr, i, x);
    std::cout << "Summation result should be " << expected << std::endl;

    // Doing the straightforward dense operation.
    ankerl::nanobench::Bench().run("dense direct", [&](){
        auto sum = run_dense<false>(nr, i, x);
        if (sum != expected) {
            std::cerr << "unexpected result from dense direct (" << sum << ")" << std::endl;
        }
    });

    // Doing the sparse conditional operation.
    ankerl::nanobench::Bench().run("dense conditional", [&](){
        auto sum = run_dense<true>(nr, i, x);
        if (sum != expected) {
            std::cerr << "unexpected result from dense conditional (" << sum << ")" << std::endl;
        }
    });

    // Doing the sparse expanded operation.
    ankerl::nanobench::Bench().run("sparse expanded", [&](){
        auto sum = run_sparse<false>(nr, i, x);
        if (sum != expected) {
            std::cerr << "unexpected result from sparse expanded (" << sum << ")" << std::endl;
        }
    });

    // Doing the sparse expanded operation with an index map.
    ankerl::nanobench::Bench().run("sparse indexed", [&](){
        auto sum = run_sparse<true>(nr, i, x);
        if (sum != expected) {
            std::cerr << "unexpected result from sparse indexed (" << sum << ")" << std::endl;
        }
    });

    return 0;
}

