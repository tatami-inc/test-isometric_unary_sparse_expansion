#include "CLI/App.hpp"
#include "CLI/Formatter.hpp"
#include "CLI/Config.hpp"

#include "tatami/tatami.hpp"

#include <cmath>
#define OPERATION(X) std::exp(X)

#include <chrono>
#include <vector>
#include <queue>
#include <random>

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
    std::vector<int> i, j;
    std::vector<double> x;

    std::mt19937_64 generator(1234567);
    std::uniform_real_distribution<double> distu;
    std::normal_distribution<double> distn;

    for (int c = 0; c < nc; ++c) {
        for (int r = 0; r < nr; ++r) {
            if (distu(generator) <= density) {
                i.push_back(r);
                j.push_back(c);
                x.push_back(distn(generator));
            }
        }
    }

    auto indptrs = tatami::compress_sparse_triplets<false>(nr, nc, x, i, j);
    tatami::ArrayView<double> x_view (x.data(), x.size());
    tatami::ArrayView<int> i_view (i.data(), i.size());
    tatami::ArrayView<size_t> p_view (indptrs.data(), indptrs.size());
    std::shared_ptr<tatami::NumericMatrix> mat(new tatami::CompressedSparseColumnMatrix<double, int, decltype(x_view), decltype(i_view), decltype(p_view)>(nr, nc, x_view, i_view, p_view));

    // Creating an indexing vector.
    std::vector<int> indices;
    std::vector<int> revmap(nr);
    for (int r = 0; r < nr; r += 5) {
        indices.push_back(r);
        revmap[r] = indices.size() - 1;
    }

    // Doing the straightforward dense operation.
    {
        std::vector<double> buffer(indices.size());
        auto start = std::chrono::high_resolution_clock::now();
        auto wrk = mat->dense_column(indices);
        double sum = 0;
        for (int c = 0; c < nc; ++c) {
            wrk->fetch_copy(c, buffer.data());
            for (int r = 0; r < indices.size(); ++r) {
                buffer[r] = OPERATION(buffer[r]);
            }
            sum += std::accumulate(buffer.begin(), buffer.end(), 0.0);
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Dense time: " << duration.count() << " for " << sum << " sum" << std::endl;
    }

    // Doing the sparse expanded operation.
    const double constant = OPERATION(0.0);
    { 
        std::vector<double> buffer(indices.size());
        auto start = std::chrono::high_resolution_clock::now();
        auto wrk = mat->dense_column(indices);
        double sum = 0;

        for (int c = 0; c < nc; ++c) {
            wrk->fetch_copy(c, buffer.data());
            for (int r = 0; r < indices.size(); ++r) {
                buffer[r] = (buffer[r] ? OPERATION(buffer[r]) : constant);
            }
            sum += std::accumulate(buffer.begin(), buffer.end(), 0.0);
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Expanded time: " << duration.count() << " for " << sum << " sum" << std::endl;
    }

    // Doing the sparse expanded operation (II).
    { 
        std::vector<double> full(indices.size());
        std::vector<double> xbuffer(indices.size());
        std::vector<int> ibuffer(indices.size());

        auto start = std::chrono::high_resolution_clock::now();
        auto wrk = mat->sparse_column(indices);
        double sum = 0;

        for (int c = 0; c < nc; ++c) {
            auto range = wrk->fetch(c, xbuffer.data(), ibuffer.data());
            std::fill(full.begin(), full.end(), constant);
            for (int r = 0; r < range.number; ++r) {
                full[revmap[range.index[r]]] = OPERATION(range.value[r]);
            }
            sum += std::accumulate(full.begin(), full.end(), 0.0);
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Sparse time: " << duration.count() << " for " << sum << " sum" << std::endl;
    }

    return 0;
}

