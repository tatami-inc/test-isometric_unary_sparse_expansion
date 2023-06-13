#include "CLI/App.hpp"
#include "CLI/Formatter.hpp"
#include "CLI/Config.hpp"

#include "tatami/tatami.hpp"

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

    // Doing the straightforward dense operation.
    {
        std::vector<double> buffer(mat->ncol());
        auto start = std::chrono::high_resolution_clock::now();
        auto wrk = mat->dense_column();
        double sum = 0;
        for (int c = 0; c < nc; ++c) {
            auto range = wrk->fetch_copy(c, buffer.data());
            for (int r = 0; r < nr; ++r) {
                buffer[r] = std::exp(buffer[r]);
            }
            sum += std::accumulate(buffer.begin(), buffer.begin() + nr, 0.0);
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Dense time: " << duration.count() << " for " << sum << " sum" << std::endl;
    }

    // Doing the sparse expanded operation.
    { 
        std::vector<double> buffer(mat->ncol());
        auto start = std::chrono::high_resolution_clock::now();
        auto wrk = mat->dense_column();
        double sum = 0;

        for (int c = 0; c < nc; ++c) {
            auto range = wrk->fetch(c, buffer.data());
            for (int r = 0; r < nr; ++r) {
                buffer[r] = (buffer[r] ? std::exp(buffer[r]) : 1.0);
            }
            sum += std::accumulate(buffer.begin(), buffer.begin() + nr, 0.0);
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Expanded time: " << duration.count() << " for " << sum << " sum" << std::endl;
    }

    // Doing the sparse expanded operation (II).
    { 
        std::vector<double> full(mat->ncol());
        std::vector<double> xbuffer(mat->ncol());
        std::vector<int> ibuffer(mat->ncol());

        auto start = std::chrono::high_resolution_clock::now();
        auto wrk = mat->sparse_column();
        double sum = 0;

        for (int c = 0; c < nc; ++c) {
            auto range = wrk->fetch(c, xbuffer.data(), ibuffer.data());
            std::fill(full.begin(), full.end(), 1.0);
            for (int r = 0; r < range.number; ++r) {
                full[range.index[r]] = std::exp(range.value[r]);
            }
            sum += std::accumulate(full.begin(), full.begin() + nr, 0.0);
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Sparse time: " << duration.count() << " for " << sum << " sum" << std::endl;
    }

    return 0;
}

