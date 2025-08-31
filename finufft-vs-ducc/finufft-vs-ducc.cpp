#include <finufft.h>

#include <splat/warnings.h>

PRAGMA_WARNING_PUSH
DISABLE_MSVC_WARNING(5030) // unknown attributes
DISABLE_MSVC_WARNING(4267 4244) // conversions
DISABLE_MSVC_WARNING(4127) // conditional expression is constant
DISABLE_MSVC_WARNING(4100) // unreferenced formal parameter
#include <ducc0/nufft/nufft.h>
PRAGMA_WARNING_POP

#include <iostream>
#include <random>
#include <limits>
#include <numbers>
#include <vector>
#include <complex>
#include <string>
#include <format>

using std::numbers::pi;

double rnd(std::minstd_rand &rng) {
    return double(rng()) / std::numeric_limits<unsigned>::max();
}

std::vector<double> make_coord_vec(int N, std::minstd_rand& rng) {
    std::vector<double> ret;
    double cur = -2;
    for (int i = 0; i < N; ++i) {
        cur += rnd(rng);
        ret.push_back(cur);
    }
    return ret;
}

constexpr double EPS = 1e-6;

void test_1d(
    const std::vector<double>& xs,
    const std::vector<std::complex<double>>& ks,
    const int U_SIZE
) {
    std::cout << "1D\n";

    const int K_SIZE = int(xs.size());

    std::vector<std::complex<double>> r_finufft(U_SIZE);
    {
        auto fxs = xs;
        auto fks = ks;

        finufft_opts opts;
        finufft_default_opts(&opts);
        opts.nthreads = 1;
        finufft1d1(
            K_SIZE,
            fxs.data(),
            fks.data(),
            1, // iflag
            EPS,
            U_SIZE,
            r_finufft.data(),
            &opts
        );
    }

    std::vector<std::complex<double>> r_ducc(U_SIZE);
    {
        ducc0::cmav<double, 2> dxs(xs.data(), { size_t(K_SIZE), 1 });

        ducc0::Nufft<double, double, double> nufft(
            false,
            dxs,
            {size_t(U_SIZE)},
            EPS,
            1, // nthreads
            1.5, 2.5, // sigma min-max
            {2 * pi}, // periodicty
            false, // fft_order
            {0} //origin
        );

        nufft.nu2u(
            false, // forward
            0, // verbosity
            ducc0::cmav<std::complex<double>, 1>{ks.data(), {ks.size()}},
            ducc0::vfmav<std::complex<double>>{r_ducc.data(), {r_ducc.size()}}
        );
    }

    std::cout << "finufft          |  ducc\n";
    for (int i = 0; i < U_SIZE; ++i) {
        std::cout << std::format(
            "{:+4.4f}{:+4.4f}i  |  {:+4.4f}{:+4.4f}i\n",
            r_finufft[i].real(), r_finufft[i].imag(),
            r_ducc[i].real(), r_ducc[i].imag()
        );
    }
}

void test_2d(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    const std::vector<std::complex<double>>& ks,
    const int U_SIZE_X,
    const int U_SIZE_Y
) {
    std::cout << "2D\n";

    const int K_SIZE = int(xs.size());

    std::vector<std::complex<double>> r_finufft(U_SIZE_X * U_SIZE_Y);
    {
        auto fxs = xs;
        auto fys = ys;
        auto fks = ks;

        finufft_opts opts;
        finufft_default_opts(&opts);
        opts.nthreads = 1;

        finufft2d1(
            K_SIZE,
            fxs.data(), fys.data(),
            fks.data(),
            1,  // iflag
            EPS,
            U_SIZE_X, U_SIZE_Y,
            r_finufft.data(),
            &opts
        );
    }

    std::vector<std::complex<double>> r_ducc(r_finufft.size());
    {
        std::vector<double> dcoords(K_SIZE * 2);
        for (int i = 0; i < K_SIZE; ++i) {
            dcoords[i * 2 + 0] = xs[i];
            dcoords[i * 2 + 1] = ys[i];
        }

        ducc0::cmav<double, 2> vdcoords(dcoords.data(), {size_t(K_SIZE), 2});

        std::vector<size_t> ushape = {size_t(U_SIZE_X), size_t(U_SIZE_Y)};

        ducc0::Nufft<double, double, double> nufft(
            false,
            vdcoords,
            ushape,
            EPS,
            1, // nthreads
            1.5, 2.5, // sigma min-max
            {2 * pi, 2 * pi}, // periodicty
            false, // fft_order
            {0, 0} //origin
        );

        nufft.nu2u(
            false, // forward
            0, // verbosity
            ducc0::cmav<std::complex<double>, 1>{ks.data(), {ks.size()}},
            ducc0::vfmav<std::complex<double>>{r_ducc.data(), ushape}
        );
    }

    for (int y = 0; y < U_SIZE_Y; ++y) {
        std::string finufft_row, ducc_row;
        for (int x = 0; x < U_SIZE_X; ++x) {
            const auto i = y * U_SIZE_X + x;
            finufft_row += std::format(
                "{:+4.4f}{:+4.4f}i ",
                r_finufft[i].real(), r_finufft[i].imag()
            );
            ducc_row += std::format(
                "{:+4.4f}{:+4.4f}i ",
                r_ducc[i].real(), r_ducc[i].imag()
            );
        }
        std::cout << finufft_row << " |  " << ducc_row;
        std::cout << "\n";
    }
}

int main() {
    unsigned seed = 42; // std::random_device{}();
    std::cout << "random seed: " << seed << "\n";

    constexpr int K_SIZE = 16;

    std::minstd_rand rng(seed);

    const auto xs = make_coord_vec(K_SIZE, rng);
    const auto ks = [&]() {
        std::vector<std::complex<double>> ret;
        for (int i = 0; i < K_SIZE; ++i) {
            double real = 2 * rnd(rng) - 1;
            double imag = 2 * rnd(rng) - 1;
            ret.push_back(std::complex<double>(real, imag));
        }
        return ret;
    }();

    test_1d(xs, ks, 15);

    const auto ys = make_coord_vec(K_SIZE, rng);

    test_2d(xs, ys, ks, 3, 5);

    return 0;
}
