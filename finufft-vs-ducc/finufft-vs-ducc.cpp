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

std::string complex_format(const std::complex<double>& c) {
    return std::format("{:+4.4f}{:+4.4f}i", c.real(), c.imag());
}

constexpr double EPS = 1e-6;

static constexpr bool forward_fourier = false;

constexpr int finufft_sign = forward_fourier ? -1 : 1;
constexpr bool ducc_forward = forward_fourier;

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
            finufft_sign,
            EPS,
            U_SIZE,
            r_finufft.data(),
            &opts
        );
    }

    std::vector<std::complex<double>> r_ducc(U_SIZE);
    {
        ducc0::cmav<double, 2> dxs(xs.data(), {size_t(K_SIZE), 1 });

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
            ducc_forward,
            0, // verbosity
            ducc0::cmav<std::complex<double>, 1>{ks.data(), {ks.size()}},
            ducc0::vfmav<std::complex<double>>{r_ducc.data(), {r_ducc.size()}}
        );
    }

    std::cout << "finufft          |  ducc\n";
    for (int i = 0; i < U_SIZE; ++i) {
        std::cout << complex_format(r_finufft[i]) << " | " << complex_format(r_ducc[i]) << "\n";
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
            finufft_sign,
            EPS,
            U_SIZE_X, U_SIZE_Y,
            r_finufft.data(),
            &opts
        );
    }

    // NOTE:
    // x and y are swapped in ducc, both in shape and in coords
    std::vector<std::complex<double>> r_ducc(r_finufft.size());
    {
        std::vector<double> dcoords(K_SIZE * 2);
        for (int i = 0; i < K_SIZE; ++i) {
            dcoords[i * 2 + 1] = xs[i];
            dcoords[i * 2 + 0] = ys[i];
        }

        ducc0::cmav<double, 2> vdcoords(dcoords.data(), {size_t(K_SIZE), 2});

        std::vector<size_t> ushape = {size_t(U_SIZE_Y), size_t(U_SIZE_X)};

        ducc0::Nufft<double, double, double> nufft(
            true,
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
            ducc_forward,
            0, // verbosity
            ducc0::cmav<std::complex<double>, 1>{ks.data(), {ks.size()}},
            ducc0::vfmav<std::complex<double>>{r_ducc.data(), ushape}
        );
    }

    std::cout << "finufft                                          |  ducc\n";
    for (int y = 0; y < U_SIZE_Y; ++y) {
        std::string finufft_row, ducc_row;
        for (int x = 0; x < U_SIZE_X; ++x) {
            const auto i = y * U_SIZE_X + x;
            finufft_row += complex_format(r_finufft[i]) + " ";
            ducc_row += complex_format(r_ducc[i]) + " ";
        }
        std::cout << finufft_row << " |  " << ducc_row;
        std::cout << "\n";
    }
}

void test_3d(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    const std::vector<double>& zs,
    const std::vector<std::complex<double>>& ks,
    const int U_SIZE_X,
    const int U_SIZE_Y,
    const int U_SIZE_Z
) {
    std::cout << "3D\n";

    const int K_SIZE = int(xs.size());

    std::vector<std::complex<double>> r_finufft(U_SIZE_X * U_SIZE_Y * U_SIZE_Z);
    {
        auto fxs = xs;
        auto fys = ys;
        auto fzs = zs;
        auto fks = ks;

        finufft_opts opts;
        finufft_default_opts(&opts);
        opts.nthreads = 1;

        finufft3d1(
            K_SIZE,
            fxs.data(), fys.data(), fzs.data(),
            fks.data(),
            finufft_sign,
            EPS,
            U_SIZE_X, U_SIZE_Y, U_SIZE_Z,
            r_finufft.data(),
            &opts
        );
    }

    std::vector<std::complex<double>> r_ducc(r_finufft.size());
    {
        std::vector<double> dcoords(K_SIZE * 3);
        for (int i = 0; i < K_SIZE; ++i) {
            dcoords[i * 2 + 2] = xs[i];
            dcoords[i * 2 + 1] = ys[i];
            dcoords[i * 2 + 0] = zs[i];
        }

        ducc0::cmav<double, 2> vdcoords(dcoords.data(), {size_t(K_SIZE), 3});

        std::vector<size_t> ushape = {size_t(U_SIZE_Z), size_t(U_SIZE_Y), size_t(U_SIZE_X)};

        ducc0::Nufft<double, double, double> nufft(
            true,
            vdcoords,
            ushape,
            EPS,
            1, // nthreads
            1.5, 2.5, // sigma min-max
            {2 * pi, 2 * pi, 2 * pi}, // periodicty
            false, // fft_order
            {0, 0, 0} //origin
        );

        nufft.nu2u(
            ducc_forward,
            0, // verbosity
            ducc0::cmav<std::complex<double>, 1>{ks.data(), {ks.size()}},
            ducc0::vfmav<std::complex<double>>{r_ducc.data(), ushape}
        );
    }

    std::cout << "finufft                                          |  ducc\n";
    for (int z = 0; z < U_SIZE_Z; ++z) {
        for (int y = 0; y < U_SIZE_Y; ++y) {
            std::string finufft_row, ducc_row;
            for (int x = 0; x < U_SIZE_X; ++x) {
                const auto i = z * U_SIZE_Y * U_SIZE_X + y * U_SIZE_X + x;
                finufft_row += complex_format(r_finufft[i]) + " ";
                ducc_row += complex_format(r_ducc[i]) + " ";
            }
            std::cout << finufft_row << " |  " << ducc_row;
            std::cout << "\n";
        }
        if (z != U_SIZE_Z - 1) {
            std::cout << std::string(100, '-') << "\n";
        }
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

    const auto zs = make_coord_vec(K_SIZE, rng);

    test_3d(xs, ys, zs, ks, 3, 5, 2);

    return 0;
}
