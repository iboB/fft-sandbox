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

void test_1d(unsigned seed) {
    std::minstd_rand rng(seed);

    constexpr int K_SIZE = 16;

    // can't be const because of bad finufft interface
    const auto xs = [&]() {
        std::vector<double> ret;
        double cur = -2;
        for (int i = 0; i < K_SIZE; ++i) {
            cur += double(rng()) / std::numeric_limits<unsigned>::max();
            ret.push_back(cur);
        }
        return ret;
    }();

    const auto ks = [&]() {
        std::vector<std::complex<double>> ret;
        for (int i = 0; i < K_SIZE; ++i) {
            double real = double(rng()) / std::numeric_limits<unsigned>::max();
            real *= 2;
            real -= 1;
            double imag = double(rng()) / std::numeric_limits<unsigned>::max();
            imag *= 2;
            imag -= 1;
            ret.push_back(std::complex<double>(real, imag));
        }

        return ret;
    }();

    constexpr int U_SIZE = 15;

    constexpr double EPS = 1e-6;


    std::vector<std::complex<double>> r_finufft(U_SIZE);
    {
        auto fxs = xs;
        auto fks = ks;

        finufft_opts opts;
        finufft_default_opts(&opts);
        opts.nthreads = 1;
        finufft1d1(K_SIZE, fxs.data(), fks.data(), /* fft order= */ 1, EPS, U_SIZE, r_finufft.data(), &opts);
    }

    std::vector<std::complex<double>> r_ducc(U_SIZE);
    {
        ducc0::cmav<double, 2> dxs(xs.data(), {size_t(K_SIZE), 1});

        ducc0::Nufft<double, double, double> nufft(
            false,
            dxs,
            {size_t(U_SIZE)},
            EPS,
            1, // nthreads
            1.5, 2.5, // sigma min-max
            {2 * pi}, // periodicty
            false, // fft order
            {0} //origin
        );

        nufft.nu2u(
            true, // forward
            0, // verbosity
            ducc0::cmav<std::complex<double>, 1>{ks.data(), {ks.size()}},
            ducc0::vfmav<std::complex<double>>{r_ducc.data(), {r_ducc.size()}}
        );
    }

    std::cout << "finufft            ducc\n";
    for (int i = 0; i < U_SIZE; ++i) {
        std::cout << std::format(
            "{:+4.4f}{:+4.4f}i    {:+4.4f}{:+4.4f}i\n",
            r_finufft[i].real(), r_finufft[i].imag(),
            r_ducc[i].real(), r_ducc[i].imag()
        );
    }
}

int main() {
    unsigned seed = 42; // std::random_device{}();
    std::cout << "random seed: " << seed << std::endl;

    test_1d(seed);

    return 0;
}
