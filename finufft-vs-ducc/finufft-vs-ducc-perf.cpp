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

using m_float_t = double;

using std::numbers::pi;

struct bench_scope {
    std::string_view name;
    std::optional<size_t> size;
    std::chrono::high_resolution_clock::time_point start;
    explicit bench_scope(std::string_view name, std::optional<size_t> size = {})
        : name(name)
        , size(size)
        , start(std::chrono::high_resolution_clock::now())
    {
    }
    ~bench_scope() {
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        auto ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(elapsed).count();
        std::string str;
        if (size) {
            str = std::format(" {}: size={} {:5.3f} ms ({:3.3f} ms/item)", name, *size, ms, ms / *size);
        }
        else {
            str = std::format(" {}: {:5.3f} ms", name, ms);
        }
        std::cout << str << '\n';
    }
};

template <typename Func>
decltype(auto) bench(std::string_view name, Func&& func) {
    bench_scope scope(name);
    return func();
}

template <typename Func>
decltype(auto) bench(std::string_view name, size_t size, Func&& func) {
    bench_scope scope(name, size);
    return func();
}


struct dataset_3d {
    std::array<int, 3> output_shape;

    // input points and values
    std::vector<m_float_t> xs, ys, zs; // finufft-friendly format
    std::vector<m_float_t> zyxs; // ducc-friendly format

    std::vector<std::vector<std::complex<m_float_t>>> samples;

    static dataset_3d generate_random(std::array<int, 3> shape, int sample_size, int num_samples, std::minstd_rand& rng) {
        auto rnd = [&]() {
            return float(double(rng()) / std::numeric_limits<unsigned>::max());
        };

        auto make_coord_vec = [&]() {
            std::vector<m_float_t> ret;
            m_float_t cur = -m_float_t(sample_size) / 4;
            for (int i = 0; i < sample_size; ++i) {
                cur += rnd();
                ret.push_back(cur);
            }
            return ret;
        };

        dataset_3d ret;
        ret.output_shape = shape;

        ret.xs = make_coord_vec();
        ret.ys = make_coord_vec();
        ret.zs = make_coord_vec();

        ret.zyxs.resize(sample_size * 3);
        for (int i = 0; i < sample_size; ++i) {
            // note the zyx (torch tensor) order here
            ret.zyxs[i * 3 + 2] = ret.xs[i];
            ret.zyxs[i * 3 + 1] = ret.ys[i];
            ret.zyxs[i * 3 + 0] = ret.zs[i];
        }

        ret.samples.reserve(num_samples);

        for (int s = 0; s < num_samples; ++s) {
            auto& ks = ret.samples.emplace_back();
            ks.reserve(sample_size);
            for (int i = 0; i < sample_size; ++i) {
                m_float_t real = 2 * rnd() - 1;
                m_float_t imag = 2 * rnd() - 1;
                ks.push_back({real, imag});
            }
        }

        return ret;
    }
};

constexpr m_float_t EPS = 1e-6;

static constexpr bool forward_fourier = false;
constexpr int finufft_sign = forward_fourier ? -1 : 1;
constexpr bool ducc_forward = forward_fourier;

static constexpr int num_threads = 4;

double bench_finufft_3d(dataset_3d& d) {
    int64_t nmodes[] = {d.output_shape[0], d.output_shape[1], d.output_shape[2]};

    finufft_opts opts;
    finufft_default_opts(&opts);
    opts.nthreads = num_threads;

    finufft_plan plan;

    std::cout << "finufft:\n";
    {
        bench_scope b("  plan");
        finufft_makeplan(1, 3, nmodes, finufft_sign, 1, EPS, &plan, &opts);
        finufft_setpts(plan, d.xs.size(), d.xs.data(), d.ys.data(), d.zs.data(), 0, nullptr, nullptr, nullptr);
    }

    std::vector<std::complex<m_float_t>> output(d.output_shape[0] * d.output_shape[1] * d.output_shape[2]);

    double dump = 0;

    for (auto& ks : d.samples) {
        {
            bench_scope b("  exec");
            finufft_execute(plan, ks.data(), output.data());
        }
        // prevent optimizing out
        for (auto v : output) {
            dump += std::abs(v);
        }
    }

    finufft_destroy(plan);

    return dump;
}

double bench_ducc_3d(dataset_3d& d) {
    std::cout << "ducc:\n";
    // again note the zyx order here
    std::vector<size_t> ushape = {size_t(d.output_shape[2]), size_t(d.output_shape[1]), size_t(d.output_shape[0])};
    ducc0::cmav<m_float_t, 2> vdcoords(d.zyxs.data(), {size_t(d.xs.size()), 3});

    auto nufft = bench(" plan", [&]() {
        return ducc0::Nufft<m_float_t, m_float_t, m_float_t>(
            true,
            vdcoords,
            ushape,
            EPS,
            num_threads,
            1.5, 2.5, // sigma min-max
            {2 * pi, 2 * pi, 2 * pi}, // periodicty
            false, // fft_order
            {0, 0, 0} //origin
        );
    });

    std::vector<std::complex<m_float_t>> output(ushape[0] * ushape[1] * ushape[2]);

    double dump = 0;
    for (auto& ks : d.samples) {
        {
            bench_scope b("  exec");
            nufft.nu2u(
                ducc_forward,
                0, // verbosity
                ducc0::cmav<std::complex<m_float_t>, 1>{ks.data(), {ks.size()}},
                ducc0::vfmav<std::complex<m_float_t>>{output.data(), ushape}
            );
        }
        // prevent optimizing out
        for (auto v : output) {
            dump += std::abs(v);
        }
    }
    return dump;
}

int main() {
    unsigned seed = 42; // std::random_device{}();
    std::cout << "random seed: " << seed << "\n";

    std::minstd_rand rng(seed);

    dataset_3d ds[] = {
        dataset_3d::generate_random({100, 100, 30}, 40160, 10, rng)
    };

    for (auto& d : ds) {
        double fsum = bench_finufft_3d(d);
        double dsum = bench_ducc_3d(d);
        std::cout << std::format(" dump finufft: {:e}, ducc: {:e}\n", fsum, dsum);
    }

    return 0;
}
