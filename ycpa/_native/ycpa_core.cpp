// ============================================================================
//  ycpa_core.cpp  —  YCPA-P performans-kritik çekirdeği (MUTABIK FINAL)
// ============================================================================
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#include <array>

namespace py = pybind11;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Index;

struct PenaltyMatrices1D {
    MatrixXd R1;
    MatrixXd R2;
};

struct PenaltyMatricesND {
    MatrixXd R1;
    MatrixXd R2;
    std::vector<std::vector<int>> multi_indices;
};

inline double median_inplace(std::vector<double> v) {
    const std::size_t n = v.size();
    if (n == 0) return 0.0;
    if (n & 1u) {
        auto mid = v.begin() + n / 2;
        std::nth_element(v.begin(), mid, v.end());
        return *mid;
    } else {
        auto hi = v.begin() + n / 2;
        std::nth_element(v.begin(), hi, v.end());
        const double a = *hi;
        const double b = *std::max_element(v.begin(), hi);
        return 0.5 * (a + b);
    }
}

inline double mad(const VectorXd& y) {
    std::vector<double> buf(y.data(), y.data() + y.size());
    const double m = median_inplace(buf);
    for (double& v : buf) v = std::abs(v - m);
    return median_inplace(std::move(buf));
}

inline double compute_SR(const VectorXd& yp, const VectorXd& yt, double eps) {
    const Index n = yt.size();
    Index hit = 0;
    for (Index i = 0; i < n; ++i) {
        if (std::abs(yp[i] - yt[i]) < eps) ++hit;
    }
    const double sr = static_cast<double>(hit) / static_cast<double>(n);
    return std::max(sr, 1.0 / (2.0 * static_cast<double>(n)));
}

struct PenaltySpec {
    const MatrixXd* R1 = nullptr;
    const MatrixXd* R2 = nullptr;
    double gamma1 = 0.0;
    double gamma2 = 0.0;
    bool active() const { return (gamma1 != 0.0 && R1) || (gamma2 != 0.0 && R2); }
};

inline double quad_form(const VectorXd& b, const MatrixXd& R) {
    return b.dot(R * b);
}

inline double mdl_score_cached(const VectorXd& yp, const VectorXd& yt, int k, int H,
                               double eps,
                               const VectorXd* beta = nullptr,
                               const PenaltySpec* pen = nullptr) {
    const Index n = yt.size();
    const double sr  = compute_SR(yp, yt, eps);
    const double logHk = (k > 0) ? static_cast<double>(k) * std::log(2.0 * H) : 0.0;
    double L = -static_cast<double>(n) * std::log(sr)
             + (logHk / static_cast<double>(n)) * static_cast<double>(k);
    if (beta && pen && pen->active()) {
        if (pen->gamma1 != 0.0 && pen->R1)
            L += pen->gamma1 * quad_form(*beta, *pen->R1);
        if (pen->gamma2 != 0.0 && pen->R2)
            L += pen->gamma2 * quad_form(*beta, *pen->R2);
    }
    return L;
}

inline double mdl_score(const VectorXd& yp, const VectorXd& yt, int k, int H,
                        const VectorXd* beta = nullptr,
                        const PenaltySpec* pen = nullptr) {
    const double m   = mad(yt);
    const double eps = 0.4 * (m > 0.0 ? m : 1.0);
    return mdl_score_cached(yp, yt, k, H, eps, beta, pen);
}

inline void deriv_row(double x, int deg, int order, Eigen::Ref<VectorXd> out) {
    for (int p = 0; p <= deg; ++p) {
        if (order == 0) {
            out[p] = std::pow(x, p);
        } else if (order == 1) {
            out[p] = (p >= 1) ? p * std::pow(x, p - 1) : 0.0;
        } else {
            out[p] = (p >= 2) ? p * (p - 1) * std::pow(x, p - 2) : 0.0;
        }
    }
}

PenaltyMatrices1D smoothness_penalty_matrices_1d(int deg, double x_min,
                                                 double x_max, int n_grid = 600) {
    if (deg < 0) throw std::invalid_argument("deg < 0");
    if (n_grid < 2) throw std::invalid_argument("n_grid < 2");
    const int M = deg + 1;
    const double dx = (x_max - x_min) / (n_grid - 1);
    MatrixXd R1 = MatrixXd::Zero(M, M);
    MatrixXd R2 = MatrixXd::Zero(M, M);
    VectorXd d1(M), d2(M);
    for (int g = 0; g < n_grid; ++g) {
        const double x = x_min + dx * g;
        deriv_row(x, deg, 1, d1);
        deriv_row(x, deg, 2, d2);
        R1.noalias() += (d1 * d1.transpose()) * dx;
        R2.noalias() += (d2 * d2.transpose()) * dx;
    }
    return {std::move(R1), std::move(R2)};
}

MatrixXd boundary_reduce_1d(int deg, double x_min, double x_max) {
    const int M = deg + 1;
    VectorXd a1(M), a2(M), b1(M), b2(M);
    deriv_row(x_min, deg, 1, a1); deriv_row(x_min, deg, 2, a2);
    deriv_row(x_max, deg, 1, b1); deriv_row(x_max, deg, 2, b2);
    MatrixXd Rb = b2 * b1.transpose() - a2 * a1.transpose();
    return 0.5 * (Rb + Rb.transpose());
}

MatrixXd design_matrix_1d(int deg, const VectorXd& x) {
    const Index n = x.size();
    MatrixXd Phi(n, deg + 1);
    if (n == 0) return Phi;
    Phi.col(0).setConstant(1.0);
    for (int p = 1; p <= deg; ++p) {
        Phi.col(p) = Phi.col(p - 1).cwiseProduct(x);
    }
    return Phi;
}

inline void generate_multi_indices(const std::vector<int>& degrees, size_t dim,
                                   std::vector<int>& current,
                                   std::vector<std::vector<int>>& out) {
    if (dim == degrees.size()) { out.push_back(current); return; }
    for (int i = 0; i <= degrees[dim]; ++i) {
        current[dim] = i;
        generate_multi_indices(degrees, dim + 1, current, out);
    }
}

inline double col_deriv_scalar(double val, int p, int order) {
    if (order == 0) return std::pow(val, p);
    if (order == 1) return (p >= 1) ? p * std::pow(val, p - 1) : 0.0;
    if (order == 2) return (p >= 2) ? p * (p - 1) * std::pow(val, p - 2) : 0.0;
    return 0.0;
}

MatrixXd design_matrix_nd(const std::vector<std::vector<int>>& multi_indices,
                           const std::vector<VectorXd>& X_cols) {
    const int M = static_cast<int>(multi_indices.size());
    const Index n = X_cols[0].size();
    MatrixXd Phi(n, M);
    for (int j = 0; j < M; ++j) {
        Phi.col(j).setConstant(1.0);
        const auto& mi = multi_indices[j];
        for (size_t d = 0; d < X_cols.size(); ++d) {
            int p = mi[d];
            if (p > 0) {
                Phi.col(j) = Phi.col(j).cwiseProduct(X_cols[d].array().pow(static_cast<double>(p)).matrix());
            }
        }
    }
    return Phi;
}

PenaltyMatricesND build_penalty_matrices_nd(const std::vector<int>& degrees,
                                            const std::vector<std::pair<double, double>>& bounds,
                                            int n_grid_per_dim = 30) {
    const size_t D = degrees.size();
    if (D != bounds.size()) throw std::invalid_argument("Dimension mismatch.");
    std::vector<std::vector<int>> multi_indices;
    std::vector<int> current(D, 0);
    generate_multi_indices(degrees, 0, current, multi_indices);
    const int M = static_cast<int>(multi_indices.size());
    Index G = 1;
    for (size_t d = 0; d < D; ++d) G *= n_grid_per_dim;
    double dV = 1.0;
    std::vector<VectorXd> axes(D);
    for (size_t d = 0; d < D; ++d) {
        double step = (bounds[d].second - bounds[d].first) / (n_grid_per_dim - 1);
        dV *= step;
        axes[d].setLinSpaced(n_grid_per_dim, bounds[d].first, bounds[d].second);
    }
    std::vector<std::array<MatrixXd, 3>> deriv_tab(D);
    for (size_t d = 0; d < D; ++d) {
        int max_deg = degrees[d];
        for (int order = 0; order <= 2; ++order) {
            deriv_tab[d][order] = MatrixXd::Zero(n_grid_per_dim, max_deg + 1);
            for (int k = 0; k < n_grid_per_dim; ++k) {
                double x = axes[d][k];
                for (int p = 0; p <= max_deg; ++p)
                    deriv_tab[d][order](k, p) = col_deriv_scalar(x, p, order);
            }
        }
    }
    std::vector<std::vector<int>> grid_k(G, std::vector<int>(D));
    for (Index g = 0; g < G; ++g) {
        Index temp = g;
        for (size_t d = 0; d < D; ++d) { grid_k[g][d] = temp % n_grid_per_dim; temp /= n_grid_per_dim; }
    }
    MatrixXd R1 = MatrixXd::Zero(M, M);
    MatrixXd R2 = MatrixXd::Zero(M, M);
    for (size_t i = 0; i < D; ++i) {
        for (size_t j = 0; j < D; ++j) {
            MatrixXd Hmat(G, M);
            for (int m = 0; m < M; ++m) {
                const auto& mi = multi_indices[m];
                for (Index g = 0; g < G; ++g) {
                    double term = 1.0;
                    for (size_t d = 0; d < D; ++d) {
                        int order = (d == i ? 1 : 0) + (d == j ? 1 : 0);
                        term *= deriv_tab[d][order](grid_k[g][d], mi[d]);
                    }
                    Hmat(g, m) = term;
                }
            }
            R2.selfadjointView<Eigen::Lower>().rankUpdate(Hmat.transpose(), dV);
        }
    }
    for (size_t i = 0; i < D; ++i) {
        MatrixXd Gmat(G, M);
        for (int m = 0; m < M; ++m) {
            const auto& mi = multi_indices[m];
            for (Index g = 0; g < G; ++g) {
                double term = 1.0;
                for (size_t d = 0; d < D; ++d) {
                    int order = (d == i ? 1 : 0);
                    term *= deriv_tab[d][order](grid_k[g][d], mi[d]);
                }
                Gmat(g, m) = term;
            }
        }
        R1.selfadjointView<Eigen::Lower>().rankUpdate(Gmat.transpose(), dV);
    }
    R2.triangularView<Eigen::Upper>() = R2.transpose().triangularView<Eigen::Upper>();
    R1.triangularView<Eigen::Upper>() = R1.transpose().triangularView<Eigen::Upper>();

    return {std::move(R1), std::move(R2), std::move(multi_indices)};
}

VectorXd ols(const MatrixXd& X, const VectorXd& y) {
    return X.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);
}

VectorXd fit_regularized_linear(const MatrixXd& Phi, const VectorXd& y,
                                const MatrixXd* R1 = nullptr,
                                const MatrixXd* R2 = nullptr,
                                double gamma1 = 0.0, double gamma2 = 0.0,
                                double ridge = 1e-10) {
    const double n = static_cast<double>(y.size());
    MatrixXd A = Phi.transpose() * Phi;
    A.diagonal().array() += ridge;
    if (gamma1 != 0.0 && R1) A.noalias() += (n * gamma1) * (*R1);
    if (gamma2 != 0.0 && R2) A.noalias() += (n * gamma2) * (*R2);
    const VectorXd b = Phi.transpose() * y;
    return A.ldlt().solve(b);
}

struct GibbsResult { double H, Hmax, He, r_heat, var_E, T; };

GibbsResult gibbs_entropy(const VectorXd& energies, double T = 1.0) {
    const Index N = energies.size();
    const double emin = energies.minCoeff();
    VectorXd w(N);
    double wsum = 0.0;
    for (Index i = 0; i < N; ++i) { w[i] = std::exp(-(energies[i] - emin) / (T + 1e-300)); wsum += w[i]; }
    double H = 0.0;
    for (Index i = 0; i < N; ++i) { const double p = w[i] / wsum; H += -p * std::log(p + 1e-300); }
    const double Hmax = std::log(static_cast<double>(std::max<Index>(N, 2)));
    const double He = H / (Hmax + 1e-300);
    const double meanE = energies.mean();
    double var = 0.0;
    for (Index i = 0; i < N; ++i) { const double d = energies[i] - meanE; var += d * d; }
    var /= static_cast<double>(N);
    return {H, Hmax, He, 0.01 + 0.99 * He, var, T};
}

struct Chain { VectorXd beta; double energy; };
struct MCMCResult { VectorXd beta_mcmc; double energy; GibbsResult gibbs; int n_chains, n_elite; };

class TournamentMCMC {
public:
    TournamentMCMC(int n_chains = 25, int n_warmup = 200, int n_deep = 2000,
                   int n_elite = 5, double step_size = 0.05, int H_piH = 20,
                   std::uint64_t seed = 42)
        : n_chains_(n_chains), n_warmup_(n_warmup), n_deep_(n_deep),
          n_elite_(n_elite), step_size_(step_size), H_piH_(H_piH), seed_(seed) {}

    MCMCResult fit(const MatrixXd& X, const VectorXd& y,
                   const MatrixXd* R1 = nullptr, const MatrixXd* R2 = nullptr,
                   double gamma1 = 0.0, double gamma2 = 0.0) const {
        const Index k = X.cols();
        PenaltySpec pen{R1, R2, gamma1, gamma2};
        const double m_cached   = mad(y);
        const double eps_cached = 0.4 * (m_cached > 0.0 ? m_cached : 1.0);

        std::vector<Chain> chains(n_chains_);
        #pragma omp parallel for schedule(dynamic)
        for (int c = 0; c < n_chains_; ++c) {
            std::mt19937_64 rng(seed_ + 1000ull * static_cast<std::uint64_t>(c));
            std::normal_distribution<double> init(0.0, 0.5);
            std::normal_distribution<double> prop(0.0, step_size_);
            std::uniform_real_distribution<double> unif(0.0, 1.0);
            VectorXd b(k);
            for (Index i = 0; i < k; ++i) b[i] = init(rng);
            auto energy_local = [&](const VectorXd& bv) -> double {
                const VectorXd yp = X * bv;
                return mdl_score_cached(yp, y, static_cast<int>(k), H_piH_, eps_cached, &bv, &pen);
            };
            double e = energy_local(b);
            double T = 2.0;
            for (int it = 0; it < n_warmup_; ++it) {
                VectorXd bc = b;
                for (Index i = 0; i < k; ++i) bc[i] += prop(rng);
                const double ec = energy_local(bc);
                if (ec < e || unif(rng) < std::exp(std::min(0.0, (e - ec) / T))) { b = bc; e = ec; }
                T *= 0.995;
            }
            chains[c] = {std::move(b), e};
        }

        VectorXd energies(n_chains_);
        for (int c = 0; c < n_chains_; ++c) energies[c] = chains[c].energy;
        const GibbsResult gb = gibbs_entropy(energies);

        std::sort(chains.begin(), chains.end(),
                  [](const Chain& a, const Chain& b) { return a.energy < b.energy; });
        std::vector<Chain> elite;
        elite.push_back(chains.front());
        for (std::size_t i = 1; i < chains.size() && (int)elite.size() < n_elite_; ++i) {
            bool diverse = true;
            for (const auto& e : elite)
                if ((chains[i].beta - e.beta).norm() <= 0.1) { diverse = false; break; }
            if (diverse) elite.push_back(chains[i]);
        }
        for (std::size_t i = elite.size(); (int)elite.size() < n_elite_ && i < chains.size(); ++i)
            elite.push_back(chains[i]);

        VectorXd best_b = elite.front().beta;
        double   best_e = elite.front().energy;
        double   T      = gb.r_heat * 0.5;
        for (std::size_t ei = 0; ei < elite.size(); ++ei) {
            std::mt19937_64 rng(seed_ + 7777ull + 13ull * ei);
            std::normal_distribution<double> prop(0.0, step_size_ * 0.3);
            std::uniform_real_distribution<double> unif(0.0, 1.0);
            VectorXd b = elite[ei].beta;
            double e = elite[ei].energy;
            auto energy_local = [&](const VectorXd& bv) -> double {
                const VectorXd yp = X * bv;
                return mdl_score_cached(yp, y, static_cast<int>(k), H_piH_, eps_cached, &bv, &pen);
            };
            for (int it = 0; it < n_deep_; ++it) {
                VectorXd bc = b;
                for (Index i = 0; i < k; ++i) bc[i] += prop(rng);
                const double ec = energy_local(bc);
                if (ec < e || unif(rng) < std::exp(std::min(0.0, (e - ec) / T))) { b = bc; e = ec; }
                if (e < best_e) { best_e = e; best_b = b; }
                T *= 0.9997;
            }
        }
        return {std::move(best_b), best_e, gb, n_chains_, n_elite_};
    }
private:
    int n_chains_, n_warmup_, n_deep_, n_elite_;
    double step_size_;
    int H_piH_;
    std::uint64_t seed_;
};

PYBIND11_MODULE(ycpa_core, m) {
    m.doc() = "YCPA-P performans-kritik cekirdegi (C++/Eigen).";
    m.def("mad", &mad, py::arg("y"));
    m.def("compute_sr", &compute_SR, py::arg("yp"), py::arg("yt"), py::arg("eps"));
    m.def("mdl_score",
          [](const VectorXd& yp, const VectorXd& yt, int k, int H) {
              return mdl_score(yp, yt, k, H, nullptr, nullptr);
          },
          py::arg("yp"), py::arg("yt"), py::arg("k"), py::arg("H") = 20);
    m.def("mdl_score_reg",
          [](const VectorXd& yp, const VectorXd& yt, int k, int H,
             const VectorXd& beta, const MatrixXd& R1, const MatrixXd& R2,
             double gamma1, double gamma2) {
              PenaltySpec pen{&R1, &R2, gamma1, gamma2};
              return mdl_score(yp, yt, k, H, &beta, &pen);
          },
          py::arg("yp"), py::arg("yt"), py::arg("k"), py::arg("H"),
          py::arg("beta"), py::arg("R1"), py::arg("R2"),
          py::arg("gamma1") = 0.0, py::arg("gamma2") = 0.0);
    m.def("design_matrix_1d", &design_matrix_1d, py::arg("deg"), py::arg("x"));
    m.def("smoothness_penalty_matrices_1d",
          [](int deg, double xmin, double xmax, int n_grid) {
              auto r = smoothness_penalty_matrices_1d(deg, xmin, xmax, n_grid);
              return py::make_tuple(std::move(r.R1), std::move(r.R2));
          },
          py::arg("deg"), py::arg("x_min"), py::arg("x_max"), py::arg("n_grid") = 600);
    m.def("boundary_reduce_1d", &boundary_reduce_1d,
          py::arg("deg"), py::arg("x_min"), py::arg("x_max"));
    m.def("design_matrix_nd", &design_matrix_nd, py::arg("multi_indices"), py::arg("X_cols"));
    m.def("build_penalty_matrices_nd", [](const std::vector<int>& degrees,
                                          const std::vector<std::pair<double, double>>& bounds,
                                          int n_grid_per_dim) {
        auto r = build_penalty_matrices_nd(degrees, bounds, n_grid_per_dim);
        return py::make_tuple(std::move(r.R1), std::move(r.R2), std::move(r.multi_indices));
    }, py::arg("degrees"), py::arg("bounds"), py::arg("n_grid_per_dim") = 30);
    m.def("ols", &ols, py::arg("X"), py::arg("y"));
    m.def("fit_regularized_linear",
          [](const MatrixXd& Phi, const VectorXd& y,
             py::object R1, py::object R2, double gamma1, double gamma2, double ridge) {
              MatrixXd r1, r2;
              const MatrixXd* p1 = nullptr;
              const MatrixXd* p2 = nullptr;
              if (!R1.is_none()) { r1 = R1.cast<MatrixXd>(); p1 = &r1; }
              if (!R2.is_none()) { r2 = R2.cast<MatrixXd>(); p2 = &r2; }
              return fit_regularized_linear(Phi, y, p1, p2, gamma1, gamma2, ridge);
          },
          py::arg("Phi"), py::arg("y"),
          py::arg("R1") = py::none(), py::arg("R2") = py::none(),
          py::arg("gamma1") = 0.0, py::arg("gamma2") = 0.0, py::arg("ridge") = 1e-10);
    py::class_<GibbsResult>(m, "GibbsResult")
        .def_readonly("H", &GibbsResult::H).def_readonly("Hmax", &GibbsResult::Hmax)
        .def_readonly("He", &GibbsResult::He).def_readonly("r_heat", &GibbsResult::r_heat)
        .def_readonly("var_E", &GibbsResult::var_E).def_readonly("T", &GibbsResult::T);
    m.def("gibbs_entropy", &gibbs_entropy, py::arg("energies"), py::arg("T") = 1.0);
    py::class_<MCMCResult>(m, "MCMCResult")
        .def_readonly("beta_mcmc", &MCMCResult::beta_mcmc)
        .def_readonly("energy", &MCMCResult::energy)
        .def_readonly("gibbs", &MCMCResult::gibbs)
        .def_readonly("n_chains", &MCMCResult::n_chains)
        .def_readonly("n_elite", &MCMCResult::n_elite);
    py::class_<TournamentMCMC>(m, "TournamentMCMC")
        .def(py::init<int, int, int, int, double, int, std::uint64_t>(),
             py::arg("n_chains") = 25, py::arg("n_warmup") = 200,
             py::arg("n_deep") = 2000, py::arg("n_elite") = 5,
             py::arg("step_size") = 0.05, py::arg("H_piH") = 20, py::arg("seed") = 42)
        .def("fit",
             [](const TournamentMCMC& self, const MatrixXd& X, const VectorXd& y,
                py::object R1, py::object R2, double gamma1, double gamma2) {
                 MatrixXd r1, r2;
                 const MatrixXd* p1 = nullptr;
                 const MatrixXd* p2 = nullptr;
                 if (!R1.is_none()) { r1 = R1.cast<MatrixXd>(); p1 = &r1; }
                 if (!R2.is_none()) { r2 = R2.cast<MatrixXd>(); p2 = &r2; }
                 return self.fit(X, y, p1, p2, gamma1, gamma2);
             },
             py::arg("X"), py::arg("y"),
             py::arg("R1") = py::none(), py::arg("R2") = py::none(),
             py::arg("gamma1") = 0.0, py::arg("gamma2") = 0.0);
}
