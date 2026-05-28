#include "gbm.h"
#include <numeric>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tgbm {

// -----------------------------------------------------------------------
//  Обучение
// -----------------------------------------------------------------------

void GBM::Fit(const MatrixView& X,
              const std::vector<double>& y)
{
    if (X.empty()) throw std::runtime_error("Empty training set");
    int n = X.Rows();

    // Начальное предсказание F₀.
    // Если base_score не задан (NaN), вычисляем оптимальную константу из данных:
    //   регрессия     → mean(y)             (минимум MSE)
    //   классификация → log(p / (1-p))      (минимум log-loss; p = доля класса 1)
    double base = params_.base_score;
    if (std::isnan(base)) {
        double mean_y = std::accumulate(y.begin(), y.end(), 0.0) / n;
        if (task_ == Task::REGRESSION) {
            base = mean_y;
        } else {
            double p = std::min(std::max(mean_y, 1e-6), 1.0 - 1e-6);
            base = std::log(p / (1.0 - p));
        }
    }
    params_.base_score = base;  // фиксируем, чтобы Predict использовал то же F₀

    // Начальные предсказания F(x) = base для всех объектов
    std::vector<double> F(n, base);
    std::vector<double> g(n), h(n);
    std::vector<int> all_indices(n);
    std::iota(all_indices.begin(), all_indices.end(), 0);

    trees_.clear();
    trees_.reserve(params_.n_trees);

    for (int t = 0; t < params_.n_trees; ++t) {
        // Шаг 1: вычисляем градиенты и гессианы
        ComputeGradients(y, F, g, h);

        // Шаг 2: строим дерево на псевдо-остатках
        Tree tree(params_.tree);
        tree.Build(X, g, h, all_indices);

        // Шаг 3: обновляем предсказания ансамбля (каждый F[i] независим)
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            F[i] += params_.learning_rate * tree.Predict(X.row(i));
        }

        trees_.push_back(std::move(tree));
    }
}

// -----------------------------------------------------------------------
//  Предсказание
// -----------------------------------------------------------------------

std::vector<double> GBM::PredictRaw(const MatrixView& X) const
{
    int n = X.Rows();
    std::vector<double> F(n, params_.base_score);
    // Параллелим по объектам (внешний цикл): каждый поток считает свои F[i]
    // по всем деревьям. Деревья внутри — последовательно. Гонки нет, т.к.
    // каждый i пишется ровно одним потоком.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double acc = params_.base_score;
        for (const Tree& tree : trees_) {
            acc += params_.learning_rate * tree.Predict(X.row(i));
        }
        F[i] = acc;
    }
    return F;
}

std::vector<double> GBM::Predict(const MatrixView& X) const
{
    auto F = PredictRaw(X);
    if (task_ == Task::BINARY_CLASSIFICATION) {
        for (auto& v : F) v = Sigmoid(v);
    }
    return F;
}

// -----------------------------------------------------------------------
//  Перегрузки для vector<vector<double>>: однократное уплощение
// -----------------------------------------------------------------------

static std::vector<double> flatten(const std::vector<std::vector<double>>& X,
                                   int& n, int& d)
{
    n = static_cast<int>(X.size());
    d = n ? static_cast<int>(X[0].size()) : 0;
    std::vector<double> flat(static_cast<std::size_t>(n) * d);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < d; ++j)
            flat[static_cast<std::size_t>(i) * d + j] = X[i][j];
    return flat;
}

void GBM::Fit(const std::vector<std::vector<double>>& X,
              const std::vector<double>& y)
{
    int n, d;
    std::vector<double> flat = flatten(X, n, d);
    Fit(MatrixView{flat.data(), n, d}, y);
}

std::vector<double> GBM::Predict(const std::vector<std::vector<double>>& X) const
{
    int n, d;
    std::vector<double> flat = flatten(X, n, d);
    return Predict(MatrixView{flat.data(), n, d});
}

std::vector<double> GBM::PredictRaw(const std::vector<std::vector<double>>& X) const
{
    int n, d;
    std::vector<double> flat = flatten(X, n, d);
    return PredictRaw(MatrixView{flat.data(), n, d});
}

// -----------------------------------------------------------------------
//  Градиенты и гессианы
//
//  Регрессия (MSE):
//    L = 0.5*(F - y)^2
//    g = F - y,   h = 1
//
//  Бинарная классификация (log-loss):
//    L = -y*log(p) - (1-y)*log(1-p),  p = sigmoid(F)
//    g = p - y,   h = p*(1-p)
// -----------------------------------------------------------------------

void GBM::ComputeGradients(const std::vector<double>& y,
                            const std::vector<double>& F,
                            std::vector<double>& g,
                            std::vector<double>& h) const
{
    int n = static_cast<int>(y.size());
    if (task_ == Task::REGRESSION) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            g[i] = F[i] - y[i];
            h[i] = 1.0;
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            double p = Sigmoid(F[i]);
            g[i] = p - y[i];
            h[i] = std::max(p * (1.0 - p), 1e-6);  // защита от нуля
        }
    }
}

// -----------------------------------------------------------------------
//  Статистика ансамбля
// -----------------------------------------------------------------------

double GBM::AvgTreeDepth() const {
    if (trees_.empty()) return 0.0;
    double s = 0.0;
    for (const auto& t : trees_) s += t.AvgDepth();
    return s / trees_.size();
}

double GBM::MaxTreeDepth() const {
    double m = 0.0;
    for (const auto& t : trees_) m = std::max(m, static_cast<double>(t.MaxDepth()));
    return m;
}

double GBM::AvgLeafCount() const {
    if (trees_.empty()) return 0.0;
    double s = 0.0;
    for (const auto& t : trees_) s += t.LeafCount();
    return s / trees_.size();
}

double GBM::AvgTernaryFrac() const {
    if (trees_.empty()) return 0.0;
    double total_nodes = 0.0, ternary_nodes = 0.0;
    for (const auto& t : trees_) {
        int n_nodes = static_cast<int>(t.Nodes().size());
        total_nodes   += n_nodes;
        ternary_nodes += t.TernaryCount();
    }
    return total_nodes > 0 ? ternary_nodes / total_nodes : 0.0;
}

std::vector<double> GBM::FeatureImportance() const {
    std::vector<double> imp;
    for (const auto& t : trees_) {
        const auto& fi = t.FeatureImportance();
        if (imp.size() < fi.size()) imp.resize(fi.size(), 0.0);
        for (std::size_t j = 0; j < fi.size(); ++j) imp[j] += fi[j];
    }
    return imp;
}

} // namespace tgbm
