#include "tgbm_api.h"
#include "gbm.h"
#include <vector>
#include <chrono>
#include <stdexcept>

// -----------------------------------------------------------------------
//  Внутренняя структура, хранящая обученную модель и метаданные
// -----------------------------------------------------------------------
struct TgbmHandle {
    tgbm::GBM*  model      = nullptr;
    double      train_time = 0.0;
    int         n_features = 0;

    ~TgbmHandle() { delete model; }
};

// -----------------------------------------------------------------------
//  Реализация публичного API
// -----------------------------------------------------------------------

extern "C" {

void* tgbm_create(int n_trees, double learning_rate, int max_depth,
                  double lambda, double gamma, double min_child_weight,
                  int n_bins, int branching, int binning,
                  double base_score, int task)
{
    tgbm::GBMParams params;
    params.n_trees       = n_trees;
    params.learning_rate = learning_rate;
    params.base_score    = base_score;
    params.tree.max_depth         = max_depth;
    params.tree.lambda            = lambda;
    params.tree.gamma             = gamma;
    params.tree.min_child_weight  = min_child_weight;
    params.tree.n_bins            = n_bins;
    params.tree.branching         = branching;
    params.tree.binning           = binning;

    tgbm::Task t = (task == 1)
                 ? tgbm::Task::BINARY_CLASSIFICATION
                 : tgbm::Task::REGRESSION;

    auto* h = new TgbmHandle();
    h->model = new tgbm::GBM(params, t);
    return h;
}

void tgbm_destroy(void* handle)
{
    delete static_cast<TgbmHandle*>(handle);
}

void tgbm_fit(void* handle, const double* X, int n, int d, const double* y)
{
    auto* h = static_cast<TgbmHandle*>(handle);
    h->n_features = d;

    // Без копирования: оборачиваем плоский массив во вью
    tgbm::MatrixView Xv{X, n, d};
    std::vector<double> yv(y, y + n);

    auto t0 = std::chrono::high_resolution_clock::now();
    h->model->Fit(Xv, yv);
    auto t1 = std::chrono::high_resolution_clock::now();

    h->train_time = std::chrono::duration<double>(t1 - t0).count();
}

void tgbm_predict(void* handle, const double* X, int n, int d, double* out)
{
    auto* h     = static_cast<TgbmHandle*>(handle);
    tgbm::MatrixView Xv{X, n, d};
    auto  preds = h->model->Predict(Xv);
    for (int i = 0; i < n; ++i) out[i] = preds[i];
}

double tgbm_avg_depth(void* handle)
{
    return static_cast<TgbmHandle*>(handle)->model->AvgTreeDepth();
}

double tgbm_max_depth(void* handle)
{
    return static_cast<TgbmHandle*>(handle)->model->MaxTreeDepth();
}

double tgbm_avg_leaves(void* handle)
{
    return static_cast<TgbmHandle*>(handle)->model->AvgLeafCount();
}

double tgbm_ternary_frac(void* handle)
{
    return static_cast<TgbmHandle*>(handle)->model->AvgTernaryFrac();
}

double tgbm_train_time(void* handle)
{
    return static_cast<TgbmHandle*>(handle)->train_time;
}

void tgbm_feature_importance(void* handle, double* out)
{
    auto* h   = static_cast<TgbmHandle*>(handle);
    auto  imp = h->model->FeatureImportance();
    for (int j = 0; j < h->n_features; ++j)
        out[j] = (j < static_cast<int>(imp.size())) ? imp[j] : 0.0;
}

} // extern "C"
