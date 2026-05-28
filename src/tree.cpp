#include "tree.h"
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cmath>
#include <stdexcept>

// OpenMP: если компилируем с -fopenmp, используем настоящий API; иначе
// подставляем заглушки (1 поток), и код корректно работает последовательно.
#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_thread_num()  { return 0; }
static inline int omp_get_max_threads() { return 1; }
#endif

namespace tgbm {

// -----------------------------------------------------------------------
//  Вспомогательные расчёты
// -----------------------------------------------------------------------

double Tree::LeafWeight(double G, double H) const {
    return -G / (H + params_.lambda);
}

double Tree::LeafScore(double G, double H) const {
    // -0.5 * G^2 / (H + lambda)  — вклад листа в целевой функционал
    return -0.5 * G * G / (H + params_.lambda);
}

// -----------------------------------------------------------------------
//  Публичные методы
// -----------------------------------------------------------------------

void Tree::Build(const MatrixView& X,
                 const std::vector<double>& g,
                 const std::vector<double>& h,
                 const std::vector<int>& indices)
{
    nodes_.clear();
    nodes_.reserve(512);
    int n_features = X.empty() ? 0 : X.Cols();
    feature_importance_.assign(n_features, 0.0);
    BuildNode(X, g, h, indices, 0);
}

double Tree::Predict(const double* row) const {
    int idx = 0;
    while (true) {
        const Node& n = nodes_[idx];
        if (n.type == NodeType::LEAF) {
            return n.weight;
        } else if (n.type == NodeType::BINARY) {
            idx = (row[n.feature] <= n.threshold1) ? n.left : n.right;
        } else {
            // TERNARY
            if      (row[n.feature] <= n.threshold1) idx = n.left;
            else if (row[n.feature] <= n.threshold2) idx = n.mid;
            else                                      idx = n.right;
        }
    }
}

// -----------------------------------------------------------------------
//  Построение узла (рекурсия)
// -----------------------------------------------------------------------

int Tree::BuildNode(const MatrixView& X,
                    const std::vector<double>& g,
                    const std::vector<double>& h,
                    const std::vector<int>& indices,
                    int depth)
{
    // Суммы градиентов и гессианов в текущем узле
    double G = 0.0, H = 0.0;
    for (int i : indices) { G += g[i]; H += h[i]; }

    // Создаём узел (пока листовой — потом заполним, если разбиём)
    int node_idx = static_cast<int>(nodes_.size());
    nodes_.push_back(Node{});
    Node& node = nodes_[node_idx];
    node.weight = LeafWeight(G, H);

    // Условия превращения в лист
    bool max_depth_reached = (depth >= params_.max_depth);
    bool too_few_samples   = (H < params_.min_child_weight);
    if (max_depth_reached || too_few_samples || indices.size() <= 1) {
        node.type = NodeType::LEAF;
        return node_idx;
    }

    // Ищем лучшее разбиение
    SplitInfo best = FindBestSplit(X, g, h, indices);

    // Если прирост слишком мал — оставляем листом
    if (best.gain <= 0.0 || best.feature < 0) {
        node.type = NodeType::LEAF;
        return node_idx;
    }

    // Разбиваем indices на 2 или 3 подмножества
    std::vector<int> left_idx, mid_idx, right_idx;
    for (int i : indices) {
        double val = X(i, best.feature);
        if (val <= best.threshold1) {
            left_idx.push_back(i);
        } else if (best.is_ternary && val <= best.threshold2) {
            mid_idx.push_back(i);
        } else {
            right_idx.push_back(i);
        }
    }

    // Проверка: каждая ветвь должна иметь достаточно объектов
    auto sum_h = [&](const std::vector<int>& idx) {
        double s = 0; for (int i : idx) s += h[i]; return s;
    };
    bool left_ok  = !left_idx.empty()  && sum_h(left_idx)  >= params_.min_child_weight;
    bool right_ok = !right_idx.empty() && sum_h(right_idx) >= params_.min_child_weight;
    bool mid_ok   = !best.is_ternary || (!mid_idx.empty() && sum_h(mid_idx) >= params_.min_child_weight);

    if (!left_ok || !right_ok || !mid_ok) {
        node.type = NodeType::LEAF;
        return node_idx;
    }

    // Заполняем поля разбиения
    node.feature    = best.feature;
    node.threshold1 = best.threshold1;
    node.threshold2 = best.threshold2;
    node.type       = best.is_ternary ? NodeType::TERNARY : NodeType::BINARY;

    // Учитываем вклад признака в важность (суммарный gain по разбиениям)
    if (best.feature >= 0 && best.feature < static_cast<int>(feature_importance_.size()))
        feature_importance_[best.feature] += best.gain;

    // Рекурсивно строим потомков
    // ВАЖНО: после вызовов BuildNode вектор nodes_ может перевыделить память,
    // поэтому обращаемся к nodes_[node_idx] только после завершения рекурсии
    int left_node  = BuildNode(X, g, h, left_idx,  depth + 1);
    int right_node = BuildNode(X, g, h, right_idx, depth + 1);
    int mid_node   = -1;
    if (best.is_ternary) {
        mid_node = BuildNode(X, g, h, mid_idx, depth + 1);
    }

    nodes_[node_idx].left  = left_node;
    nodes_[node_idx].right = right_node;
    nodes_[node_idx].mid   = mid_node;

    return node_idx;
}

// -----------------------------------------------------------------------
//  Поиск лучшего разбиения
// -----------------------------------------------------------------------

Tree::SplitInfo Tree::FindBestSplit(
        const MatrixView& X,
        const std::vector<double>& g,
        const std::vector<double>& h,
        const std::vector<int>& indices) const
{
    SplitInfo best;
    int n_features = X.Cols();

    // Пул буферов: по одному на поток, размером n. Аллоцируется один раз
    // (первый узел дерева), далее переиспользуется. Каждый поток пишет
    // только в свой буфер fvals_pool_[tid] — гонки нет.
    int n_threads = omp_get_max_threads();
    if (static_cast<int>(fvals_pool_.size()) < n_threads)
        fvals_pool_.resize(n_threads);
    for (auto& buf : fvals_pool_)
        if (static_cast<int>(buf.size()) < X.Rows()) buf.resize(X.Rows());

    // Результат по каждому признаку пишется в свой слот (без гонки),
    // а затем сворачивается ПОСЛЕДОВАТЕЛЬНО в исходном порядке j — это
    // даёт побитово тот же результат, что и серийная версия (одинаковое
    // разрешение ничьих: при равном gain побеждает меньший индекс признака).
    std::vector<SplitInfo> per_feat(n_features);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::vector<double>& fvals = fvals_pool_[tid];

        #pragma omp for schedule(static)
        for (int j = 0; j < n_features; ++j) {
            // Значения признака j для текущих объектов
            for (int i : indices) fvals[i] = X(i, j);

            SplitInfo bj;  // лучшее разбиение по признаку j (gain = -inf по умолчанию)

            if (params_.branching == 2 || params_.branching == 0) {
                SplitInfo s = FindBestBinary(fvals, g, h, indices, j);
                if (s.gain > bj.gain) bj = s;
            }
            if (params_.branching == 3 || params_.branching == 0) {
                SplitInfo s = FindBestTernary(fvals, g, h, indices, j);
                if (s.gain > bj.gain) bj = s;
            }
            per_feat[j] = bj;
        }
    }

    // Последовательная детерминированная свёртка
    for (int j = 0; j < n_features; ++j)
        if (per_feat[j].gain > best.gain) best = per_feat[j];

    return best;
}

// -----------------------------------------------------------------------
//  Построение гистограммы (равноширинный или квантильный биннинг)
//
//  Возвращает суммы градиентов/гессианов по корзинам и пороги между
//  корзинами. Способ биннинга выбирается параметром params_.binning:
//    0 — равноширинный  (equal-width): δ = (xmax - xmin) / B
//    1 — квантильный    (quantile):    границы по эмпирическим квантилям
//
//  В обоих случаях thresholds[b] — это верхняя граница корзины b, то есть
//  объекты со значением <= thresholds[b] лежат в корзинах 0..b. Такое
//  единое представление позволяет использовать одну и ту же логику поиска
//  разбиения для обоих режимов.
// -----------------------------------------------------------------------

Tree::Histogram Tree::BuildHistogram(
        const std::vector<double>& fvals,
        const std::vector<double>& g,
        const std::vector<double>& h,
        const std::vector<int>& indices) const
{
    Histogram hist;

    // Диапазон значений признака в текущем узле
    double fmin = std::numeric_limits<double>::max();
    double fmax = std::numeric_limits<double>::lowest();
    for (int i : indices) { fmin = std::min(fmin, fvals[i]); fmax = std::max(fmax, fvals[i]); }
    if (fmax - fmin < 1e-12) { hist.n_bins = 0; return hist; }  // признак константный

    const int B = params_.n_bins;

    if (params_.binning == 1) {
        // ---------------------------------------------------------------
        //  Квантильный (равночастотный) биннинг
        // ---------------------------------------------------------------
        // Сортируем значения признака активных объектов и берём B-1
        // разделителей на позициях ⌊b·n/B⌋, b = 1..B-1.
        std::vector<double> sorted_vals;
        sorted_vals.reserve(indices.size());
        for (int i : indices) sorted_vals.push_back(fvals[i]);
        std::sort(sorted_vals.begin(), sorted_vals.end());
        const int n = static_cast<int>(sorted_vals.size());

        std::vector<double>& bnd = hist.thresholds;
        bnd.reserve(B - 1);
        for (int b = 1; b < B; ++b) {
            long long pos = static_cast<long long>(b) * n / B;
            if (pos >= n) pos = n - 1;
            bnd.push_back(sorted_vals[pos]);
        }
        // Дедупликация: при повторяющихся значениях признака соседние
        // границы совпадают; такие вырожденные (пустые) корзины схлопываем.
        bnd.erase(std::unique(bnd.begin(), bnd.end()), bnd.end());
        // Граница, равная максимуму, порождает пустую правую корзину — убираем.
        while (!bnd.empty() && bnd.back() >= fmax) bnd.pop_back();

        hist.n_bins = static_cast<int>(bnd.size()) + 1;
        hist.bin_g.assign(hist.n_bins, 0.0);
        hist.bin_h.assign(hist.n_bins, 0.0);
        for (int i : indices) {
            // Номер корзины = число границ, строго меньших значения.
            // upper_bound даёт первую границу > val, поэтому объекты, равные
            // границе, попадают в левую корзину (согласовано с правилом
            // x <= threshold → влево в Predict()).
            auto it = std::upper_bound(bnd.begin(), bnd.end(), fvals[i]);
            int b = static_cast<int>(it - bnd.begin());
            hist.bin_g[b] += g[i];
            hist.bin_h[b] += h[i];
        }
    } else {
        // ---------------------------------------------------------------
        //  Равноширинный биннинг (поведение исходной версии)
        // ---------------------------------------------------------------
        const double bin_width = (fmax - fmin) / B;
        hist.n_bins = B;
        hist.bin_g.assign(B, 0.0);
        hist.bin_h.assign(B, 0.0);
        for (int i : indices) {
            int b = std::min(static_cast<int>((fvals[i] - fmin) / bin_width), B - 1);
            hist.bin_g[b] += g[i];
            hist.bin_h[b] += h[i];
        }
        hist.thresholds.resize(B - 1);
        for (int b = 0; b < B - 1; ++b)
            hist.thresholds[b] = fmin + (b + 1) * bin_width;
    }

    return hist;
}

// -----------------------------------------------------------------------
//  Бинарное разбиение через гистограмму
// -----------------------------------------------------------------------

Tree::SplitInfo Tree::FindBestBinary(
        const std::vector<double>& fvals,
        const std::vector<double>& g,
        const std::vector<double>& h,
        const std::vector<int>& indices,
        int feat_idx) const
{
    SplitInfo best;
    best.feature    = feat_idx;
    best.is_ternary = false;

    Histogram hist = BuildHistogram(fvals, g, h, indices);
    const int B = hist.n_bins;
    if (B < 2) return best;  // признак константный или почти-константный

    // Суммарные G и H по всему узлу
    double G = 0.0, H = 0.0;
    for (int b = 0; b < B; ++b) { G += hist.bin_g[b]; H += hist.bin_h[b]; }
    double node_score = LeafScore(G, H);

    // Сканируем пороги: разбиение после корзины b (b = 0..B-2)
    double GL = 0.0, HL = 0.0;
    for (int b = 0; b < B - 1; ++b) {
        GL += hist.bin_g[b];
        HL += hist.bin_h[b];
        double GR = G - GL, HR = H - HL;
        if (HL < params_.min_child_weight || HR < params_.min_child_weight) continue;

        // Gain2 = -(score_L + score_R - score_node) - gamma
        //       =  0.5*(GL²/(HL+λ) + GR²/(HR+λ) - G²/(H+λ)) - γ
        double gain = -(LeafScore(GL, HL) + LeafScore(GR, HR) - node_score) - params_.gamma;
        if (gain > best.gain) {
            best.gain       = gain;
            best.threshold1 = hist.thresholds[b];  // граница между корзиной b и b+1
        }
    }
    return best;
}

// -----------------------------------------------------------------------
//  Троичное разбиение через гистограмму
// -----------------------------------------------------------------------

Tree::SplitInfo Tree::FindBestTernary(
        const std::vector<double>& fvals,
        const std::vector<double>& g,
        const std::vector<double>& h,
        const std::vector<int>& indices,
        int feat_idx) const
{
    SplitInfo best;
    best.feature    = feat_idx;
    best.is_ternary = true;

    Histogram hist = BuildHistogram(fvals, g, h, indices);
    const int B = hist.n_bins;
    if (B < 3) return best;  // для троичного нужно минимум 3 корзины

    // Префиксные суммы для быстрого вычисления GL, GM, GR за O(1)
    std::vector<double> cum_g(B + 1, 0.0), cum_h(B + 1, 0.0);
    for (int b = 0; b < B; ++b) {
        cum_g[b + 1] = cum_g[b] + hist.bin_g[b];
        cum_h[b + 1] = cum_h[b] + hist.bin_h[b];
    }
    double G = cum_g[B], H = cum_h[B];
    double node_score = LeafScore(G, H);

    // Перебираем пары порогов (b1 < b2):
    //   threshold1 = thresholds[b1], threshold2 = thresholds[b2]
    for (int b1 = 0; b1 < B - 2; ++b1) {
        double GL = cum_g[b1 + 1], HL = cum_h[b1 + 1];
        if (HL < params_.min_child_weight) continue;

        for (int b2 = b1 + 1; b2 < B - 1; ++b2) {
            double GM = cum_g[b2 + 1] - cum_g[b1 + 1];
            double HM = cum_h[b2 + 1] - cum_h[b1 + 1];
            if (HM < params_.min_child_weight) continue;

            double GR = G - cum_g[b2 + 1];
            double HR = H - cum_h[b2 + 1];
            if (HR < params_.min_child_weight) continue;

            // Gain3 = -(score_L + score_M + score_R - score_node) - 2*gamma
            double gain = -(LeafScore(GL, HL) + LeafScore(GM, HM)
                          + LeafScore(GR, HR) - node_score) - 2.0 * params_.gamma;
            if (gain > best.gain) {
                best.gain       = gain;
                best.threshold1 = hist.thresholds[b1];
                best.threshold2 = hist.thresholds[b2];
            }
        }
    }
    return best;
}

// -----------------------------------------------------------------------
//  Статистика дерева
// -----------------------------------------------------------------------

void Tree::DepthStats(int idx, int depth,
                      int& max_d, double& sum_d,
                      int& leaf_cnt, int& ternary_cnt) const
{
    if (idx < 0 || idx >= static_cast<int>(nodes_.size())) return;
    const Node& n = nodes_[idx];

    if (n.type == NodeType::LEAF) {
        max_d = std::max(max_d, depth);
        sum_d += depth;
        ++leaf_cnt;
        return;
    }
    if (n.type == NodeType::TERNARY) ++ternary_cnt;

    DepthStats(n.left,  depth + 1, max_d, sum_d, leaf_cnt, ternary_cnt);
    DepthStats(n.right, depth + 1, max_d, sum_d, leaf_cnt, ternary_cnt);
    if (n.type == NodeType::TERNARY)
        DepthStats(n.mid, depth + 1, max_d, sum_d, leaf_cnt, ternary_cnt);
}

int Tree::LeafCount() const {
    int max_d = 0, leaf_cnt = 0, ternary_cnt = 0;
    double sum_d = 0.0;
    if (!nodes_.empty()) DepthStats(0, 0, max_d, sum_d, leaf_cnt, ternary_cnt);
    return leaf_cnt;
}

int Tree::MaxDepth() const {
    int max_d = 0, leaf_cnt = 0, ternary_cnt = 0;
    double sum_d = 0.0;
    if (!nodes_.empty()) DepthStats(0, 0, max_d, sum_d, leaf_cnt, ternary_cnt);
    return max_d;
}

double Tree::AvgDepth() const {
    int max_d = 0, leaf_cnt = 0, ternary_cnt = 0;
    double sum_d = 0.0;
    if (!nodes_.empty()) DepthStats(0, 0, max_d, sum_d, leaf_cnt, ternary_cnt);
    return leaf_cnt > 0 ? sum_d / leaf_cnt : 0.0;
}

int Tree::TernaryCount() const {
    int max_d = 0, leaf_cnt = 0, ternary_cnt = 0;
    double sum_d = 0.0;
    if (!nodes_.empty()) DepthStats(0, 0, max_d, sum_d, leaf_cnt, ternary_cnt);
    return ternary_cnt;
}

} // namespace tgbm
