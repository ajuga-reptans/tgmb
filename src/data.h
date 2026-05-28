#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <iomanip>

namespace tgbm {

// -----------------------------------------------------------------------
//  Датасет
// -----------------------------------------------------------------------
struct Dataset {
    std::vector<std::vector<double>> X;  // матрица признаков [n x d]
    std::vector<double>              y;  // целевая переменная [n]
    std::vector<std::string>         feature_names;

    int Rows()     const { return static_cast<int>(X.size()); }
    int Features() const { return X.empty() ? 0 : static_cast<int>(X[0].size()); }
};

// -----------------------------------------------------------------------
//  Загрузка CSV (последний столбец — целевая переменная)
// -----------------------------------------------------------------------
inline Dataset LoadCSV(const std::string& path, bool has_header = true) {
    Dataset ds;
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    std::string line;
    bool first = true;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, ',')) tokens.push_back(token);

        if (first && has_header) {
            for (int i = 0; i + 1 < static_cast<int>(tokens.size()); ++i)
                ds.feature_names.push_back(tokens[i]);
            first = false;
            continue;
        }
        first = false;

        std::vector<double> row;
        for (int i = 0; i + 1 < static_cast<int>(tokens.size()); ++i)
            row.push_back(std::stod(tokens[i]));
        ds.X.push_back(row);
        ds.y.push_back(std::stod(tokens.back()));
    }
    return ds;
}

// -----------------------------------------------------------------------
//  Разбивка на train / test
// -----------------------------------------------------------------------
inline void TrainTestSplit(const Dataset& ds,
                           Dataset& train, Dataset& test,
                           double test_frac = 0.2,
                           unsigned seed = 42)
{
    int n = ds.Rows();
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    int n_test = static_cast<int>(n * test_frac);
    int n_train = n - n_test;

    train.feature_names = ds.feature_names;
    test.feature_names  = ds.feature_names;

    for (int i = 0; i < n_train; ++i) {
        train.X.push_back(ds.X[idx[i]]);
        train.y.push_back(ds.y[idx[i]]);
    }
    for (int i = n_train; i < n; ++i) {
        test.X.push_back(ds.X[idx[i]]);
        test.y.push_back(ds.y[idx[i]]);
    }
}

// -----------------------------------------------------------------------
//  Метрики
// -----------------------------------------------------------------------

// Среднеквадратичная ошибка (RMSE)
inline double RMSE(const std::vector<double>& y_true,
                   const std::vector<double>& y_pred)
{
    double s = 0.0;
    int n = static_cast<int>(y_true.size());
    for (int i = 0; i < n; ++i) {
        double d = y_pred[i] - y_true[i];
        s += d * d;
    }
    return std::sqrt(s / n);
}

// Средняя абсолютная ошибка (MAE)
inline double MAE(const std::vector<double>& y_true,
                  const std::vector<double>& y_pred)
{
    double s = 0.0;
    int n = static_cast<int>(y_true.size());
    for (int i = 0; i < n; ++i) s += std::abs(y_pred[i] - y_true[i]);
    return s / n;
}

// AUC-ROC (бинарная классификация, прогнозы — вероятности [0,1])
inline double AUC(const std::vector<double>& y_true,
                  const std::vector<double>& y_prob)
{
    int n = static_cast<int>(y_true.size());
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return y_prob[a] > y_prob[b]; });

    double tp = 0, fp = 0;
    double pos = 0, neg = 0;
    for (int i = 0; i < n; ++i) {
        if (y_true[i] > 0.5) ++pos; else ++neg;
    }
    if (pos == 0 || neg == 0) return 0.5;

    double auc = 0.0, prev_fp = 0.0, prev_tp = 0.0;
    for (int i = 0; i < n; ++i) {
        if (y_true[order[i]] > 0.5) tp += 1.0;
        else {
            fp += 1.0;
            auc += (tp + prev_tp) * 0.5;  // трапециевидное правило
            prev_tp = tp;
        }
        prev_fp = fp;
    }
    auc /= (pos * neg);
    return auc;
}

// -----------------------------------------------------------------------
//  Сохранение датасета в CSV
// -----------------------------------------------------------------------
inline void SaveCSV(const Dataset& ds, const std::string& path) {
    std::ofstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Cannot write file: " + path);

    // Заголовок
    for (int j = 0; j < ds.Features(); ++j) {
        std::string name = j < (int)ds.feature_names.size()
                         ? ds.feature_names[j]
                         : ("x" + std::to_string(j));
        file << name << ",";
    }
    file << "target\n";

    // Данные
    file << std::fixed << std::setprecision(6);
    for (int i = 0; i < ds.Rows(); ++i) {
        for (int j = 0; j < ds.Features(); ++j)
            file << ds.X[i][j] << ",";
        file << ds.y[i] << "\n";
    }
}

// -----------------------------------------------------------------------
//  Генерация синтетического датасета для тестирования
//  (регрессия: y = sin(x0) + x1^2 + noise)
// -----------------------------------------------------------------------
inline Dataset MakeSynthetic(int n = 2000, int d = 5, unsigned seed = 0) {
    std::mt19937 rng(seed);
    std::normal_distribution<> nd(0, 1);
    std::normal_distribution<> noise(0, 0.1);

    Dataset ds;
    ds.feature_names.resize(d);
    for (int j = 0; j < d; ++j)
        ds.feature_names[j] = "x" + std::to_string(j);

    for (int i = 0; i < n; ++i) {
        std::vector<double> row(d);
        for (int j = 0; j < d; ++j) row[j] = nd(rng);
        ds.X.push_back(row);
        double target = std::sin(row[0]) + row[1] * row[1] + 0.5 * row[2] + noise(rng);
        ds.y.push_back(target);
    }
    return ds;
}

} // namespace tgbm