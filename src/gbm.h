#pragma once
#include "tree.h"
#include <vector>
#include <string>
#include <functional>
#include <cmath>

namespace tgbm {

// -----------------------------------------------------------------------
//  Параметры GBM
// -----------------------------------------------------------------------
struct GBMParams {
    int        n_trees       = 100;   // число деревьев
    double     learning_rate = 0.1;   // шаг обучения η
    // Начальное предсказание F₀. NaN = вычислить из данных автоматически:
    //   регрессия      → mean(y)
    //   классификация  → log(p / (1-p)), p = доля положительного класса
    double     base_score    = std::numeric_limits<double>::quiet_NaN();
    TreeParams tree;                  // параметры дерева
};

// -----------------------------------------------------------------------
//  Тип задачи
// -----------------------------------------------------------------------
enum class Task { REGRESSION, BINARY_CLASSIFICATION };

// -----------------------------------------------------------------------
//  Градиентный бустинг
// -----------------------------------------------------------------------
class GBM {
public:
    explicit GBM(const GBMParams& p, Task task = Task::REGRESSION)
        : params_(p), task_(task) {}

    // ---- Основной API (без копирования): данные как MatrixView ----

    // Обучение
    void Fit(const MatrixView& X, const std::vector<double>& y);

    // Предсказания (сырые значения)
    std::vector<double> PredictRaw(const MatrixView& X) const;

    // Предсказания с постобработкой
    // (для регрессии — то же самое; для классификации — сигмоида)
    std::vector<double> Predict(const MatrixView& X) const;

    // ---- Перегрузки для vector<vector<double>> ----
    // Однократно уплощают данные в непрерывный буфер и делегируют основному
    // API. Удобны для C++-экспериментов с Dataset; путь из Python использует
    // MatrixView напрямую и копий не делает.
    void Fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& y);
    std::vector<double> Predict(const std::vector<std::vector<double>>& X) const;
    std::vector<double> PredictRaw(const std::vector<std::vector<double>>& X) const;

    // Средняя глубина деревьев
    double AvgTreeDepth()   const;
    double MaxTreeDepth()   const;
    double AvgLeafCount()   const;
    double AvgTernaryFrac() const; // доля узлов с троичным разбиением

    // Важность признаков: суммарный прирост (gain) по всем разбиениям
    // ансамбля, агрегированный по каждому признаку. Длина = числу признаков.
    std::vector<double> FeatureImportance() const;

    int NumTrees() const { return static_cast<int>(trees_.size()); }

private:
    GBMParams        params_;
    Task             task_;
    std::vector<Tree> trees_;

    // Вычисление градиентов и гессианов
    void ComputeGradients(const std::vector<double>& y,
                          const std::vector<double>& F,
                          std::vector<double>& g,
                          std::vector<double>& h) const;

    static double Sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
};

} // namespace tgbm
