#pragma once

// C-совместимый интерфейс для вызова из Python через ctypes.
// extern "C" отключает искажение имён (name mangling), делая функции
// доступными по их точным именам из разделяемой библиотеки.

#ifdef _WIN32
  #define TGBM_API __declspec(dllexport)
#else
  #define TGBM_API __attribute__((visibility("default")))
#endif

extern "C" {

// -----------------------------------------------------------------------
//  Создание и уничтожение модели
// -----------------------------------------------------------------------

// Создать объект модели с заданными параметрами.
// Возвращает непрозрачный указатель (handle), передаваемый во все функции.
// task: 0 = регрессия, 1 = бинарная классификация
// branching: 0 = adaptive, 2 = binary, 3 = ternary
// binning: 0 = равноширинный (equal-width), 1 = квантильный (quantile)
TGBM_API void* tgbm_create(
    int    n_trees,
    double learning_rate,
    int    max_depth,
    double lambda,
    double gamma,
    double min_child_weight,
    int    n_bins,
    int    branching,
    int    binning,
    double base_score,
    int    task
);

// Освободить память, выделенную для модели.
TGBM_API void tgbm_destroy(void* handle);

// -----------------------------------------------------------------------
//  Обучение и предсказание
// -----------------------------------------------------------------------

// Обучить модель.
// X — матрица признаков, хранится построчно (row-major): X[i*d + j] = x_{ij}.
// y — вектор целевых значений длиной n.
TGBM_API void tgbm_fit(
    void*        handle,
    const double* X,
    int           n,     // число объектов
    int           d,     // число признаков
    const double* y
);

// Вычислить предсказания для новых данных.
// out — предварительно выделенный буфер длиной n.
// Для регрессии: вещественные значения.
// Для классификации: вероятности P(y=1).
TGBM_API void tgbm_predict(
    void*        handle,
    const double* X,
    int           n,
    int           d,
    double*       out
);

// -----------------------------------------------------------------------
//  Статистика обученного ансамбля
// -----------------------------------------------------------------------
TGBM_API double tgbm_avg_depth(void* handle);
TGBM_API double tgbm_max_depth(void* handle);
TGBM_API double tgbm_avg_leaves(void* handle);
TGBM_API double tgbm_ternary_frac(void* handle); // доля троичных узлов [0,1]
TGBM_API double tgbm_train_time(void* handle);   // время обучения в секундах

// Важность признаков: суммарный прирост (gain) по каждому признаку.
// out — предварительно выделенный буфер длиной d (число признаков).
TGBM_API void tgbm_feature_importance(void* handle, double* out);

} // extern "C"
