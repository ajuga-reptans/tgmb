#pragma once
#include <vector>
#include <limits>
#include <cstddef>

namespace tgbm {

// -----------------------------------------------------------------------
//  Лёгкая обёртка-вью над плоской матрицей признаков (row-major).
//  НЕ владеет памятью — только ссылается на чужой непрерывный буфер
//  размера n*d, расположенный построчно: элемент (i, j) = data[i*d + j].
//  Позволяет передавать данные без копирования и с хорошей локальностью
//  кэша (один непрерывный блок вместо n разрозненных аллокаций).
// -----------------------------------------------------------------------
struct MatrixView {
    const double* data = nullptr;
    int n = 0;   // число строк (объектов)
    int d = 0;   // число столбцов (признаков)

    double operator()(int i, int j) const {
        return data[static_cast<std::size_t>(i) * d + j];
    }
    // Указатель на начало строки i (для предсказания по одному объекту)
    const double* row(int i) const {
        return data + static_cast<std::size_t>(i) * d;
    }
    int  Rows()  const { return n; }
    int  Cols()  const { return d; }
    bool empty() const { return n == 0; }
};

// -----------------------------------------------------------------------
//  Параметры дерева
// -----------------------------------------------------------------------
struct TreeParams {
    int    max_depth        = 6;    // максимальная глубина
    double lambda           = 1.0;  // L2-регуляризация весов листьев
    double gamma            = 0.0;  // минимальный прирост для разбиения
    double min_child_weight = 1.0;  // мин. сумма гессианов в листе
    int    n_bins           = 64;   // число корзин гистограммы
    // 2 = только бинарное, 3 = только троичное, 0 = адаптивное (выбор лучшего)
    int    branching        = 0;
    // Способ построения корзин гистограммы:
    // 0 = равноширинный (equal-width), 1 = квантильный (quantile / равночастотный)
    int    binning          = 0;
};

// -----------------------------------------------------------------------
//  Узел дерева
// -----------------------------------------------------------------------
enum class NodeType { LEAF, BINARY, TERNARY };

struct Node {
    NodeType type        = NodeType::LEAF;
    int      feature     = -1;    // индекс признака для разбиения
    double   threshold1  = 0.0;   // первый порог  (x <= t1 → левый)
    double   threshold2  = 0.0;   // второй порог  (t1 < x <= t2 → средний)
    double   weight      = 0.0;   // вес листа
    int      left        = -1;    // индекс левого потомка
    int      mid         = -1;    // индекс среднего потомка (только троичный)
    int      right       = -1;    // индекс правого потомка
};

// -----------------------------------------------------------------------
//  Дерево решений
// -----------------------------------------------------------------------
class Tree {
public:
    explicit Tree(const TreeParams& p) : params_(p) {}

    // Построить дерево по градиентам g и гессианам h
    void Build(const MatrixView& X,
               const std::vector<double>& g,
               const std::vector<double>& h,
               const std::vector<int>& indices);

    // Предсказание для одного объекта (row — указатель на строку из d значений)
    double Predict(const double* row) const;
    // Удобная перегрузка для вектора
    double Predict(const std::vector<double>& x) const { return Predict(x.data()); }

    // Статистика
    int    LeafCount()  const;
    int    MaxDepth()   const;
    double AvgDepth()   const;
    int    TernaryCount() const;   // число узлов с троичным разбиением

    // Важность признаков: суммарный прирост (gain) по разбиениям этого дерева.
    // Длина вектора = числу признаков (заполняется в Build).
    const std::vector<double>& FeatureImportance() const { return feature_importance_; }

    const std::vector<Node>& Nodes() const { return nodes_; }

private:
    // Описание лучшего разбиения, найденного при поиске
    struct SplitInfo {
        int    feature    = -1;
        double threshold1 = 0.0;
        double threshold2 = 0.0;
        bool   is_ternary = false;
        double gain       = -std::numeric_limits<double>::infinity();
    };

    // Результат построения гистограммы по одному признаку.
    // Объединяет равноширинный и квантильный режимы: оба возвращают
    // одинаковую структуру, поэтому логика поиска разбиения не зависит
    // от способа биннинга.
    struct Histogram {
        std::vector<double> bin_g;       // сумма градиентов по корзинам
        std::vector<double> bin_h;       // сумма гессианов по корзинам
        std::vector<double> thresholds;  // верхние границы корзин 0..n_bins-2
                                         // thresholds[b] — порог между корзиной b и b+1
        int n_bins = 0;                  // эффективное число корзин (0 = признак константен)
    };

    TreeParams         params_;
    std::vector<Node>  nodes_;
    std::vector<double> feature_importance_;  // прирост gain по признакам
    // Переиспользуемый буфер значений одного признака. mutable, т.к.
    // FindBestSplit — const-метод. Аллоцируется один раз на дерево и
    // переиспользуется на всех узлах/признаках (вместо аллокации на
    // каждый признак каждого узла).
    // Пул буферов значений признака — по одному на поток OpenMP. При
    // параллельном переборе признаков каждый поток пишет в свой буфер,
    // что исключает гонку. Аллоцируется один раз на дерево и переиспользуется
    // на всех узлах (без аллокаций на каждый признак/узел).
    mutable std::vector<std::vector<double>> fvals_pool_;

    // Рекурсивное построение узла; возвращает индекс созданного узла
    int BuildNode(const MatrixView& X,
                  const std::vector<double>& g,
                  const std::vector<double>& h,
                  const std::vector<int>& indices,
                  int depth);

    // Поиск лучшего разбиения по всем признакам
    SplitInfo FindBestSplit(const MatrixView& X,
                            const std::vector<double>& g,
                            const std::vector<double>& h,
                            const std::vector<int>& indices) const;

    // Поиск лучшего бинарного разбиения для одного признака
    SplitInfo FindBestBinary(const std::vector<double>& fvals,
                             const std::vector<double>& g,
                             const std::vector<double>& h,
                             const std::vector<int>& indices,
                             int feat_idx) const;

    // Поиск лучшего троичного разбиения для одного признака
    SplitInfo FindBestTernary(const std::vector<double>& fvals,
                              const std::vector<double>& g,
                              const std::vector<double>& h,
                              const std::vector<int>& indices,
                              int feat_idx) const;

    // Построение гистограммы по признаку.
    // Способ биннинга (равноширинный / квантильный) выбирается по params_.binning.
    Histogram BuildHistogram(const std::vector<double>& fvals,
                             const std::vector<double>& g,
                             const std::vector<double>& h,
                             const std::vector<int>& indices) const;

    // Оптимальный вес листа: w* = -G / (H + lambda)
    double LeafWeight(double G, double H) const;

    // Оценка выигрыша от листа: -0.5 * G^2 / (H + lambda)
    double LeafScore(double G, double H) const;

    // Статистика дерева (рекурсивный обход)
    void DepthStats(int idx, int depth,
                    int& max_d, double& sum_d,
                    int& leaf_cnt, int& ternary_cnt) const;
};

} // namespace tgbm
