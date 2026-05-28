"""
tgbm.py — Python-интерфейс к библиотеке TernaryGBM
Вызывает C++-ядро (libtgbm.so / tgbm.dll) через ctypes.
Все вычисления выполняются в C++.
"""
# tgbm.py
import ctypes, numpy as np, os
from pathlib import Path
from sklearn.base import BaseEstimator 

def _find_library():
    here = Path(__file__).parent
    # 🔑 ОБЯЗАТЕЛЬНО ДЛЯ WINDOWS (Python 3.8+):
    if os.name == 'nt':
        os.add_dll_directory(str(here))

    # Имена, которые может выдать сборка. CMake собирает цель tgbm_core,
    # поэтому первыми идут tgbm_core.*, затем — запасные варианты.
    candidates = [
        "tgbm_core.dll", "tgbm_core.so", "tgbm_core.dylib", "tgbm_core.pyd",
        "libtgbm_core.so", "libtgbm_core.dylib",
        "tgbm.dll", "libtgbm.so", "libtgbm.dylib",
    ]
    for name in candidates:
        p = here / name
        if p.exists():
            return ctypes.CDLL(str(p))

    # На случай нестандартного суффикса — ищем по шаблону
    for pat in ("tgbm_core.*", "libtgbm_core.*", "tgbm.*", "libtgbm.*"):
        for p in here.glob(pat):
            if p.suffix.lower() in (".dll", ".so", ".dylib", ".pyd"):
                return ctypes.CDLL(str(p))

    raise FileNotFoundError(
        "Библиотека tgbm_core не найдена рядом с tgbm.py. "
        "Соберите проект (pip install .) или положите tgbm_core.dll/.so рядом."
    )

_lib = _find_library()

_dbl  = ctypes.c_double
_pdbl = ctypes.POINTER(ctypes.c_double)
_int  = ctypes.c_int
_ptr  = ctypes.c_void_p

_lib.tgbm_create.restype  = _ptr
_lib.tgbm_create.argtypes = [_int,_dbl,_int,_dbl,_dbl,_dbl,_int,_int,_int,_dbl,_int]
_lib.tgbm_destroy.restype  = None;  _lib.tgbm_destroy.argtypes  = [_ptr]
_lib.tgbm_fit.restype      = None;  _lib.tgbm_fit.argtypes      = [_ptr,_pdbl,_int,_int,_pdbl]
_lib.tgbm_predict.restype  = None;  _lib.tgbm_predict.argtypes  = [_ptr,_pdbl,_int,_int,_pdbl]
_lib.tgbm_avg_depth.restype    = _dbl; _lib.tgbm_avg_depth.argtypes    = [_ptr]
_lib.tgbm_max_depth.restype    = _dbl; _lib.tgbm_max_depth.argtypes    = [_ptr]
_lib.tgbm_avg_leaves.restype   = _dbl; _lib.tgbm_avg_leaves.argtypes   = [_ptr]
_lib.tgbm_ternary_frac.restype = _dbl; _lib.tgbm_ternary_frac.argtypes = [_ptr]
_lib.tgbm_train_time.restype   = _dbl; _lib.tgbm_train_time.argtypes   = [_ptr]
_lib.tgbm_feature_importance.restype = None
_lib.tgbm_feature_importance.argtypes = [_ptr,_pdbl]

def _ptr_of(arr):
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr, arr.ctypes.data_as(_pdbl)

class TernaryGBM(BaseEstimator):
    """
    Градиентный бустинг с троичным ветвлением.
    Вычисления выполняются в C++-ядре (libtgbm.so / tgbm.dll).

    Параметры
    ---------
    n_estimators : int          Число деревьев (default 100)
    learning_rate: float        Шаг обучения η (default 0.1)
    max_depth    : int          Максимальная глубина (default 6)
    branching    : str          'binary', 'ternary' или 'adaptive' (default)
    reg_lambda   : float        L2-регуляризация λ (default 1.0)
    gamma        : float        Мин. прирост для разбиения γ (default 0.0)
    min_child_weight: float     Мин. сумма гессианов в ветви (default 1.0)
    n_bins       : int          Корзин гистограммы B (default 64)
    binning      : str          'equal_width' (default) или 'quantile'.
                                Способ построения границ корзин гистограммы.
                                'equal_width' — равноширинные корзины;
                                'quantile'    — равночастотные (квантильные).
    base_score   : float | str  Начальное предсказание F₀. Число — фиксированное
                                значение; 'auto' (default) — вычислить из данных:
                                mean(y) для регрессии, log(p/(1-p)) для классификации.
    task         : str          'regression' или 'classification'
    """
    _BRANCHING = {'binary':2, 'ternary':3, 'adaptive':0}
    _BINNING   = {'equal_width':0, 'quantile':1}
    _TASK      = {'regression':0, 'classification':1}

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6,
                 branching='adaptive', reg_lambda=1.0, gamma=0.0,
                 min_child_weight=1.0, n_bins=64, binning='equal_width',
                 base_score='auto', task='regression'):
        if branching not in self._BRANCHING:
            raise ValueError(f"branching: {list(self._BRANCHING)}")
        if binning not in self._BINNING:
            raise ValueError(f"binning: {list(self._BINNING)}")
        if task not in self._TASK:
            raise ValueError(f"task: {list(self._TASK)}")

        self.n_estimators=n_estimators; self.learning_rate=learning_rate
        self.max_depth=max_depth;       self.branching=branching
        self.reg_lambda=reg_lambda;     self.gamma=gamma
        self.min_child_weight=min_child_weight; self.n_bins=n_bins
        self.binning=binning
        self.base_score=base_score;     self.task=task
        self._n_features = 0

        # 'auto' → NaN: C++-ядро вычислит оптимальное F₀ из данных
        base_val = float('nan') if base_score == 'auto' else float(base_score)

        self._handle = _lib.tgbm_create(
            n_estimators, learning_rate, max_depth,
            reg_lambda, gamma, min_child_weight,
            n_bins, self._BRANCHING[branching], self._BINNING[binning],
            base_val, self._TASK[task])
        if not self._handle:
            raise RuntimeError("Не удалось создать объект модели в C++.")
        self._fitted = False

    def __del__(self):
        if getattr(self,'_handle',None):
            _lib.tgbm_destroy(self._handle); self._handle=None

    def fit(self, X, y):
        """Обучение (вычисления в C++)."""
        X_a, Xp = _ptr_of(X); y_a, yp = _ptr_of(y)
        self._n_features = X_a.shape[1]
        _lib.tgbm_fit(self._handle, Xp, X_a.shape[0], X_a.shape[1], yp)
        self._fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Предсказания (вычисления в C++)."""
        self._chk()
        X_a, Xp = _ptr_of(X); n = X_a.shape[0]
        out = np.empty(n, dtype=np.float64)
        _lib.tgbm_predict(self._handle, Xp, n, X_a.shape[1],
                          out.ctypes.data_as(_pdbl))
        return out

    def predict_proba(self, X) -> np.ndarray:
        p = self.predict(X); return np.column_stack([1-p, p])

    def score(self, X, y) -> float:
        y=np.asarray(y,dtype=float); p=self.predict(X)
        if self.task=='regression':
            ss_res=((y-p)**2).sum(); ss_tot=((y-y.mean())**2).sum()
            return float(1-ss_res/ss_tot) if ss_tot>0 else 0.0
        else:  # classification
        # Гарантируем, что y — бинарные 0/1
            y_bin = (y >= 0.5).astype(int)
        # Предсказания: вероятности -> классы
            y_pred = (p >= 0.5).astype(int)
            return float((y_pred == y_bin).mean())

    def rmse(self, X, y) -> float:
        y=np.asarray(y,dtype=float); p=self.predict(X)
        return float(np.sqrt(((y-p)**2).mean()))

    def tree_stats(self) -> dict:
        """Статистика деревьев (из C++-ядра)."""
        self._chk()
        return {
            'avg_leaf_depth':  _lib.tgbm_avg_depth(self._handle),
            'max_tree_depth':  _lib.tgbm_max_depth(self._handle),
            'avg_leaf_count':  _lib.tgbm_avg_leaves(self._handle),
            'ternary_nodes_%': _lib.tgbm_ternary_frac(self._handle)*100,
            'train_time_sec':  _lib.tgbm_train_time(self._handle),
        }

    @property
    def feature_importances_(self) -> np.ndarray:
        """Важность признаков: суммарный прирост (gain) по разбиениям,
        нормированный к сумме 1 (соглашение sklearn)."""
        self._chk()
        out = np.empty(self._n_features, dtype=np.float64)
        _lib.tgbm_feature_importance(self._handle, out.ctypes.data_as(_pdbl))
        total = out.sum()
        return out / total if total > 0 else out

    def _chk(self):
        if not self._fitted: raise RuntimeError("Сначала вызовите fit().")

    def __repr__(self):
        return (f"TernaryGBM(n_estimators={self.n_estimators}, "
                f"max_depth={self.max_depth}, branching='{self.branching}')")
