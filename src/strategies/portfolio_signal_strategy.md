# Portfolio & Signal Strategy Framework

本文档详细阐述了 FinRL-Trading-Refactor 项目中基于信号的投资组合策略框架。该框架旨在实现从股票池管理、信号生成、权重分配到交易执行的完整流程，支持高度模块化和可扩展的策略开发。

## 1. 整体架构 (Architecture)

该框架采用分层设计，主要包含以下核心组件：

1.  **Universe Layer (`UniverseManager`)**: 负责管理动态股票池，处理股票的入选和剔除时间，提供每日有效的交易标的。
2.  **Signal Layer (`BaseSignalEngine`, `TSMOMSignalEngine`, `MLStockSelectionStrategy`)**: 负责生成交易信号。支持基于规则的动量策略（TSMOM）和基于机器学习的选股策略（ML）。
3.  **Execution Layer (`ExecutionManager`)**: 负责将信号转化为目标仓位权重。包含风控逻辑（持仓限制、杠杆控制）、调仓频率控制和资金分配。
4.  **Infrastructure (`StrategyLogger`)**: 提供全链路的异步日志记录，追踪信号、持仓变化和异常信息。

### 数据流 (Data Flow)

```mermaid
graph LR
    A[Stock Selection / Fundamentals] --> B(UniverseManager)
    C[Market Data (Price/Volume)] --> D(Signal Engine)
    B --> D
    D -->|Signal DataFrame| E(ExecutionManager)
    E -->|Target Weights| F[Backtest / Live Trading]
    G[StrategyLogger] -.-> B
    G[StrategyLogger] -.-> D
    G[StrategyLogger] -.-> E
```

---

## 2. 核心类详解 (Class Details)

### 2.1. UniverseManager

*   **文件**: `src/strategies/universe_manager.py`
*   **功能**: 将低频（如季度）的选股结果映射到日频交易日历，管理每日可交易的股票白名单。
*   **核心方法**:
    *   `__init__(stock_selection_df, col_map, trading_calendar, ...)`:
        *   初始化管理器。
        *   **输入**:
            *   `stock_selection_df`: 包含选股结果的 DataFrame（需含日期和股票代码）。
            *   `col_map`: 列名映射字典。
            *   `trading_calendar`: 交易日历（DatetimeIndex）。
        *   **逻辑**: 计算每条选股记录的生效日（`activate_date`）和失效日，构建每日的 `universe_df` 和快速查询索引 `universe_map`。
    *   `get_universe(date)`:
        *   **输入**: `date` (日期)。
        *   **输出**: `Set[str]` (当日有效股票代码集合)。
        *   **描述**: O(1) 复杂度查询指定日期在池内的所有股票。
    *   `is_in_universe(tic_name, date)`:
        *   **输入**: `tic_name` (股票代码), `date` (日期)。
        *   **输出**: `bool`。
        *   **描述**: 判断单只股票在当日是否在池内。
    *   `log_universe_events_for_date(date)`:
        *   **描述**: 比较当日与前一日的股票池，识别新增（In）和剔除（Out）的股票，并通过 `StrategyLogger` 记录事件。

### 2.2. BaseSignalEngine

*   **文件**: `src/strategies/base_signal.py`
*   **功能**: 信号生成的抽象基类，处理数据加载、时间对齐、信号扩展和 Universe 过滤。
*   **核心方法**:
    *   `load_price_data_multi_file(folder, tics)` / `load_price_data_single_file(filepath)`:
        *   **功能**: 加载行情数据。支持从文件夹读取多文件（每个股票一个 CSV）或从单一大文件读取。支持 `chunk_size` 分块读取以优化内存。
    *   `compute_signals(price_source, tics, position_df)`:
        *   **功能**: **主入口**。
        *   **流程**:
            1.  调用数据加载方法获取行情数据。
            2.  应用数据时间过滤器（`data_start_date`, `data_end_date`）。
            3.  遍历每只股票，调用 `generate_signal_one_ticker` 计算原始信号。
            4.  调用 `_expand_signal_to_daily` 将低频信号（如月频）填充为日频。
            5.  **关键**: 结合 `UniverseManager`，将不在当日 Universe 中的股票信号强制置为 0。
        *   **输出**: `signal_df` (Index=Date, Columns=Tickers, Values=Signal)。
    *   `generate_signal_one_ticker(df)`:
        *   **功能**: 抽象方法，需子类实现。计算单只股票的信号序列。
    *   `_expand_signal_to_daily(signal_df)`:
        *   **功能**: 根据 `get_signal_frequency()` 的返回（D/W/M），将信号向前填充至下一个调仓日。

### 2.3. TSMOMSignalEngine (继承自 BaseSignalEngine)

*   **文件**: `src/strategies/tsmomsignal.py`
*   **功能**: 实现时序动量（Time Series Momentum）策略。
*   **策略逻辑**:
    *   计算过去 `lookback_months`（默认12个月）的收益率：$R_{t-12 \to t-1} = \frac{P_{t-1}}{P_{t-12}} - 1$。
    *   如果 $R > \text{neutral\_band}$，信号为 +1（做多）。
    *   如果 $R < -\text{neutral\_band}$，信号为 -1（做空）。
    *   否则信号为 0。
*   **核心方法**:
    *   `generate_signal_one_ticker(df)`:
        *   重采样行情数据为月频（`resample("M")`）。
        *   计算动量并生成信号。
        *   应用信号时间过滤器（`signal_start_date`, `signal_end_date`）。

### 2.4. ExecutionManager

*   **文件**: `src/strategies/execution_engine.py`
*   **功能**: 将信号转换为目标权重，应用投资组合约束。
*   **核心参数**:
    *   `rebalance_freq`: 调仓频率 ('D', 'W', 'M')。
    *   `max_positions`: 最大持仓股票数。
    *   `max_weight` / `min_weight`: 单股最大/最小权重限制。
    *   `gross_leverage`: 总杠杆限制。
    *   `cooling_days`: 平仓后的冷冻期天数。
*   **核心方法**:
    *   `generate_weight_matrix(signal_df)`:
        *   **输入**: 日频信号矩阵。
        *   **流程**: 按日遍历，调用 `step()` 更新权重。
        *   **输出**: `weights_df` (Index=Date, Columns=Tickers, Values=Weight)。
        *   **特性**: 如果存在 `_compute_target_weights` 方法（如 ML 策略中可能用到），会尝试调用以获取更复杂的权重分配，否则使用基于 `step` 的逐步调整逻辑。
    *   `step(date, signal_series)`:
        *   **功能**: 单日状态推进。
        *   **逻辑**:
            1.  更新冷却期计数器。
            2.  判断当日是否为调仓日（`_should_rebalance`）。
            3.  处理 **Close-Only** 逻辑：如果股票被移出 Universe 但仍有持仓，仅允许平仓。
            4.  根据信号计算目标方向，调用 `_update_weight_one_name`。
            5.  应用组合级约束（最大持仓数截断、杠杆归一化）。
            6.  记录日志（`logger.log_signal`）。
    *   `_compute_target_weights(signal_df)`:
        *   **功能**: （可选）基于信号直接计算目标权重矩阵，通常用于 ML 策略或定期重平衡策略。实现等权或基于预测值的权重分配。

### 2.5. StrategyLogger

*   **文件**: `src/strategies/strategylogger.py`
*   **功能**: 高性能异步日志记录器。
*   **特性**:
    *   **AsyncWriterThread**: 后台线程写入磁盘，避免 I/O 阻塞策略计算。
    *   **Log Rotation**: 按日期建立子目录存储日志。
    *   **Categories**: 支持 `signal`, `portfolio`, `universe`, `error` 等多种日志类别。
*   **核心方法**:
    *   `log_signal(...)`: 记录单笔交易信号及执行动作（OPEN, CLOSE, HOLD, ADJUST）。
    *   `log_portfolio(...)`: 记录每日组合状态。
    *   `log_universe(...)`: 记录 Universe 变更事件。
    *   `close()`: 停止后台线程并强制刷新缓冲区。

### 2.6. MLStockSelectionStrategy (机器学习选股)

*   **文件**: `src/strategies/ml_strategy.py`
*   **功能**: 基于机器学习模型的选股策略。
*   **特性**:
    *   支持 **Rolling** (滚动) 和 **Single** (单次) 预测模式。
    *   **Sector Neutral**: 支持行业中性化选股（`SectorNeutralMLStrategy`）。
    *   **Weighting**: 支持等权重 (`equal`) 和最小方差 (`min_variance`) 权重分配。
    *   **Same-Day Adjustment**: 支持利用当日/最近日行情修正预测值。
*   **核心方法**:
    *   `generate_weights(data, ...)`: 主入口，执行特征准备、模型训练、预测、选股和权重分配。
    *   `_rolling_train_all_date(...)`: 执行滚动训练和预测。
    *   `_compute_min_variance_weights(...)`: 计算最小方差组合权重。

---

## 3. 调用流程示例 (Pipeline Example)

以下代码展示了如何使用该框架进行一次完整的 **TS-MOM 策略** 回测准备。

```python
import pandas as pd
import pandas_market_calendars as mcal
from strategies.strategylogger import StrategyLogger
from strategies.universe_manager import UniverseManager
from strategies.tsmomsignal import TSMOMSignalEngine
from strategies.execution_engine import ExecutionManager

# 1. 配置与初始化
CONFIG = {
    "price_file": "./feature/daily_SPX_500_feature_engineering.csv", # 单一大文件路径
    "universe_file": "./data/stock_selected_updated.csv",
    "data_start": "2016-06-01",
    "data_end": "2024-12-31",
    "backtest_start": "2018-01-01", # 信号生成开始时间
    "backtest_end": "2024-12-31",
    "col_map": {
        "tic_name": "tic", "trade_date": "trade_date", # Universe 映射
        "datetime": "datadate", "tic": "tic", "close": "adj_close" # Price 映射
    }
}

# 2. 准备基础设施
logger = StrategyLogger("TSMOM_Demo", async_mode=True)
nyse = mcal.get_calendar('NYSE')
trading_days = nyse.schedule(start_date="2016-01-01", end_date="2025-12-31").index

# 3. 初始化 UniverseManager
stock_selected = pd.read_csv(CONFIG["universe_file"], parse_dates=["trade_date"])
uni_mgr = UniverseManager(
    stock_selection_df=stock_selected,
    col_map=CONFIG["col_map"],
    trading_calendar=trading_days,
    backtest_start=CONFIG["backtest_start"],
    backtest_end=CONFIG["backtest_end"],
    logger=logger
)
all_tics = sorted(stock_selected["tic"].unique()) # 获取所有涉及的股票

# 4. 初始化 Signal Engine 并计算信号
sig_engine = TSMOMSignalEngine(
    strategy_name="tsmom_demo",
    universe_mgr=uni_mgr,
    logger=logger,
    multi_file=False,
    chunk_size=500000,
    lookback_months=12,
    neutral_band=0.10,
    signal_start_date=CONFIG["backtest_start"],
    signal_end_date=CONFIG["backtest_end"],
    data_start_date=CONFIG["data_start"],
    data_end_date=CONFIG["data_end"],
    col_map=CONFIG["col_map"]
)

# compute_signals 会自动处理：加载数据 -> 计算月度信号 -> 扩展为日频 -> 过滤 Universe
signal_df = sig_engine.compute_signals(CONFIG["price_file"], all_tics)
print("Signal Generated shape:", signal_df.shape)

# 5. 初始化 Execution Manager 并生成权重
exe_mgr = ExecutionManager(
    universe_mgr=uni_mgr,
    rebalance_freq="M",   # 月度调仓
    max_positions=20,     # 最多持仓 20 只
    max_weight=0.10,      # 单票最大 10%
    logger=logger
)

weights_df = exe_mgr.generate_weight_matrix(signal_df)
print("Weights Generated shape:", weights_df.shape)

# 6. 保存结果与清理
weights_df.to_csv("./log/final_weights.csv")
logger.close() # 确保日志写入磁盘
```

## 4. 扩展指南 (Extension Guide)

*   **开发新信号策略**:
    1.  继承 `BaseSignalEngine`。
    2.  实现 `get_signal_frequency()` 返回 'D', 'W' 或 'M'。
    3.  实现 `generate_signal_one_ticker(df)`，输入为单只股票的 DataFrame，输出为包含信号 (-1/0/1) 的 Series。
*   **自定义调仓逻辑**:
    1.  修改或继承 `ExecutionManager`。
    2.  重写 `step()` 方法以实现特殊的风控逻辑。
    3.  或实现 `_compute_target_weights(signal_df)` 以支持基于优化器（如 Mean-Variance）的权重分配。

