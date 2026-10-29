# 单因子周度／月度回测框架

版本 **0.4.0**：统一离线研究报告整合 IC、十分组、年度、行情及行业／风格／市值持仓分析；详见 [统一报告说明](docs/unified_report.md)。

此前 **0.3.0** 统一采用 AIQ 字段补缺规则；实现、原始缺失/供值溯源及真实周/月回测验收见 [版本核验记录](docs/aiq_matrix_validation_20261028.md)。

读取外部已经计算好的因子值，分别测试 `ROIC_new` 和 `fundnum_promotion`，不重新计算财务或持仓变化公式。新推荐配置采用 `missing_values.method=aiq_matrix`，按 AIQ 字段矩阵规则补缺后测试；EMA 平滑、去极值、标准化和中性化仍可选，支持周度或月度调仓。输出包含覆盖度、RankIC/Pearson IC、十分组净值、超额净值、Q10−Q1、多头及多空超额夏普、年度表现和换手诊断。**不训练预测模型，也不使用持仓优化器**；目标持仓直接使用冻结的组内等权，缺失收益的统计权重另行记录，执行账本仅落实权重及成交约束。配置中的因子列表可以扩展到其他明细文件。

当前推荐 `configs/factor_test.json`（与 `factor_test_aiq_weekly.json` 相同）：AIQ本地行业、ST、直接流通市值及辅助数据经过独立适配和异常核验，按周比较 `raw`、`industry`、`size`、`industry_size`。`factor_test_aiq_monthly.json` 提供同口径月度测试，另有可选EMA配置。行业严格使用 `FACFUNC.get_industry → zxclassified() → get_classname()` 的35类名称，并保留原始中信一级；该35类不等同标准中信一级30类。收益继续使用 `/data/mongo/quotefac_repaired.feather`，不替换已修复行情。完整接入与日期规则见 [AIQ数据接入说明](docs/aiq_data_integration.md)。

`factor_test_direct_weekly.json`、`factor_test_direct_monthly.json` 只测试 `raw`；关闭去极值、标准化和EMA，使用独立样本，原始版本不依赖行业、市值或 Barra 暴露。纯行业中性不需要行业市值或个股市值，纯市值中性不需要行业或 Barra。原 `factor_test_size_proxy_weekly.json`、`factor_test_size_proxy_monthly.json` 保留行情代理市值的明确配置，不静默改其口径。沪深A股、`missing_returns=available_case` 和 `empty_groups=cash` 继续保留。旧经典配置、标准中信一级30类配置和已有冻结产物仍可用于复现此前研究。

经典模式把核心因子统计与可选筛选、执行数据分开：完整历史 ST、完整停复牌/涨跌停例外、公司行动明细及真实退市现金不作为统一门槛。已有官方 ST 历史和 GF 停牌/涨跌停代理继续使用；未覆盖状态保持未知。推荐配置在冻结组内按有效收益统计，不因一只股票缺失标签而使整个组合当周无收益；同时披露缺失股票数和目标权重。可选执行账本未满足附加记账要求时单独跳过，并披露原因。此前2026-10-13的数据报告属于历史审计记录。

这一输入分层参考 [Alphalens 的因子与未来收益接口](https://github.com/quantopian/alphalens/blob/master/alphalens/utils.py)、[Qlib 的因子评价接口](https://github.com/microsoft/qlib/blob/main/qlib/contrib/eva/alpha.py)以及[独立交易模块](https://github.com/microsoft/qlib/blob/main/qlib/backtest/exchange.py)。冻结信号后不因缺失未来收益而重分组，是本项目保留的研究约定；不声称与这些框架的所有默认处理逐项相同。

## 运行

Python ≥3.10，依赖 pandas、PyArrow、NumPy、SciPy、statsmodels、openpyxl。当前机器的 `/usr/bin/python3` 已有依赖；若 pyenv 默认解释器缺包，将下列命令中的 `python3` 替换为 `/usr/bin/python3`。其他环境可用 `python3 -m pip install -e .` 安装。

```bash
python3 scripts/prepare_aiq_inputs.py
python3 scripts/prepare_aiq_matrix_inputs.py
python3 -m fundamental_factor audit --config configs/factor_test.json
python3 -m fundamental_factor build-panel --config configs/factor_test.json
OPENBLAS_NUM_THREADS=1 python3 -m fundamental_factor run --config configs/factor_test.json
OPENBLAS_NUM_THREADS=1 python3 -m fundamental_factor run --config configs/factor_test_aiq_monthly.json
```

直接使用算好的因子值时，关键配置如下（数据映射与价格口径仍沿用完整示例配置）：

```json
{
  "backtest": {"rebalance_frequency": "monthly"},
  "missing_values": {"method": "preserve"},
  "preprocess": {"winsorize": false, "standardize": false},
  "neutralization": {
    "include_float_mcap": false,
    "variants": ["raw"],
    "sample_policy": "independent"
  }
}
```

- `rebalance_frequency` 可为 `weekly` 或 `monthly`，分别在每周／每月首个交易日开盘调仓，持有到下一期首个交易日开盘。先按完整日历定位，再截取研究区间；不会把月中起始日当作月初。因子更新频率独立于调仓频率，月更因子也可以按周测试；输入 `date` 必须表示真实可用日期，月末收盘后可用的值用于其后的调仓。
- `missing_values.method` 可为 `aiq_matrix` 或 `preserve`。新推荐配置显式选择前者；省略此项仍按 `preserve` 兼容旧冻结配置。上例选择 `preserve`，保留原快照的显式缺失；改为 `aiq_matrix` 后，普通因子字段的NaN向前延续，并记录原值、填充标记及提供值日期。
- `winsorize` 和 `standardize` 可分别开启。两者及 EMA 均关闭时，`raw` 直接测试所选缺失策略产生的输入；同时选择 `preserve` 才保留外部快照的显式缺失语义。开启预处理后 `raw` 表示未中性化版本。标准化关闭时，中性化残差也不重新标准化。有限零值和负值不因补缺被改写。
- `variants=["raw"]` 关闭中性化；改成 `["raw", "industry", "full"]` 并设 `include_float_mcap=false`，可比较未中性化、行业中性和行业加Barra中性。`size` 仅控制个股对数市值，`industry_size` 同时控制行业及市值；二者不依赖 `include_float_mcap` 开关，该开关只决定 `full` 是否额外控制市值。每个版本分别产生结果，`signal`记录最终测试值，原快照、补缺后的输入与可选平滑结果分别可追溯。
- `sample_policy="independent"` 时，各版本仅要求自身所需字段有效；原始版本不受行业、市值或 Barra 缺失、某个中性化版本失败的影响。覆盖表的 `variant_<name>_count/coverage` 记录真实测试样本，`common_count/common_coverage` 仅记录控制字段交集作为诊断。样本不同意味着版本间结果差异未必全来自中性化。
- `sample_policy="common"` 保留共同样本比较规则；原始版本也会使用全部所选控制的有效交集。旧配置省略新增选项时，仍采用周度、去极值和标准化开启、共同样本，以复现原口径。新默认示例显式开启独立样本，不改变旧冻结配置。

频率为周度／月度时，缺省年化频数分别为 52／12，Newey–West 滞后分别为 13／3 个持有期；可显式指定滞后。显式指定调仓频率时，年化参数必须与频率一致，不能只修改年化数字冒充月度调仓。

## 行业分类、EMA 与换手诊断

当前推荐配置使用 `industry.classification=aiq_zx35`，行业来源为 AIQ 历史中信多级区间，按 `FACFUNC.get_industry → zxclassified() → get_classname()` 的35类名称处理；详见 [AIQ接入说明](docs/aiq_data_integration.md) 和 [35类映射](configs/aiq_zx35_mapping.csv)。下面的 `citic_l1`、30类映射与记录更新时间代理属于保留的兼容配置，不是当前默认配置。

中信一级的 30 类映射在 [configs/citic_industry_l1.csv](configs/citic_industry_l1.csv)，同时记录中文名称、数据字段、`CI005...` 行业指数代码及官方两位一级分类码；后两者含义不同。行业适配保留原字段与映射状态，旧 `class_*`、混合类别或无法识别的值不猜测映射。设置示例：

```json
{
  "industry": {"classification": "citic_l1", "source": "industry", "column": "industry_raw"},
  "smoothing": {"method": "none", "span": 4, "min_periods": 1},
  "metrics": {"turnover_warning_threshold": 0.5}
}
```

示例的独立行业源为 `outputs/citic_industry_20261027/stock_industry_citic_l1.parquet`，原文件保持只读。原观测始于 2021-07，但存在晚于观测的补录，且 `updateTime` 未核实其时区；本次以 `max(观测日, 记录更新日)+1自然日` 作为保守可用日代理，而非声称真实首次发布时间。此前日期不倒填，实际可用范围以该来源清单及运行审计为准。独立样本中，行业缺失只影响需要行业的版本。

EMA 默认仍为 `smoothing.method=none`。改为 `ema` 后，在输入补缺之后、横截面去极值/标准化/中性化之前，按每只股票的调仓期序列递推 `s_t=αx_t+(1−α)s_(t−1)`，`α=2/(span+1)`。`span=4` 表示4个调仓期，并非统一的4天；每段从首个有效输入初始化，补缺后仍缺失或股票池断档时重置，`min_periods` 控制连续有效输入的最小数量。AIQ前填成功的因子不因原快照NaN而重置。首个研究期不隐式读取区间外数据作EMA预热，因此改变研究起点可能改变早期EMA。补缺原值与来源、`smoothed_value` 和 `factor_inputs/<因子名>.parquet` 分别记录输入和平滑处理；不能据此判断外部因子本身是否已做平滑。EMA可能降低换手并引入滞后，是否降低需实测，不保证结果。

保留的 `factor_test_citic_weekly.json`、`factor_test_citic_monthly.json` 及对应 `_ema.json` 比较 `raw/industry/size/industry_size`，其中 `_ema` 配置开启 EMA。当前 AIQ 主配置、直接因子和市值代理预设默认关闭 EMA；每份配置使用其声明的 `output_dir`。旧冻结配置及产物用于复现原口径。

```bash
OPENBLAS_NUM_THREADS=1 python3 -m fundamental_factor run --config configs/factor_test_citic_weekly.json
OPENBLAS_NUM_THREADS=1 python3 -m fundamental_factor run --config configs/factor_test_citic_weekly_ema.json
OPENBLAS_NUM_THREADS=1 python3 -m fundamental_factor run --config configs/factor_test_citic_monthly.json
OPENBLAS_NUM_THREADS=1 python3 -m fundamental_factor run --config configs/factor_test_citic_monthly_ema.json
```

组合换手已按旧持仓收益漂移计算，本次新增单边口径、均值、95%分位、超警戒比例与缺失计数。单边换手为既有双边换手的一半；持续调仓诊断排除首次实际建仓，未知漂移换手不填零。`turnover_warning_threshold=0.5` 表示单次调仓单边换手超过 50% 时标记，只是可配置诊断而非通用标准。目标换手另行展示，不替代缺失的漂移换手。

新增超额夏普为 `mean(x)/std(x, ddof=1) × sqrt(A)`：多头 `x=Q10−基准`，多空 `x=(Q10−基准)−(Q1−基准)=Q10−Q1`，不再次扣基准。采用 [Empyrical 官方 Sharpe 实现](https://github.com/quantopian/empyrical/blob/master/empyrical/stats.py) 的算术均值与样本标准差年化公式；常数或不足两期为 NA，部分缺失时披露有效期数。旧几何年化收益/波动率字段保留，两类指标不混用。资金口径和完整字段见 [本轮说明](docs/industry_ema_turnover_sharpe.md)。

## 行情代理市值与市值中性化

按用户指定公式计算 `market_cap_proxy = close * volume * 100 / (turnover / 100)`。`close` 必须是未复权收盘价，`volume` 单位为手（1 手=100 股），`turnover` 为百分数（1 表示 1%）。例如 `close=10`、`volume=1000`、`turnover=2`，代理市值为 5,000 万元。复权价格用于持有期收益；代理市值使用原始价格，不能混用。

这项数值是由成交量及换手率反推的市值代理，依赖源数据的单位与换手率分母，不等于独立核验过的实际流通市值。开启方式如下；完整数据映射见两个 `factor_test_size_proxy_*.json` 示例配置：

```json
{
  "neutralization": {
    "variants": ["raw", "size"],
    "sample_policy": "independent",
    "include_float_mcap": false,
    "market_cap": {
      "method": "quote_proxy",
      "quotes_source": "market_cap_quotes",
      "turnover_source": "quotes",
      "close_column": "close",
      "volume_column": "volume",
      "turnover_column": "turnover",
      "volume_unit": "lot",
      "turnover_unit": "percent",
      "price_basis": "unadjusted"
    }
  }
}
```

示例中的 `market_cap_quotes` 指向既有原始行情 `/home/liqinghua/FundamentalFactor/outputs/gfoptim_inputs/raw_quotes.parquet`，`quotes` 指向 `/data/mongo/quotefac_repaired.feather`，后者提供换手率。源文件只读，代理值写入运行输出。23 个日期的外部行情样本核对中，收盘价一致，成交量差异不超过 0.5 手、换手率差异不超过 0.0049 个百分点，与舍入精度一致；这只验证所查样本与单位，不能证明全历史每条记录完整、真实，也不能证明换手率分母始终等于实际流通股本。

原始收盘价、成交量和换手率须描述**同一天、同一股票**，再匹配调仓截点之前的记录。周一开盘调仓通常用上周五数据，绝不使用调仓当日完整成交信息。保留的代理市值预设采用preserve：最新记录缺失或无效时不跳过它寻找旧有效值，两次向历史匹配后要求日期相同。若显式选择aiq_matrix，则先对各源字段应用对应规则，再检查同日和有效性；不能用已过期换手率与新价格拼出有效市值。`controls.parquet` 单独记录 `market_cap_proxy`、`market_cap_date`、各来源日期、输入及无效原因，不覆盖 `float_mcap`；填充值与实际观察值分别追溯。

收盘价、成交量、换手率及最终代理值必须均为有限正数；零、负数、缺失或无穷值转为代理缺失。停牌日的零成交量不反推市值，也不填零。`size` 用截距及 `log(market_cap_proxy)` 做当期等权 OLS，残差作为测试信号；不需要行业或 Barra。`industry_size` 在此基础上加入行业；`full` 按 `include_float_mcap` 决定是否加入相同市值控制。`sample_policy=independent` 下，代理缺失只减少相应中性化版本的样本，原始因子样本和排名不受影响；报告分别展示各版本覆盖率。

省略 `market_cap` 或使用 `method=history` 时，保留原来的 `history.float_mcap` 实际流通市值接口，不自动替换成代理。仅配置代理方法而未选择任何市值版本，不为原始因子增加代理数据要求。去极值与标准化仍分别由原有开关控制；没有新增预测模型或优化器。

```bash
OPENBLAS_NUM_THREADS=1 python3 -m fundamental_factor run --config configs/factor_test_size_proxy_weekly.json
OPENBLAS_NUM_THREADS=1 python3 -m fundamental_factor run --config configs/factor_test_size_proxy_monthly.json
```

退出码 `0`：完成；`2`：核心输入或显式选择的研究要求未满足；`1`：配置或执行错误。`audit` 未通过也生成审计 HTML；`run` 在无缓存时先构建面板。默认 `data_policy.mode=classic`，未知 ST、未知停牌与缺失收益按经典统计口径披露；`strict` 保留可选的完整数据要求，`proxy` 兼容旧代理配置。核心计算通过、数据覆盖完整及执行模拟可用是分别报告的状态。

配置使用 `schema_version: 2`；旧训练配置明确报错。`configs/roic_fund.json` 与主配置同步，本次更新使用 `*_diagnostics` 等新输出目录，已有旧研究产物保留。`data_root`、`output_dir`、工作簿路径相对配置文件解析；`sources` 相对 `data_root`，均支持绝对路径。输出不得位于源数据目录中。冻结规格或数据变更后使用新的输出目录。

## 数据接口

支持 Feather v2、Parquet、CSV。日期支持 `YYYYMMDD`、ISO 日期和无时区日级时间戳；代码使用六位证券代码及 `.SH/.SZ/.BJ` 后缀。所有来源按键连接，重复键和空键会失败。

股票主表覆盖检查以行情末日与配置结束日的较早日期为上界。仅在此后才出现的因子代码不会阻断较早区间的测试；范围内缺少股票属性仍会报错。源文件的全部记录仍纳入版本指纹，不裁改输入数据。

| 来源 | 接入字段 | 何时需要 |
|---|---|---|
| 配置中的因子来源 | `date, code, <column>` | 截点之前已可用的历史快照；来源和字段由 `factors` 列表指定 |
| `quotes` | `date, code, acc_open, acc_close` | 核心输入，一致口径价格水平；可在 `price_contract` 配置列名 |
| `risk` | `date, code` 和所选风险字段 | 仅全中性使用，缺失暴露不补零 |
| `stocks` | `code, listing_date, delisting_date` | 上市年龄等已选择的股票池规则使用；保留已退市证券 |
| `history` | `date, code`，按需读取 `industry, is_st, float_mcap` | 行业/市值仅对应中性化使用；ST 是可选股票池信息 |
| `industry.source` | `date, code, <industry.column>` | 中信模式的独立历史行业来源；仅行业相关版本读取，`date` 必须是可用日期 |
| `market_cap.source` | `date, code, <market_cap.column>` | `method=history` 的直接流通市值来源，统一按元；可以独立于ST历史表，原始版本不因缺市值缩样 |
| `market_cap.quotes_source` | `date, code, close, volume`，列名可配置 | 仅启用 `quote_proxy` 市值中性时需要；未复权收盘价和以手计的成交量 |
| `market_cap.turnover_source` | `date, code, turnover`，列名可配置 | 仅启用 `quote_proxy` 市值中性时需要；百分数换手率，可与原始行情同源 |
| `calendar` | `date` | 构造周度／月度持有期与上市年龄；优先使用真实开市日历 |
| `status` | `date, code, suspended`；可附 `limit_up, limit_down` | 可选交易状态或代理；缺失状态不自动令有效价格收益失效 |
| `terminal` | `date, available_date, code, terminal_value` | 可选同价格基准的终止估值；缺失时不编造退市收益 |

严格模式布尔值使用非空 `true/false` 或 `0/1`；经典/代理模式允许 ST 和交易状态为空，按规则处理并记录。经典模式默认 `st_unknown=include`、`suspension_unknown=include`、`missing_returns=report`，不把未知状态改写成非 ST 或未停牌。已有入场停牌仍剔除；缺少停牌记录但有有效价格时，理论收益可以计算。行业分类体系须保持一致。`market_cap.method=history` 时 `float_mcap` 必须是真实流通市值；显式 `quote_proxy` 使用另存的行情代理值。不能用 `STtype` 代替ST，不能把 Barra SIZE、加工后的 `tmv`、规模分档或行情代理冒充已核验实际市值。

20项暴露固定为：`AnalystSentiment, BETA, BTOP, DividendYield, EarningQuality, EarningsYield, EarningVariability, GROWTH, INDMOM, InvestmentQuality, LEVERAGE, LIQUIDTY, LongTermReversal, MOMENTUM, Profitability, RESVOL, SEASON, SIZE, STREV, MIDCAP`。`LIQUIDTY` 沿用源文件拼写。新增源字段不会自动改变全中性口径。

`price_contract.verified/total_return` 只有核验价格处理后才能设为 true，`description` 记录实际复权口径。经典统计不要求先证明覆盖全部公司行动才允许计算价格收益；尚未验证的价格说明会作为审计提示保留，不能将其标为已验证全收益。价格非有限、非正、键冲突与时点问题仍须处理。严格模式下的 `total_return/suspended_total_return` 是用户额外选择的完整全收益校验，不是经典因子测试的通用要求。

经典/代理模式可设置 `return_kind=adjusted_price` 和 `suspension_valuation=previous_adjusted_close`。推荐配置继续使用已修复的复权价格和最近复权收盘估值，`total_return=false` 保留真实口径限制。`st_unknown=include` 不代表已完整剔除历史 ST。

缺失处理由 `missing_values.method` 选择；新推荐配置使用 `aiq_matrix`，旧配置省略时为 `preserve`：

| 数据 | 处理 |
|---|---|
| 因子 | AIQ按源字段规则补缺；普通因子字段ffill可越过显式NaN，保留原值和提供值日期。preserve保留最新记录的显式缺失。处理后仍无有效值才不进入分组 |
| 行业、市值、Barra等控制 | AIQ中默认字段ffill；行业跨空事件延续的是测试类别，不是新的真实分类，原状态保留。市值恢复原值后只填NaN，0/负/inf不越过。独立样本下控制仍缺失只影响所需版本，raw不受影响 |
| ST、停牌、涨跌停状态 | 保留未知及代理标记；按显式股票池/成交规则处理，不把未知改成正常 |
| 价格与已知退出停牌 | 按源字段规则：acc_open/acc_close默认ffill，raw open保持缺失；已知停牌另依配置使用此前复权收盘估值。两类填充各自披露，均不代表实际成交 |
| 无法恢复的价格及退市退出值 | 个股收益保持 NA，不向退市后无限延续价格，不假定兑付为零 |
| IC | 冻结信号后只用有效因子/收益配对；常数或有效配对不足时 IC 为 NA，不插值 |
| 分组与基准收益 | 推荐 `available_case`：在冻结组内以有效标签对应权重归一计算；保留原目标及缺失权重。`report` 保留旧的任一标签缺失即整组 NA；`error` 保留显式停止语义 |
| 空组 | 推荐 `empty_groups=cash`：当周整体信号存在、仅某组因并列值为空时现金收益为 0；整周无信号或非空组全缺收益仍为 NA |

AIQ矩阵规则按源字段名选择：`volume/amount/return/vwap` 首个非空值之前保持缺失、之后缺失补0；`open/high/low/turnover/free_turnover/zdtmark/fy1/CAGR` 保留缺失；特定日内 `vwap/twap_*toe` 用同日 `last`，填充源不可得时退化为ffill；其余字段默认ffill。`acc_open`前填的是它自身，不是前收盘。补缺层只填NaN，非法非空值再通过价格/市值等有效性检查；不把日收益列规则套到所有未来收益标签。原文件和已修复行情均不改写。

ST仍按事件及证据覆盖处理，过期未知不因数值矩阵补缺而延长。行业则按AIQ `mark_industry` 的前填口径提供测试类别，原始区间空缺/冲突与填充结果分别保留。省略 `missing_values` 或显式选择 `preserve`，仍采用此前“最新显式缺失阻止旧值延续”的规则。

本轮不额外调用 `getfactor()` 的横截面回归补缺：那是按中信一级行业与原始总市值拟合缺失因子的另一入口，既不等于矩阵ffill，也不等于行业/市值中性化。去极值、标准化和EMA不随补缺策略自动开启。新输出目录带 `_aiq_matrix_20261028`，原运行及冻结配置保留为历史结果。

逐项说明、AIQ接口差异及本轮核查结果见 [数据使用与缺失值核对](docs/data_missing_policy.md)。

推荐方法借鉴 [Qlib `calc_long_short_return`](https://github.com/microsoft/qlib/blob/main/qlib/contrib/eva/alpha.py)：其默认先按信号选出组内股票，再对有效 `label` 求均值。[Alphalens 的清洗器](https://github.com/stefan-jansen/alphalens-reloaded/blob/main/src/alphalens/utils.py)则在分组前删除不完整记录并披露损失比例；本框架保留冻结分组，避免未来标签改变排名。空组现金是本项目显式约定，不声称来自上述框架的统一默认值。

对于原组内目标权重 `w_i`，统计收益为 `sum(w_i*r_i, 有效标签) / sum(w_i, 有效标签)`。`holdings` 中 `target_weight` 不变，另记 `evaluation_weight`；两者不同意味着这是有效样本统计结果，并非冻结持仓的实际现金盈亏。缺失若与退市或亏损相关，仍可能存在选择偏差，需结合缺失权重判断；连续曲线不等于数据已完整。

[Qlib 持仓账本](https://github.com/microsoft/qlib/blob/main/qlib/backtest/account.py)在停牌时保留已有估值；[Zipline 历史价格接口](https://github.com/stefan-jansen/zipline-reloaded/blob/main/src/zipline/data/data_portal.py)允许价格前向延续，但会将证券有效期结束之后恢复为缺失。这类估值延续与统计均值属于不同层次；不能据此把未知退市现金当作已验证价格。

## 回测逻辑

参考本机 `GFOptim/base/basebacktest.py` 的交易计划、持仓状态和结果导出分层组织；本框架按已冻结的周度或月度开盘口径计算分组持仓、收益及漂移换手。

1. 从完整日历寻找每周／每月首个交易日；先按配置频率定位再截取区间，避免从期中起始时伪造调仓日。截点为入场前一自然日结束，持有至下一期首个交易日开盘。无训练预热和样本划分。
2. 基础股票池为配置的历史交易所范围，当前为沪深A股，含金融；剔除已知ST、上市不足120个交易日及入场停牌代理。未知 ST 由配置决定，不能与正常股票状态混淆。基准独立于因子和Barra缺失。
3. 每个因子按选定缺失策略准备后匹配截点：`aiq_matrix`按源列规则补缺，普通因子可延续此前非空值；`preserve`保留最新记录的显式缺失。两者都不使用未来值，原记录日期与提供值日期分别可追溯。独立样本按每个版本所需字段筛选，raw不受行业、市值或Barra缺失影响；共同样本额外取控制交集。纯行业不要求市值，纯市值不要求行业或Barra。
4. 若开启EMA，先对补缺后的输入按股票/调仓期因果递推，仍缺失及股票池断档重置。随后按开关去极值、标准化：中位数 ± `5×1.4826×MAD` 截尾，MAD为零改用1%/99%截尾；标准化以总体标准差计算。关闭这三项直接使用补缺后的输入；选择preserve才保留原显式缺失语义。纯行业中性取行业哑变量及截距的等权OLS残差，等价于行业内去均值；纯市值中性取截距及个股对数市值残差，行业市值中性同时控制两者，全中性再包含20项Barra。历史市值按 `market_cap.source/column` 读取并映射为 `float_mcap`，默认兼容 `history.float_mcap`；行情代理另存 `market_cap_proxy`。QR去除冗余控制列并记录；残差是否重新标准化由开关决定。未来收益不参与这些处理，中性化失败不影响独立样本下其它版本。
5. 按各版本测试信号的平均排名分位 `p=rank_average/N` 分为 `ceil(10p)`，Q1低、Q10高。同值同组，允许空组，不随机拆零值。每个非空组期初等权，冻结后连接未来收益。
6. 个股收益 `exit_price_level/entry_level−1`，价格和端点填充口径见冻结配置。标签完整时分组收益为冻结权重加权和；部分标签缺失时，推荐在原组内按有效标签对应目标权重归一计算统计均值。退市有真实同尺度终止值时接入；经典模式允许有效且未标记停牌的实际观察退出价格计算收益，仍保留 `delisting_without_terminal`。退市后的前填旧价不能充当实际观察退出价；没有观察价格/可靠结算、仅停牌旧估值或其它端点无效时标签仍缺失。非经典模式保留终止值校验。`complete`表示统计收益可计算，`data_complete`单独表示标签完整。
7. 超额为每期组合收益减独立基准，再单独复利；多空为 `r_Q10−r_Q1`，不除以2。另输出各版本自身测试样本的ALL等权组合，帮助区分因子覆盖与基准差异。报告为费前理论分组收益，披露受限交易比例。
8. 换手使用入场前漂移权重：`w_old*(1+r_stock)/(1+r_portfolio)`；对旧新持仓并集计算 `sum(abs(w_target−w_drift))`，不除以2。缺少旧持仓收益时，漂移和该期换手仍为 NA，不用统计收益补算。另列 `target_turnover`，仅比较相邻冻结目标股票权重，供观察信号变化，不能替代漂移换手。首期建仓计入，不添加末期清仓；多空换手为两腿之和。输出每只股票的目标、漂移及变动权重，包括退出股票。

推荐配置每期有有效统计收益时，从初始净值 1 连续复利；有效信号下的空组按现金收益 0 处理。非空组全缺收益、整期信号缺失仍保留真实缺口，不能为连线填收益。存在无法计算期次时，全历史累计、年化及回撤不可计算，各可用区段仅作分段展示；年化漂移换手与年化目标换手分别判断是否完整。`empty_groups=missing` 保留旧的空组 NA 行为。常数信号和少于3只股票的子样本不计算IC。真实归零与多空亏损超过本金分别记录，停止虚构后续资本重建。

## 指标定义

- 主要IC为Spearman RankIC，附Pearson。TOP20/TOP50使用 `p>.8/p>.5`，BOTTOM50/BOTTOM20使用 `p<=.5/p<=.2`；并列可能使实际比例偏离名称。
- 令年化频数 `A=52`（周度）或 `A=12`（月度）。IC均值为算术均值，波动为样本标准差，ICIR为 `mean/std×sqrt(A)`，普通t为 `mean/std×sqrt(n)`，胜率为IC>0比例；累计IC为算术累加。另附配置滞后的NW稳健t和95%区间，不压缩缺失持有期计算HAC。
- 总收益为末净值−1；年化为 `V_end^(A/T)−1`；波动为持有期收益样本std×sqrt(A)。附件“信息比率”沿用几何年化收益/年化波动，并非减无风险利率的夏普。胜率为持有期收益>0，最大回撤包含初值1、基于相应频率净值，不代表期内回撤。
- 年化双边换手为每期换手均值×A；超额绩效对逐期超额序列使用相同公式。年度结果按入场年份划分。
- 新增 `sharpe_ratio` 使用零参考收益；`excess_sharpe` 使用上述基准相对收益的算术矩，多头超额夏普亦是主动收益的信息比率。`turnover_one_way`、`target_turnover_one_way` 分别是漂移与目标双边换手的一半；持续调仓统计排除首次实际建仓，但原 `annual_turnover` 仍保留计入建仓的口径。多空名义多、空各 1 单位，其单边换手为两腿双边换手之和/2，没有再除以 2 单位毛敞口。

## 产物与复现

根目录保留 `audit.json`、冻结配置、环境版本、字段映射、覆盖度、预处理诊断、`weeks/universe/controls/features/labels.parquet`、独立基准和未解决收益清单。审计中的 `gate_passed/blocking` 表示核心计算要求，`execution_gate_passed/execution_blocking` 单独记录可选账本要求；面板的 `quality_status` 和覆盖率记录数据完整性。报告并列展示计算状态、ST 未知比例、收益覆盖和可选账本状态。预测模型评分不再输出。

中信配置另保存 `industry_mapping.csv`；EMA 开启时保存 `factor_inputs/<因子名>.parquet`，包含原始值、平滑值、连续观测数和状态重置标记。新增换手诊断和超额夏普进入组合绩效及年度表，HTML 提供多头/多空夏普摘要与逐期换手图。

新运行另提供不依赖频率的 `periods.parquet`、`ic_periodic.csv`、`portfolio_periodic.csv`。为兼容旧调用方，仍保留 `weeks.parquet`、`ic_weekly.csv`、`portfolio_weekly.csv`；月度配置下这些兼容文件也存月度记录。以冻结配置的 `rebalance_frequency` 与实际 `entry/exit` 为准，报告按实际频率显示；不能从旧文件名推断调仓频率。

每个 `factors/<name>/` 提供 `ic_weekly.csv`、`ic_summary.csv`、`portfolio_weekly.csv`、`performance.csv`、完整`holdings.parquet`、`annual_performance.csv`、摘要；原逐因子HTML地址跳转到统一报告。缺失统计单列 `data_complete`、`missing_weight`、`evaluation_weight_sum`、`cash_weight`、`return_method`；换手单列 `turnover/annual_turnover` 与 `target_turnover/annual_target_turnover`。逐股账本不超过100万行时另存CSV；大型账本仅存Parquet，避免数GB重复导出，摘要记录行数和格式。根 `report.html` 为唯一完整研究报告，包含IC、分组、年度、行情、持仓行业/风格/市值、基准覆盖和审计；`report_data.json`保留可复核数据。旧版报告生成代码已移除，详见[统一报告说明](docs/unified_report.md)。图表离线渲染，不依赖CDN。源文件和缓存使用首尾各1MiB、大小及mtime的版本指纹，明确不是全文件哈希。重复运行同一配置得出相同数值；实验时间单独记录。

分类表的 `incentive_amount` 映射仅在输出中修正为 `alternative_incentive_amount.feather`，不修改附件。明细与加工汇总值不混用，`y_5` 不作为原始收益。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -W error::FutureWarning -m unittest discover -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/synthetic_smoke.py
```

合成验收报告在 `outputs/factor_demo/run/report.html`；其中两个因子验证完整流程，显式标注模拟数据，不能作为真实因子结论。

## GFOptim 停牌及受限成交账本

新增 `fundamental_factor/execution.py`，参考 `GFOptim/base/basebacktest.py` 所调用的 `discount_strategy/backtest.py`、`accounting.py`、`corporate_actions.py`。不导入原项目或执行其数据连接。理论因子测试保留原口径，另以 `execution.enabled=true` 启用持续账本。下述原始股数规则适用于 `price_mode=raw`，复权单位代理见文末。经典模式下附加数据未就绪只跳过账本，不阻止因子统计：

1. 每个因子的所选信号版本分别建立 ALL、Q1—Q10 独立多头账户，使用与主报告相同的周度／月度首个交易日开盘调仓。旧方案的日度VWAP不混入开盘收益；独立样本模式下各账户使用所属版本的样本。
2. 当日先给昨日持仓处理显式公司行动，再用有效原始开盘价估值。停牌、缺失报价采用上次估值；这只是估值，不能当作成交价。公司行动发生而无新报价时，上次标价调整为 `(旧标价−每旧股现金分红)/(1+送转比例)`，同时增加现金与股数，防止分红重复增厚净值。除息日新买入不领取旧股分红。
3. 目标为冻结分组的等权持仓。交易集合包括目标股和旧持仓；股票离开当期信号池不会从账本消失。停牌双向禁止，涨停禁止买入，跌停禁止卖出；未知状态或未知方向限价同样禁止对应订单。
4. 先执行允许的卖单，再按实际可用现金等比例缩减买单，现金不得为负。未卖出的仓位留下；未成交资金留为现金。下次调仓重新计算订单，日内及持有期中不重试。整期信号缺失时保持原股数/现金，不制造清仓择时；有效信号的某个空组才以现金为目标。
5. 每日以有效原始收盘价估值，否则延续已知价格并记录原因、最后真实报价日期。下一入场日交易前资产作为前一期结束资产。终止事件按可知结算日将每当前股现金兑付并移除持仓，不能把复权终止水平当作现金。

执行账本允许分数股，不含费税、冲击、借券、手数及现金利息；分红在除息事件日按现金入账，未模拟实际派息到账日差异。这是成交限制敏感性分析，仍不等于完整可实现策略。Q10和Q1是独立多头账户，不生成声称可融券实现的执行多空。

| 附加来源 | 字段 | 约定 |
|---|---|---|
| `execution_quotes` | `date, code, open, close` | 未复权价格，含已退出因子池的历史持仓 |
| `execution_status` | `date, code, suspended, limit_up, limit_down` | 日级状态，涨跌停指开盘方向限制；空值禁止对应交易 |
| `corporate_actions` | `date, code, share_ratio, cash_per_share, notice_date` | 生效/除息日、每旧股送转比例/税前现金；当日或此前已公布；无事件可为空表但必须确认覆盖完整 |
| `execution_terminal` | `date, code, cash_per_share, available_date` | 每代码一次最终现金结算；date为结算日且不早于退市日，available_date不晚于结算日；零值有效 |

`execution` 配置还包括 `initial_capital`、`raw_prices_verified`、`status_verified`、`corporate_actions_complete`、`coverage_start/end` 和 `description`。覆盖必须包含最后持有结束日。配股、合并、特殊重组及其他未建模公司行动必须先解决；不能只因已有分红CSV就确认全样本事件完整。

输出为每因子目录的 `execution_weekly.csv`、`execution_performance.csv`、`execution_holdings.parquet`、`execution_trades.parquet`、`execution_daily.parquet`。逐股保留目标/实际权重、股数、估值来源，订单保留申请/实际成交与拒绝原因；报告附实际净值、现金权重、受阻金额、延续估值及缺信号冻结次数。真实数据未就绪时，可运行独立合成演示：

```bash
OPENBLAS_NUM_THREADS=1 python3 scripts/synthetic_smoke.py --with-execution --output-root outputs/execution_demo
```

## GFOptim 数据复用与历史审计记录（2026-10-13）

本节保留最初核查记录；2026-10-14已授权代理并完成的修复见下节，新配置不会沿用这里的旧阻断结论。

```bash
OPENBLAS_NUM_THREADS=1 python3 scripts/prepare_gfoptim_inputs.py
python3 -m fundamental_factor audit --config configs/factor_test_gfoptim.json
```

适配器只读 `/home/liqinghua/data/quote.feather`、`quote_2024-2026.feather`、`combined_exp.feather` 及原策略显式公司行动文件。按日期/代码优先采用近期源的**原始**行情，不拼接复权因子。旧源中的非标准代码 `T00018.SH` 单列隔离清单，证券主表仍须明确历史A股资格。

可查看 `outputs/gfoptim_inputs/data_requirements.html`、`reference_inventory.json`、`adapter_manifest.json` 和 `price_anomalies.csv`。适配器导出原始行情、成交量停牌代理、候选日历、近段行业及不完整公司行动；`configs/factor_test_gfoptim.json` 指向这些产物，但保留未通过的数据门槛。

已核实旧/新原始行情重叠3,121,636行，open/close/last/volume一致。停牌可按原项目 `volume==0 or close==0` 推导；全天成交量是事后观测，不能未经确认便宣称开盘前已知状态。缺失行情状态仍未知。涨跌停不照搬固定板块百分比，以免忽略历史ST、IPO例外和制度变化。

重要价格缺陷：`000001.SZ` 在2025-05-27至05-28原始收盘11.49→11.53，旧行情与 `/data/mongo/quotefac.feather` 的复权收盘1468.2433→904.4114，产生约−38.4%的异常变化。该个案已在下节说明的独立版本中修复，原文件保持原样。旧记录中的完整全收益门槛不用于当前经典统计配置。

2026-10-13复核后，原先缺项清单修订如下：

- 日历已补：新浪响应与2005-12-29—2026-08-31本地行情的5,021个日期全部一致，已保存为`/data/mongo/public_trade_calendar_sina.feather`。
- 上市/终止日期已从沪深交易所取得，5,558代码，覆盖现有核心因子、风险和行情的全部沪深代码；主表为`public_stock_master_sh_sz.feather`。特殊重新上市及代码沿革仍须按具体样本核验。
- 深圳7,341条官方简称变更已保存为`public_name_change_szse.feather`，只保留事件前后ST名称标记；不把没有事件的区间补成非ST。上海与完整区间ST状态仍未补齐。GFOptim调用链中的`stop`是停牌，`risk`是风险暴露缺失，不是ST。
- `on_all_temp.feather/on_all_nn.feather` 中已找到2013—2026的`class`行业候选，1—34与近段行业逐类吻合，35为混合类别；尚需确认分类定义和可用时点。不能再称早期行业完全缺失。当前纯行业模式不需要实际市值。
- 历史逐股涨跌停价及限制例外、停牌代理核验、完整公司行动和复权异常修复仍有缺项。API存在不代表历史覆盖完整，未验证的部分未保存为正式输入。
- 官方退市日期保存在`public_stock_delisting_sh_sz.feather`，不等于现金清算。只对实际持有至退市的股票要求验证后续估值或退出事件；已提前卖出的股票不再触发执行账本的现金结算门槛。
- 核心ROIC、Barra和理论行情均无北交所。新配置明确采用沪深A股，因此北交所不构成本次缺项；若扩展沪深北全A，则需要补完整的北交所因子、行情及控制信息，不只是尾段行情。

新增公共文件使用现有Feather格式，`date`为int64 YYYYMMDD、`code`为六位代码加交易所后缀。`/data/mongo/public_reference_manifest.json` 保存来源URL、原始响应和产物SHA256、覆盖边界；原始HTTP响应另保存在项目`outputs/public_data_review/raw/`。未覆盖或未知内容不编造、不补零，已有源文件未覆盖。

## 2026-10-14 代理接入、ST历史与价格修复

新增文件保存在 `/data/mongo/`，原文件不覆盖：

| 文件 | 用途与真实性边界 |
|---|---|
| `quotefac_repaired.feather` | 保留原字段和所有行，只对已证实断点统一重标度；主价格输入 |
| `quotefac_repair_patches.feather` / `quotefac_repair_manifest.json` | 逐行修改、原文件全SHA256、多源比较及公开证据 |
| `public_name_change_szse_validated.feather` | 当天刷新深交所7,343条真实简称事件，修正GST/G*ST识别 |
| `public_st_history_szse.feather` | 由官方变更后简称得到的历史ST；首次事件前未知，含截止日后的失效哨兵 |
| `public_st_history_manifest.json` | 官方URL、HTTP状态、原响应与产物SHA256、覆盖及失败API记录 |
| `stock_history_proxy.feather` | 日期/代码/原生class行业类别及已知ST；未知ST保持nullable |
| `stock_trade_status_proxy.feather` | 停牌、开盘涨跌停代理，未知与方法字段单列 |

价格修复核对旧、新、Mongo三本地源及东方财富、新浪公开响应。`000001.SZ`在2025-05-28之前的3,008行全部复权价格乘0.6138449932017157；常数重标度不改变边界前区间收益。异常日复权收盘收益由−38.401803%改为+0.348130%。其他9个正向特殊事件不能据机械公式判错，保留原值。12,640条/57股没有原始行情对应键，留在缺口清单，不伪造停牌或成交状态。

官方历史名称在34,224个已知深圳抽样记录中与本地`secShortName`有1,465条冲突，因此不用当前名称倒填历史。上海完整历史ST尚未取得，BaoStock公共连接两次超时，未编造数据。原生`class`两汇总源11,632,446行完全一致；只复用分类，不复用其中经过加工的因子值。

涨跌停用供应商`last`除权参考价，不直接拿昨日原始收盘；按分四舍五入，并加入创业板2020-08-24及主板ST2026-07-06规则边界。未知ST使用普通板块比例作限价近似，首次上市5个交易日保持限价未知。复牌、重组等特殊例外仍未完整覆盖，不宣称真实逐股历史限价。

```bash
python3 scripts/prepare_historical_st.py --help
python3 scripts/prepare_price_repair.py --help
python3 scripts/prepare_proxy_inputs.py
OPENBLAS_NUM_THREADS=1 python3 -m fundamental_factor run --config configs/factor_test_execution_proxy.json
```

`configs/factor_test_execution_proxy.json`保留2025年上半年原代理账本配置；本次经典口径复跑使用 `configs/factor_test_execution_classic.json`，输出 `outputs/factor_test_execution_classic/`。`execution.price_mode=adjusted_units`在经典/代理模式下允许使用复权单位，不重复加入原始股数的公司行动和退出现金；上述两个来源必须设null。卖不出保留旧仓，买不进保留现金，未知方向限制禁止成交；实际持有跨退市且退出价值未知的账户从受影响周起停止收益展示与交易。表中`adjusted_units`不是实际股数，`indicative_nav`只是缺口追溯值。原始股数账本仍可用`price_mode=raw`。
