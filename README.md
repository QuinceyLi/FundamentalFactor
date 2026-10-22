# 单因子周度回测框架

对照附件测试报告，分别测试 `ROIC_new` 和 `fundnum_promotion`。默认每个因子生成原始信号、中性化、全中性三套结果，也可按配置选择其中的版本。输出包含覆盖度、RankIC/Pearson IC、十分组净值、超额净值、Q10−Q1、年度表现和双边换手。**不训练预测模型，也不使用持仓优化器**；理论组合直接使用冻结的组内等权，执行账本仅落实权重及成交约束。配置中的因子列表可以扩展到其他明细文件。

当前推荐 `configs/factor_test_classic.json`，产物写入 `outputs/factor_test_classic/`：明确使用沪深A股，`neutralization.include_float_mcap=false`，输出原始、纯行业中性、行业加20项Barra中性三套结果。纯行业中性不需要行业市值或个股市值。历史行业采用现有 `class` 分类控制，保留类别35的混合性质。`neutralization.variants` 可只选 `raw`，此时不要求行业、市值或 Barra 暴露。旧配置及冻结产物保留。

经典模式把核心因子统计与可选筛选、执行数据分开：完整历史 ST、完整停复牌/涨跌停例外、公司行动明细及真实退市现金不作为统一门槛。已有官方 ST 历史和 GF 停牌/涨跌停代理继续使用；未覆盖状态保持未知。缺失标签仅影响对应 IC 配对或持仓组合，未受影响结果继续计算。可选执行账本未满足附加记账要求时单独跳过，并披露原因。此前2026-10-13的数据报告属于历史审计记录。

这一输入分层参考 [Alphalens 的因子与未来收益接口](https://github.com/quantopian/alphalens/blob/master/alphalens/utils.py)、[Qlib 的因子评价接口](https://github.com/microsoft/qlib/blob/main/qlib/contrib/eva/alpha.py)以及[独立交易模块](https://github.com/microsoft/qlib/blob/main/qlib/backtest/exchange.py)。冻结信号后不因缺失未来收益而重分组，是本项目保留的研究约定；不声称与这些框架的所有默认处理逐项相同。

## 运行

Python ≥3.10，依赖 pandas、PyArrow、NumPy、SciPy、statsmodels、openpyxl。当前机器的 `/usr/bin/python3` 已有依赖；若 pyenv 默认解释器缺包，将下列命令中的 `python3` 替换为 `/usr/bin/python3`。其他环境可用 `python3 -m pip install -e .` 安装。

```bash
python3 -m fundamental_factor audit --config configs/factor_test_classic.json
python3 -m fundamental_factor build-panel --config configs/factor_test_classic.json
OPENBLAS_NUM_THREADS=1 python3 -m fundamental_factor run --config configs/factor_test_classic.json
python3 -m fundamental_factor report --run-dir outputs/factor_test_classic
```

退出码 `0`：完成；`2`：核心输入或显式选择的研究要求未满足；`1`：配置或执行错误。`audit` 未通过也生成审计 HTML；`run` 在无缓存时先构建面板。默认 `data_policy.mode=classic`，未知 ST、未知停牌与缺失收益按经典统计口径披露；`strict` 保留可选的完整数据要求，`proxy` 兼容旧代理配置。核心计算通过、数据覆盖完整及执行模拟可用是分别报告的状态。

配置使用 `schema_version: 2`；旧训练配置明确报错。`configs/roic_fund.json` 同步更新为新版配置，输出位于新的 `outputs/factor_test`，已有旧研究产物保留。`data_root`、`output_dir`、工作簿路径相对配置文件解析；`sources` 相对 `data_root`，均支持绝对路径。输出不得位于源数据目录中。冻结规格或数据变更后使用新的输出目录。

## 数据接口

支持 Feather v2、Parquet、CSV。日期支持 `YYYYMMDD`、ISO 日期和无时区日级时间戳；代码使用六位证券代码及 `.SH/.SZ/.BJ` 后缀。所有来源按键连接，重复键和空键会失败。

| 来源 | 接入字段 | 何时需要 |
|---|---|---|
| 配置中的因子来源 | `date, code, <column>` | 截点之前已可用的历史快照；来源和字段由 `factors` 列表指定 |
| `quotes` | `date, code, acc_open, acc_close` | 核心输入，一致口径价格水平；可在 `price_contract` 配置列名 |
| `risk` | `date, code` 和所选风险字段 | 仅全中性使用，缺失暴露不补零 |
| `stocks` | `code, listing_date, delisting_date` | 上市年龄等已选择的股票池规则使用；保留已退市证券 |
| `history` | `date, code`，按需读取 `industry, is_st, float_mcap` | 行业/市值仅对应中性化使用；ST 是可选股票池信息 |
| `calendar` | `date` | 构造周界与上市年龄；优先使用真实开市日历 |
| `status` | `date, code, suspended`；可附 `limit_up, limit_down` | 可选交易状态或代理；缺失状态不自动令有效价格收益失效 |
| `terminal` | `date, available_date, code, terminal_value` | 可选同价格基准的终止估值；缺失时不编造退市收益 |

严格模式布尔值使用非空 `true/false` 或 `0/1`；经典/代理模式允许 ST 和交易状态为空，按规则处理并记录。经典模式默认 `st_unknown=include`、`suspension_unknown=include`、`missing_returns=report`，不把未知状态改写成非 ST 或未停牌。已有入场停牌仍剔除；缺少停牌记录但有有效价格时，理论收益可以计算。行业分类体系须保持一致；启用市值中性时 `float_mcap` 必须是真实市值。不能用 `STtype` 代替ST，不能用 Barra SIZE 或规模分档代替实际市值。

20项暴露固定为：`AnalystSentiment, BETA, BTOP, DividendYield, EarningQuality, EarningsYield, EarningVariability, GROWTH, INDMOM, InvestmentQuality, LEVERAGE, LIQUIDTY, LongTermReversal, MOMENTUM, Profitability, RESVOL, SEASON, SIZE, STREV, MIDCAP`。`LIQUIDTY` 沿用源文件拼写。新增源字段不会自动改变全中性口径。

`price_contract.verified/total_return` 只有核验价格处理后才能设为 true，`description` 记录实际复权口径。经典统计不要求先证明覆盖全部公司行动才允许计算价格收益；尚未验证的价格说明会作为审计提示保留，不能将其标为已验证全收益。价格非有限、非正、键冲突与时点问题仍须处理。严格模式下的 `total_return/suspended_total_return` 是用户额外选择的完整全收益校验，不是经典因子测试的通用要求。

经典/代理模式可设置 `return_kind=adjusted_price` 和 `suspension_valuation=previous_adjusted_close`。推荐配置继续使用已修复的复权价格和最近复权收盘估值，`total_return=false` 保留真实口径限制。`missing_returns=report` 时 IC 在已冻结样本中仅使用有效收益对，输出标签覆盖率；任一持仓标签缺失的组当周收益为 NA，不能删除该股票后重新等权。`st_unknown=include` 不代表已完整剔除历史 ST。

## 回测逻辑

参考本机 `GFOptim/base/basebacktest.py` 的交易计划、持仓状态和结果导出分层组织；本框架按已冻结的周度开盘口径计算分组持仓、收益及漂移换手。

1. 从完整官方日历寻找每周首个交易日；先按周定位再截取区间，避免从周中起始时伪造调仓日。截点为入场前一自然日结束，持有至下周首个交易日开盘。无训练预热和样本划分。
2. 基础股票池为配置的历史交易所范围，当前为沪深A股，含金融；剔除已知ST、上市不足120个交易日及入场停牌代理。未知 ST 由配置决定，不能与正常股票状态混淆。基准独立于因子和Barra缺失。
3. 每个因子分别取历史最近快照，保留显式缺失更新，不跨过缺失记录寻找旧非空值。所选信号使用其需要的控制字段均可用的共同样本；纯行业模式不要求、筛选或生成实际市值。共同样本是版本比较约定，单独选择原始等权分组与IC不需要行业、市值或Barra。
4. 原始处理采用中位数 ± `5×1.4826×MAD` 截尾，MAD为零改用1%/99%截尾，再以总体标准差标准化。纯行业中性取行业哑变量及截距的等权OLS残差，等价于行业内去均值；可选市值模式另加个股对数流通市值。全中性再包含20项Barra。QR去除冗余控制列并记录，不删除股票；残差重新标准化。未来收益不参与这些处理。
5. 按处理后信号的平均排名分位 `p=rank_average/N` 分为 `ceil(10p)`，Q1低、Q10高。同值同组，允许空组，不随机拆零值。每个非空组周初等权，冻结后连接未来收益。
6. 个股收益 `exit_price_level/entry_level−1`，价格口径见冻结配置。分组收益为冻结权重加权和。退市有真实同尺度终止值时接入，否则保留缺失；经典/代理模式继续输出有效IC和未受影响组合，缺失组保持NA；仅显式严格模式停止。
7. 超额为每周组合收益减独立基准，再单独复利；多空为 `r_Q10−r_Q1`，不除以2。另输出共同样本的ALL等权组合，帮助区分因子覆盖与基准差异。报告为费前理论分组收益，披露受限交易比例。
8. 换手使用入场前漂移权重：`w_old*(1+r_stock)/(1+r_portfolio)`；对旧新持仓并集计算 `sum(abs(w_target−w_drift))`，不除以2。首期建仓计入，不添加末期清仓；多空换手为两腿之和。输出每只股票的目标、漂移及变动权重，包括退出股票。

空组收益为NA，净值断线；新连续区段从1展示，不跨断点漂移或拼接。全历史不完整时，累计、年化、回撤和年化换手不可计算；有效周数、波动和胜率仍可列示。常数信号和少于3只股票的子样本不计算IC。真实归零与多空亏损超过本金分别记录，停止虚构后续资本重建。

## 指标定义

- 主要IC为Spearman RankIC，附Pearson。TOP20/TOP50使用 `p>.8/p>.5`，BOTTOM50/BOTTOM20使用 `p<=.5/p<=.2`；并列可能使实际比例偏离名称。
- IC均值为算术均值，波动为样本标准差，ICIR为 `mean/std×sqrt(52)`，普通t为 `mean/std×sqrt(n)`，胜率为IC>0比例；累计IC为算术累加。另附NW13稳健t和95%区间，不压缩缺失周计算HAC。
- 总收益为末净值−1；年化为 `V_end^(52/T)−1`；波动为周收益样本std×sqrt(52)。附件“信息比率”沿用几何年化收益/年化波动，并非减无风险利率的夏普。胜率为周收益>0，最大回撤包含初值1、基于周度净值，不代表周内回撤。
- 年化双边换手为周换手均值×52；超额绩效对逐周超额序列使用相同公式。年度结果按入场年份划分。

## 产物与复现

根目录保留 `audit.json`、冻结配置、环境版本、字段映射、覆盖度、预处理诊断、`weeks/universe/controls/features/labels.parquet`、独立基准和未解决收益清单。审计中的 `gate_passed/blocking` 表示核心计算要求，`execution_gate_passed/execution_blocking` 单独记录可选账本要求；面板的 `quality_status` 和覆盖率记录数据完整性。报告并列展示计算状态、ST 未知比例、收益覆盖和可选账本状态。预测模型评分不再输出。

每个 `factors/<name>/` 提供 `ic_weekly.csv`、`ic_summary.csv`、`portfolio_weekly.csv`、`performance.csv`、完整`holdings.parquet`、`annual_performance.csv`、摘要和独立HTML。逐股账本不超过100万行时另存CSV；大型账本仅存Parquet，避免数GB重复导出，摘要记录行数和格式。根 `report.html` 为汇总索引及审计。图表离线渲染，不依赖CDN。源文件和缓存使用首尾各1MiB、大小及mtime的版本指纹，明确不是全文件哈希。重复运行同一配置得出相同数值；实验时间单独记录。

分类表的 `incentive_amount` 映射仅在输出中修正为 `alternative_incentive_amount.feather`，不修改附件。明细与加工汇总值不混用，`y_5` 不作为原始收益。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -W error::FutureWarning -m unittest discover -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/synthetic_smoke.py
```

合成验收报告在 `outputs/factor_demo/run/report.html`；其两个因子报告验证完整流程，显式标注模拟数据，不能作为真实因子结论。

## GFOptim 停牌及受限成交账本

新增 `fundamental_factor/execution.py`，参考 `GFOptim/base/basebacktest.py` 所调用的 `discount_strategy/backtest.py`、`accounting.py`、`corporate_actions.py`。不导入原项目或执行其数据连接。理论因子测试保留原口径，另以 `execution.enabled=true` 启用持续账本。下述原始股数规则适用于 `price_mode=raw`，复权单位代理见文末。经典模式下附加数据未就绪只跳过账本，不阻止因子统计：

1. 每个因子的所选信号版本分别建立 ALL、Q1—Q10 独立多头账户。沿用周度首个交易日开盘调仓，旧方案的日度VWAP不混入周度开盘收益。
2. 当日先给昨日持仓处理显式公司行动，再用有效原始开盘价估值。停牌、缺失报价采用上次估值；这只是估值，不能当作成交价。公司行动发生而无新报价时，上次标价调整为 `(旧标价−每旧股现金分红)/(1+送转比例)`，同时增加现金与股数，防止分红重复增厚净值。除息日新买入不领取旧股分红。
3. 目标为冻结分组的等权持仓。交易集合包括目标股和旧持仓；股票离开当期信号池不会从账本消失。停牌双向禁止，涨停禁止买入，跌停禁止卖出；未知状态或未知方向限价同样禁止对应订单。
4. 先执行允许的卖单，再按实际可用现金等比例缩减买单，现金不得为负。未卖出的仓位留下；未成交资金留为现金。下个周度调仓重新计算订单，日内及周中不重试。信号整周缺失时保持原股数/现金，不制造清仓择时；有效信号的某个空组才以现金为目标。
5. 每日以有效原始收盘价估值，否则延续已知价格并记录原因、最后真实报价日期。下一入场日交易前资产作为前一周结束资产。终止事件按可知结算日将每当前股现金兑付并移除持仓，不能把复权终止水平当作现金。

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
