# v0.3.0 AIQ字段补缺验收

验证日期：2026-10-28。推荐配置统一为 `missing_values.method=aiq_matrix`，按AIQ字段规则应用zero、keep、ffill和last；保留原值、原始缺失、填充标记和真实供值日。原始AIQ文件和已修复收益行情只读。

## 方法和版本

- 普通因子、市值和复权开盘等按字段前填；raw open/high/low、turnover等保持缺失；volume等在首个非空值后补0；日内指定字段使用last。
- `acc_open` 沿用自身历史开盘，不换成前收盘。字段只对NaN补缺，0、负值、无穷不被当作可跳过的NaN；有效性检查在其后。
- 行业35类严格沿用 `FACFUNC.get_industry → zxclassified() → get_classname()`；分类前填与原始区间未知分别披露。
- ST执行/取消及覆盖到期仍按事件处理；不把其未知状态通用填成正常。原始数据缺失不等于真实停牌事实，零成交量只是研究代理。
- `getfactor()` 的行业/总市值横截面回归补缺不与矩阵规则叠加。EMA、去极值、标准化、中性化仍为独立设置。
- 首个有效值之前不补，禁止未来供值；上市前供值、退市后缺实际退出观察的延续值不能生成有效结算收益。
- 显式选择preserve或旧配置省略missing_values保留旧面板语义；复现旧结果还须使用旧冻结配置及原状态来源。

## 真实数据结果

本轮测试ROIC与公募持仓变化两个成品因子，每个均有raw、industry、size、industry_size四版本。EMA、去极值及标准化关闭；各版本独立取样。完整持有期不代表每个个股收益标签完整。

| 项目 | 周度 | 月度 |
|---|---:|---:|
| 完整持有期 | 697 | 162 |
| 起点—最后退出 | 2013-01-07—2026-08-17 | 2013-02-01—2026-08-03 |
| 基础股期次 | 2,473,584 | 575,006 |
| ROIC原始缺失 / 已前填 | 59,370 / 25 | 13,691 / 3 |
| 持仓变化原始缺失 / 已前填 | 114,970 / 4,041 | 27,263 / 1,504 |
| 行业已前填 / 仍未知 | 236 / 10 | 221 / 6 |
| 市值已前填 | 3 | 0 |
| 开盘填充端点：入场 / 退出 | 39 / 242 | 3 / 170 |
| 未解决退市相关收益 | 148 | 126 |

开盘填充数包含随后因生命周期被拒绝的值，不能当作合法收益观察数。周度94个、月度44个合法填充退出端点逐一匹配历史acc_open；被拒绝的端点未生成收益。原始缺失及无效标签在报告中保留，统计曲线按已声明的有效收益样本口径计算。

AIQ前填不设最大年龄；3条周度市值填充的供值距目标约2、10、14个月。验收已逐项核对供值记录，日期完整保留，不能把它们解释为新的当期观察。

交易状态覆盖12,781,525个股票日，其中13,648个原无行情键日按规则处理：close/last前填、volume补0、open保持缺失。原始缺行、填充及来源日可追溯。代理可计算不代表已获得完整真实停复牌和开盘限制。

本轮与此前preserve结果的股票池、字段处理和样本均可能不同，不能把收益变化归因于单个处理步骤，也未据结果翻转因子方向。

## 验证证据

- 全量自动测试311项通过，另有25项报告/集成重复回归通过；纯矩阵补缺与隔离的AIQ原函数进行120组随机对照。
- 新状态全表逐行核验，无未来/上市前供值或生命周期外记录。
- 周度19,107,538条、月度4,442,238条特征记录逐批验收，无未来供值、原有效值被补缺覆盖或行业供值元数据错配。
- 原始因子填充值逐项匹配源文件供值；开盘填充值逐项匹配受保护的历史acc_open。
- 两因子四版本的所有组合分别有697/162期统计收益及净值。真实未解决的个股收益仍披露，不因曲线连续而称数据完整。
- 已核对来源及保护行情指纹，未改AIQ数据、旧准备产物或已修复行情。大数据指纹采用首尾抽样、大小和mtime，不宣称全文件逐字节证明。

## 复现与产物

```bash
python3 scripts/prepare_aiq_inputs.py
python3 scripts/prepare_aiq_matrix_inputs.py
OPENBLAS_NUM_THREADS=1 python3 -m fundamental_factor run --config configs/factor_test_aiq_weekly.json
OPENBLAS_NUM_THREADS=1 python3 -m fundamental_factor run --config configs/factor_test_aiq_monthly.json
python3 scripts/report_aiq_matrix_validation.py --with-baselines
```

本地产物位于 `outputs/aiq_matrix_validation_20261028/`：`report.html`、`summary.json`、逐频率验收JSON及填充/策略摘要CSV。两份完整报告位于 `outputs/factor_test_weekly_aiq_matrix_20261028/report.html` 和 `outputs/factor_test_monthly_aiq_matrix_20261028/report.html`。原始数据及本地大面板未纳入Git。
