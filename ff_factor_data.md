# 基本面因子分类数据

`classifications` 为有本地同名数据字段的因子；`unimplemented_factors` 收录停用项及尚无同名字段的旧标签。B/C 编码和名称沿用 `~/data/factor_data.md` 的既有分类体系；旧标签 `f1`–`f5` 缺少机制定义，B/C 暂置为 `null`。因子编号独立于参考文件。

分类方法参考 [XALPHA 论文](</home/liqinghua/.codex/attachments/4ef709b6-9f0c-4a81-bbcc-7924d8926b03/XALPHA A Memory-Driven AI Quant Researcher for(2).pdf>) 第 58–59 页：依据实际主导机制先选 B 层，再在该 B 层下选 C 层。论文第 5 页及附录 E 使用的是 OHLCV 机制体系；本文件沿用的“财务基本面”等类别属于原 `factor_data.md` 的本地扩展。分类依据中的原目录先例与推断范围均逐项说明。

`f1_new`–`f7_new` 的回归口径由用户确认：最近八个季度（两年）、逐股回归、输出当期残差。公式以含当期的 `W_q={q-7,…,q}` 表达该窗口，以 `m_hat` 表示源实现的拟合回归函数，不额外指定 OLS、截距或后续标准化。分类表和本地预计算文件未给出单季/TTM、估计方法、最少有效样本及缺失处理；`Delta` 表示源实现定义的变化量，其环比/同比或其他差分间隔尚未核验。`f7_new` 的工资字段也未确认等同于 `salary_avg`。两年指回归窗口，不自动作为变化量的差分间隔。

来源：《因子分类.xlsx》、`/data/mongo/feature_type.csv` 和 `/data/mongo` 下的 Feather 明细表。已核对启用因子的 `date`、`code` 与同名字段；`incentive_amount` 的实际来源为 `alternative_incentive_amount.feather`。`f1_new`–`f7_new` 的 `initial_ic` 按参考文件明示的日期、10 交易日和最少 5 只股票参数，使用本地复权收盘价计算；详细口径见 `outputs/ff_ic_2019_2024/method.json`，与参考文件的 DolphinDB 结果尚未核实可比。旧名与 `_new` 字段的对应关系待确认。

```json
{
  "classifications": [
    {
      "factor_id": "factor_001",
      "b_id": "B6",
      "b_name": "另类、分析师与股东因子",
      "c_id": "C6-2",
      "c_name": "分析师预期因子",
      "classification_basis": "原表列为“一致预期／预期变动”；信号来自分析师盈利预测的加权变化，主导机制为预期修正，与原目录“加权的盈余调整幅度”同属分析师预期因子。",
      "factor_name": "cpp_weighted",
      "factor_desc": "数据表：forecast_cpp.feather。",
      "formula": "读取 forecast_cpp.feather 中的 cpp_weighted 预计算字段，按 date、code 对齐；原表未提供源因子的计算公式。"
    },
    {
      "factor_id": "factor_002",
      "b_id": "B6",
      "b_name": "另类、分析师与股东因子",
      "c_id": "C6-2",
      "c_name": "分析师预期因子",
      "classification_basis": "原表列为“一致预期／预期分歧”；刻画盈利预测的分歧，与原目录“预期分歧度”同属分析师预期因子；具体离散度计算口径仍以源实现为准。",
      "factor_name": "profit_var",
      "factor_desc": "数据表：forecast_cpp.feather。",
      "formula": "读取 forecast_cpp.feather 中的 profit_var 预计算字段，按 date、code 对齐；原表未提供源因子的计算公式。"
    },
    {
      "factor_id": "factor_003",
      "b_id": "B6",
      "b_name": "另类、分析师与股东因子",
      "c_id": "C6-3",
      "c_name": "股东与股权因子",
      "classification_basis": "原表列为“事件驱动／股权激励”；股权激励通过管理层与股东的利益联系反映治理和激励机制，参照原目录“公司治理指数”归入股东与股权因子。",
      "factor_name": "incentive_amount",
      "factor_desc": "数据表：alternative_incentive_amount.feather；附件表名：alternative_research_num.feather（本地字段核对后更正）。",
      "formula": "读取 alternative_incentive_amount.feather 中的 incentive_amount 预计算字段，按 date、code 对齐；原表未提供源因子的计算公式。"
    },
    {
      "factor_id": "factor_004",
      "b_id": "B3",
      "b_name": "财务基本面因子",
      "c_id": "C3-2",
      "c_name": "成长因子",
      "classification_basis": "原表说明为“净利润断层”，据此暂按财务业绩的突变归入成长因子。若源实现实际衡量公告后的股价跳空或相对指数回报，应参照原目录“盈余公告回报”改归 C4-1；当前无生成代码确认该细节。",
      "factor_name": "jump_zz500",
      "factor_desc": "数据表：fs_jump.feather；原表逻辑：净利润断层。",
      "formula": "读取 fs_jump.feather 中的 jump_zz500 预计算字段，按 date、code 对齐；原表未提供源因子的计算公式。"
    },
    {
      "factor_id": "factor_005",
      "b_id": "B6",
      "b_name": "另类、分析师与股东因子",
      "c_id": "C6-1",
      "c_name": "另类外部数据因子",
      "classification_basis": "原表说明为“调研机构数量”；以机构调研记录刻画外部关注度，属于另类外部数据机制，参照原目录关注度与新闻类指标归入另类外部数据因子。",
      "factor_name": "research_num",
      "factor_desc": "数据表：alternative_research_num.feather；原表逻辑：调研机构数量。",
      "formula": "读取 alternative_research_num.feather 中的 research_num 预计算字段，按 date、code 对齐；原表未提供源因子的计算公式。"
    },
    {
      "factor_id": "factor_006",
      "b_id": "B6",
      "b_name": "另类、分析师与股东因子",
      "c_id": "C6-3",
      "c_name": "股东与股权因子",
      "classification_basis": "原表说明为“大股东增持”；直接刻画股东持股变动及增持行为，与原目录股东户数、机构持股等同属股东与股权因子。",
      "factor_name": "trade_pos_amount",
      "factor_desc": "数据表：alternative_holdingchg.feather；原表逻辑：大股东增持。",
      "formula": "读取 alternative_holdingchg.feather 中的 trade_pos_amount 预计算字段，按 date、code 对齐；原表未提供源因子的计算公式。"
    },
    {
      "factor_id": "factor_007",
      "b_id": "B3",
      "b_name": "财务基本面因子",
      "c_id": "C3-2",
      "c_name": "成长因子",
      "classification_basis": "原表说明为“预增扭亏”；核心信息是公司业绩预告所反映的盈利增长或由亏转盈，按财务业绩变化归入成长因子；资料未说明分析师预期比较机制。",
      "factor_name": "yznk_ratio",
      "factor_desc": "数据表：alternative_yznk_ratio.feather；原表逻辑：预增扭亏。",
      "formula": "读取 alternative_yznk_ratio.feather 中的 yznk_ratio 预计算字段，按 date、code 对齐；原表未提供源因子的计算公式。"
    },
    {
      "factor_id": "factor_008",
      "b_id": "B3",
      "b_name": "财务基本面因子",
      "c_id": "C3-3",
      "c_name": "估值 & 杠杆因子",
      "classification_basis": "原表列为“估值／相对估值”；核心机制是相对价格与财务价值的比较，归入原目录“估值 & 杠杆因子”，具体估值基准仍待源公式确认。",
      "factor_name": "Value_relative",
      "factor_desc": "数据表：fs_zx.feather。",
      "formula": "读取 fs_zx.feather 中的 Value_relative 预计算字段，按 date、code 对齐；原表未提供源因子的计算公式。"
    },
    {
      "factor_id": "factor_009",
      "b_id": "B6",
      "b_name": "另类、分析师与股东因子",
      "c_id": "C6-3",
      "c_name": "股东与股权因子",
      "classification_basis": "原表说明为“公募基金持仓变动”；反映机构股东参与和持股变化，与原目录“机构持股比例”同属股东与股权因子。",
      "factor_name": "fundnum_promotion",
      "factor_desc": "数据表：alternative_fundnum_promotion.feather；原表逻辑：公募基金持仓变动。",
      "formula": "读取 alternative_fundnum_promotion.feather 中的 fundnum_promotion 预计算字段，按 date、code 对齐；原表未提供源因子的计算公式。"
    },
    {
      "factor_id": "factor_010",
      "b_id": "B6",
      "b_name": "另类、分析师与股东因子",
      "c_id": "C6-3",
      "c_name": "股东与股权因子",
      "classification_basis": "原表说明为“社保基金持仓变动”；信息来自特定机构股东的持股状况，按机构持仓机制归入股东与股权因子。",
      "factor_name": "ssfund_ratio",
      "factor_desc": "数据表：alternative_ssfund_ratio.feather；原表逻辑：社保基金持仓变动。",
      "formula": "读取 alternative_ssfund_ratio.feather 中的 ssfund_ratio 预计算字段，按 date、code 对齐；原表未提供源因子的计算公式。"
    },
    {
      "factor_id": "factor_011",
      "b_id": "B6",
      "b_name": "另类、分析师与股东因子",
      "c_id": "C6-3",
      "c_name": "股东与股权因子",
      "classification_basis": "原表说明为“散户持仓变动”；反映股东户数或持股分布变化，与原目录“股东户数”同属股东与股权因子。",
      "factor_name": "holdernum_promotion",
      "factor_desc": "数据表：alternative_holdingchg.feather；原表逻辑：散户持仓变动。",
      "formula": "读取 alternative_holdingchg.feather 中的 holdernum_promotion 预计算字段，按 date、code 对齐；原表未提供源因子的计算公式。"
    },
    {
      "factor_id": "factor_013",
      "b_id": "B6",
      "b_name": "另类、分析师与股东因子",
      "c_id": "C6-3",
      "c_name": "股东与股权因子",
      "classification_basis": "原表说明为“管理层工资”；按管理层薪酬对应的公司治理和激励机制，参照原目录“公司治理指数”归入股东与股权因子；薪酬的具体统计范围仍以源数据为准。",
      "factor_name": "salary_avg",
      "factor_desc": "数据表：alternative_executive_salary.feather；原表逻辑：管理层工资。",
      "formula": "读取 alternative_executive_salary.feather 中的 salary_avg 预计算字段，按 date、code 对齐；原表未提供源因子的计算公式。"
    },
    {
      "factor_id": "factor_014",
      "b_id": "B3",
      "b_name": "财务基本面因子",
      "c_id": "C3-2",
      "c_name": "成长因子",
      "classification_basis": "原表说明为基于 ROE 和净利润变化的短期成长类别变量；主导信息是核心财务指标的增长变化，归入原目录成长因子。",
      "factor_name": "STtype",
      "factor_desc": "数据表：fs_lstype.feather；原表逻辑：类别变量：短期成长（ROE的变化值、净利润的变化值）。",
      "formula": "读取 fs_lstype.feather 中的 STtype 预计算字段，按 date、code 对齐；原表未提供源因子的计算公式。"
    },
    {
      "factor_id": "factor_015",
      "b_id": "B3",
      "b_name": "财务基本面因子",
      "c_id": "C3-1",
      "c_name": "财务质量因子",
      "classification_basis": "原表说明基于 ROE、ROA 五年均值及方差刻画长期类型；主导信息是持续盈利能力及其稳定性，参照原目录盈利持续性、盈余可预测性归入财务质量因子。",
      "factor_name": "LTtype",
      "factor_desc": "数据表：fs_lstype.feather；原表逻辑：类别变量：长期成长（ROE\\ROA5年的均值/方差。",
      "formula": "读取 fs_lstype.feather 中的 LTtype 预计算字段，按 date、code 对齐；原表未提供源因子的计算公式。"
    },
    {
      "factor_id": "factor_016",
      "b_id": "B3",
      "b_name": "财务基本面因子",
      "c_id": "C3-2",
      "c_name": "成长因子",
      "classification_basis": "原表以营业收入相对过去八个季度的时序标准化刻画收入趋势；反映营收随时间的改善或减弱，按原目录营收变化的逻辑归入成长因子。",
      "factor_name": "sales_trend",
      "factor_desc": "数据表：fs_zx.feather；原表逻辑：营业收入趋势（时序标准化，减过去8个季度的均值，除标准差）。",
      "formula": "营业收入的时序标准化：当期值减去过去8个季度的均值，再除以同期标准差。原表未说明窗口是否包含当期。"
    },
    {
      "factor_id": "factor_017",
      "b_id": "B3",
      "b_name": "财务基本面因子",
      "c_id": "C3-2",
      "c_name": "成长因子",
      "classification_basis": "原表以净利润相对过去九个季度的时序标准化刻画盈利趋势；反映利润随时间的改善或减弱，按原目录核心盈利变化的逻辑归入成长因子。",
      "factor_name": "profit_trend",
      "factor_desc": "数据表：fs_zx.feather；原表逻辑：净利润趋势（时序标准化，减过去9个季度的均值，除标准差）。",
      "formula": "净利润的时序标准化：当期值减去过去9个季度的均值，再除以同期标准差。原表未说明窗口是否包含当期。"
    },
    {
      "factor_id": "factor_018",
      "b_id": "B3",
      "b_name": "财务基本面因子",
      "c_id": "C3-1",
      "c_name": "财务质量因子",
      "classification_basis": "原表以净利润序列的二次项系数刻画业绩加速度；原目录“业绩趋势”同样采用利润序列二次项衡量业绩形态并归入财务质量，本项沿用该分类先例。",
      "factor_name": "NP_Acc",
      "factor_desc": "数据表：fs_growth_qqc.feather；原表逻辑：加速度a：净利润=𝑎*t^2+𝑏*𝑡+𝑐。",
      "formula": "净利润拟合为 a×t²+b×t+c，以二次项 a 表示加速度；原表未提供拟合窗口和频率。"
    },
    {
      "factor_id": "factor_019",
      "b_id": "B3",
      "b_name": "财务基本面因子",
      "c_id": "C3-1",
      "c_name": "财务质量因子",
      "classification_basis": "原表综合业绩增速分组得分与组内增长加速度标准分，反映业绩趋势的强弱；参照原目录“业绩趋势”的既有归类，归入财务质量因子。",
      "factor_name": "QPT",
      "factor_desc": "数据表：fs_growth_qqc.feather；原表逻辑：业绩趋势:首先，依据业绩增速指标将全市场股票均分成三组，给每组股票打分，增\n速最高一组得分为 3 ，增速最低一组得分为 1 ，中间一组得分为 2\n• 其次，每一组内依据业绩增长加速度指标进行打分（标准分法），加速度越\n高得分也越高；\n• 最后，将两步骤的得分相加为每只股票的最终得分。。",
      "formula": "先按业绩增速将全市场股票均分三组，分别给3、2、1分；再于各组内按业绩增长加速度计算标准分；两项得分相加。原表未给出指标及标准分的精确定义。"
    },
    {
      "factor_id": "factor_020",
      "b_id": "B3",
      "b_name": "财务基本面因子",
      "c_id": "C3-1",
      "c_name": "财务质量因子",
      "classification_basis": "原表说明为采用归母扣非业绩的 QPT；主要机制仍为业绩趋势评价，参照原目录“业绩趋势”归入财务质量因子，扣非口径属于指标构造细节。",
      "factor_name": "QPT_gmkf",
      "factor_desc": "数据表：fs_growth_qqc.feather；原表逻辑：同QPT，业绩用归母扣非。",
      "formula": "读取 fs_growth_qqc.feather 中的 QPT_gmkf 预计算字段，按 date、code 对齐；原表未提供源因子的计算公式。"
    },
    {
      "factor_id": "factor_021",
      "b_id": "B3",
      "b_name": "财务基本面因子",
      "c_id": "C3-1",
      "c_name": "财务质量因子",
      "classification_basis": "原表说明在 QPT 系列中增加业绩预告信息，主导机制仍为业绩趋势评价，参照原目录“业绩趋势”归入财务质量因子；原表存在自指，具体基础版本待源实现核验。",
      "factor_name": "QPT_gmkf4",
      "factor_desc": "数据表：fs_growth_qqc.feather；原表逻辑：同QPT_gmkf4;增加业绩预告的信息。",
      "formula": "读取 fs_growth_qqc.feather 中的 QPT_gmkf4 预计算字段，按 date、code 对齐；原表未提供源因子的计算公式。"
    },
    {
      "factor_id": "factor_022",
      "b_id": "B3",
      "b_name": "财务基本面因子",
      "c_id": "C3-1",
      "c_name": "财务质量因子",
      "classification_basis": "原表列为“财报／成长”；实际逻辑是以毛利率解释净利润并取最近八个季度逐股回归的当期残差。主导机制为财务项目提纯，与原目录“线性纯化净利润”和“营业能力改善”同类，归入 B3 财务基本面因子／C3-1 财务质量因子。",
      "factor_name": "f1_new",
      "factor_desc": "利用过去两年的季度财务关系，剔除毛利率能够解释的净利润成分，以当期残差刻画相对于该基准的盈利偏离。正残差表示当期净利润高于回归拟合值。数据源：fs_quality_fnew.feather 的同名字段；回归窗口为最近八个季度（两年）。",
      "formula": "按用户确认口径：对每只股票 i，在 W_q={q-7,…,q} 的最近八个季度上，以 NP（净利润）为被解释变量、GM（毛利率）为解释变量估计回归函数 m_hat_(i,q)。f1_new(i,q)=NP_(i,q)-m_hat_(i,q)(GM_(i,q))，即窗口末季 q 的当期回归残差，保留原始残差方向。财务变量口径及回归估计细节沿用源实现，未核验部分见文档说明。",
      "initial_ic": {
        "mean_ic": 0.00656372545658293,
        "mean_rank_ic": 0.00819584347514195,
        "rank_ic_std": 0.0571075237861224,
        "ic_dates": 1446,
        "mean_assets": 4316.11065006916,
        "first_ic_date": "2019-01-02",
        "last_ic_date": "2024-12-17",
        "signal_start": "2019-01-02",
        "target_cutoff": "2024-12-31",
        "horizon_trading_days": 10,
        "min_cross_section_assets": 5,
        "source": "/data/mongo/fs_quality_fnew.feather + quotefac_repaired.feather"
      }
    },
    {
      "factor_id": "factor_023",
      "b_id": "B3",
      "b_name": "财务基本面因子",
      "c_id": "C3-1",
      "c_name": "财务质量因子",
      "classification_basis": "原表列为“财报／成长”；实际逻辑是以营业收入解释营业利润并取最近八个季度逐股回归的当期残差。主导机制为财务项目提纯，与原目录“线性纯化净利润”和“营业能力改善”同类，归入 B3 财务基本面因子／C3-1 财务质量因子。",
      "factor_name": "f2_new",
      "factor_desc": "利用过去两年的季度收入与营业利润关系，剔除收入规模能够解释的营业利润成分。正残差表示给定营业收入下，营业利润高于历史关系的拟合值。数据源：fs_quality_fnew.feather 的同名字段；回归窗口为最近八个季度（两年）。",
      "formula": "按用户确认口径：对每只股票 i，在 W_q={q-7,…,q} 的最近八个季度上，以 OP（营业利润）为被解释变量、Revenue（营业收入）为解释变量估计回归函数 m_hat_(i,q)。f2_new(i,q)=OP_(i,q)-m_hat_(i,q)(Revenue_(i,q))，即窗口末季 q 的当期回归残差，保留原始残差方向。财务变量口径及回归估计细节沿用源实现，未核验部分见文档说明。",
      "initial_ic": {
        "mean_ic": 0.0116937756960819,
        "mean_rank_ic": 0.0138310673562828,
        "rank_ic_std": 0.0472186827699651,
        "ic_dates": 1446,
        "mean_assets": 4345.63278008299,
        "first_ic_date": "2019-01-02",
        "last_ic_date": "2024-12-17",
        "signal_start": "2019-01-02",
        "target_cutoff": "2024-12-31",
        "horizon_trading_days": 10,
        "min_cross_section_assets": 5,
        "source": "/data/mongo/fs_quality_fnew.feather + quotefac_repaired.feather"
      }
    },
    {
      "factor_id": "factor_024",
      "b_id": "B3",
      "b_name": "财务基本面因子",
      "c_id": "C3-1",
      "c_name": "财务质量因子",
      "classification_basis": "原表列为“财报／成长”；实际逻辑是以营业成本解释主营业务收入并取最近八个季度逐股回归的当期残差。主导机制为财务项目提纯，与原目录“线性纯化净利润”和“营业能力改善”同类，归入 B3 财务基本面因子／C3-1 财务质量因子。",
      "factor_name": "f3_new",
      "factor_desc": "利用过去两年的季度主营业务收入与营业成本关系，提取成本因素未解释的收入成分。正残差表示给定营业成本下，主营业务收入高于历史关系的拟合值。数据源：fs_quality_fnew.feather 的同名字段；回归窗口为最近八个季度（两年）。",
      "formula": "按用户确认口径：对每只股票 i，在 W_q={q-7,…,q} 的最近八个季度上，以 MainRevenue（主营业务收入）为被解释变量、OperatingCost（营业成本）为解释变量估计回归函数 m_hat_(i,q)。f3_new(i,q)=MainRevenue_(i,q)-m_hat_(i,q)(OperatingCost_(i,q))，即窗口末季 q 的当期回归残差，保留原始残差方向。财务变量口径及回归估计细节沿用源实现，未核验部分见文档说明。",
      "initial_ic": {
        "mean_ic": 0.012431645427475,
        "mean_rank_ic": 0.0175816115986886,
        "rank_ic_std": 0.051894971902186,
        "ic_dates": 1446,
        "mean_assets": 4316.11065006916,
        "first_ic_date": "2019-01-02",
        "last_ic_date": "2024-12-17",
        "signal_start": "2019-01-02",
        "target_cutoff": "2024-12-31",
        "horizon_trading_days": 10,
        "min_cross_section_assets": 5,
        "source": "/data/mongo/fs_quality_fnew.feather + quotefac_repaired.feather"
      }
    },
    {
      "factor_id": "factor_025",
      "b_id": "B3",
      "b_name": "财务基本面因子",
      "c_id": "C3-1",
      "c_name": "财务质量因子",
      "classification_basis": "原表列为“财报／成长”；实际逻辑是以应计项目变化解释净利润并取最近八个季度逐股回归的当期残差。主导机制为财务项目提纯，与原目录“线性纯化净利润”和“营业能力改善”同类，归入 B3 财务基本面因子／C3-1 财务质量因子。",
      "factor_name": "f4_new",
      "factor_desc": "利用过去两年的季度关系，剔除应计项目变化能够解释的净利润成分，刻画应计变化之外的盈利偏离。残差本身不直接等同于现金利润或盈余管理程度。数据源：fs_quality_fnew.feather 的同名字段；回归窗口为最近八个季度（两年）。",
      "formula": "按用户确认口径：对每只股票 i，在 W_q={q-7,…,q} 的最近八个季度上，以 NP（净利润）为被解释变量、DeltaAccruals（应计项目变化）为解释变量估计回归函数 m_hat_(i,q)。f4_new(i,q)=NP_(i,q)-m_hat_(i,q)(DeltaAccruals_(i,q))，即窗口末季 q 的当期回归残差，保留原始残差方向。财务变量口径及回归估计细节沿用源实现，未核验部分见文档说明。",
      "initial_ic": {
        "mean_ic": 0.0119636556133836,
        "mean_rank_ic": 0.0154633912172056,
        "rank_ic_std": 0.0735373980682867,
        "ic_dates": 1446,
        "mean_assets": 4333.0836791148,
        "first_ic_date": "2019-01-02",
        "last_ic_date": "2024-12-17",
        "signal_start": "2019-01-02",
        "target_cutoff": "2024-12-31",
        "horizon_trading_days": 10,
        "min_cross_section_assets": 5,
        "source": "/data/mongo/fs_quality_fnew.feather + quotefac_repaired.feather"
      }
    },
    {
      "factor_id": "factor_026",
      "b_id": "B3",
      "b_name": "财务基本面因子",
      "c_id": "C3-1",
      "c_name": "财务质量因子",
      "classification_basis": "原表列为“财报／成长”；实际逻辑是以长期资产变化、无形资产变化及商誉变化解释净利润并取最近八个季度逐股回归的当期残差。主导机制为财务项目提纯，与原目录“线性纯化净利润”和“营业能力改善”同类，归入 B3 财务基本面因子／C3-1 财务质量因子。",
      "factor_name": "f5_new",
      "factor_desc": "利用过去两年的季度关系，剔除长期资产、无形资产和商誉变化能够共同解释的净利润成分。主导机制是盈利提纯，资产类别作为解释变量参与回归。数据源：fs_quality_fnew.feather 的同名字段；回归窗口为最近八个季度（两年）。",
      "formula": "按用户确认口径：对每只股票 i，在 W_q={q-7,…,q} 的最近八个季度上，以 NP（净利润）为被解释变量、(DeltaLongTermAssets, DeltaIntangibles, DeltaGoodwill)（长期资产变化、无形资产变化及商誉变化）为解释变量估计回归函数 m_hat_(i,q)。f5_new(i,q)=NP_(i,q)-m_hat_(i,q)((DeltaLongTermAssets, DeltaIntangibles, DeltaGoodwill)_(i,q))，即窗口末季 q 的当期回归残差，保留原始残差方向。财务变量口径及回归估计细节沿用源实现，未核验部分见文档说明。",
      "initial_ic": {
        "mean_ic": 0.0115986838298313,
        "mean_rank_ic": 0.0146734924182438,
        "rank_ic_std": 0.0726456497477203,
        "ic_dates": 1446,
        "mean_assets": 4352.81189488243,
        "first_ic_date": "2019-01-02",
        "last_ic_date": "2024-12-17",
        "signal_start": "2019-01-02",
        "target_cutoff": "2024-12-31",
        "horizon_trading_days": 10,
        "min_cross_section_assets": 5,
        "source": "/data/mongo/fs_quality_fnew.feather + quotefac_repaired.feather"
      }
    },
    {
      "factor_id": "factor_027",
      "b_id": "B3",
      "b_name": "财务基本面因子",
      "c_id": "C3-1",
      "c_name": "财务质量因子",
      "classification_basis": "原表列为“财报／成长”；实际逻辑是以固定资产解释营业成本并取最近八个季度逐股回归的当期残差。主导机制为财务项目提纯，与原目录“线性纯化净利润”和“营业能力改善”同类，归入 B3 财务基本面因子／C3-1 财务质量因子。",
      "factor_name": "f6_new",
      "factor_desc": "利用过去两年的季度固定资产与营业成本关系，提取固定资产因素未解释的成本偏离。按用户确认直接取当期残差：正残差表示成本高于拟合基准，负残差表示成本低于拟合基准。数据源：fs_quality_fnew.feather 的同名字段；回归窗口为最近八个季度（两年）。",
      "formula": "按用户确认口径：对每只股票 i，在 W_q={q-7,…,q} 的最近八个季度上，以 OperatingCost（营业成本）为被解释变量、FixedAssets（固定资产）为解释变量估计回归函数 m_hat_(i,q)。f6_new(i,q)=OperatingCost_(i,q)-m_hat_(i,q)(FixedAssets_(i,q))，即窗口末季 q 的当期回归残差，保留原始残差方向。财务变量口径及回归估计细节沿用源实现，未核验部分见文档说明。",
      "initial_ic": {
        "mean_ic": 0.00208944750488406,
        "mean_rank_ic": 0.00247636607359135,
        "rank_ic_std": 0.0396434452579527,
        "ic_dates": 1446,
        "mean_assets": 3811.29668049793,
        "first_ic_date": "2019-01-02",
        "last_ic_date": "2024-12-17",
        "signal_start": "2019-01-02",
        "target_cutoff": "2024-12-31",
        "horizon_trading_days": 10,
        "min_cross_section_assets": 5,
        "source": "/data/mongo/fs_quality_fnew.feather + quotefac_repaired.feather"
      }
    },
    {
      "factor_id": "factor_028",
      "b_id": "B3",
      "b_name": "财务基本面因子",
      "c_id": "C3-1",
      "c_name": "财务质量因子",
      "classification_basis": "原表列为“财报／成长”；实际逻辑是以营业外收入及工资解释净利润并取最近八个季度逐股回归的当期残差。主导机制为财务项目提纯，与原目录“线性纯化净利润”和“营业能力改善”同类，归入 B3 财务基本面因子／C3-1 财务质量因子。",
      "factor_name": "f7_new",
      "factor_desc": "利用过去两年的季度关系，剔除营业外收入和工资能够共同解释的净利润成分，提取其余盈利偏离。工资的具体统计范围沿用源实现。数据源：fs_quality_fnew.feather 的同名字段；回归窗口为最近八个季度（两年）。",
      "formula": "按用户确认口径：对每只股票 i，在 W_q={q-7,…,q} 的最近八个季度上，以 NP（净利润）为被解释变量、(NonOperatingIncome, Salary)（营业外收入及工资）为解释变量估计回归函数 m_hat_(i,q)。f7_new(i,q)=NP_(i,q)-m_hat_(i,q)((NonOperatingIncome, Salary)_(i,q))，即窗口末季 q 的当期回归残差，保留原始残差方向。财务变量口径及回归估计细节沿用源实现，未核验部分见文档说明。",
      "initial_ic": {
        "mean_ic": 0.00929774982986658,
        "mean_rank_ic": 0.0126298176299767,
        "rank_ic_std": 0.0595667278067966,
        "ic_dates": 1446,
        "mean_assets": 4176.21092669433,
        "first_ic_date": "2019-01-02",
        "last_ic_date": "2024-12-17",
        "signal_start": "2019-01-02",
        "target_cutoff": "2024-12-31",
        "horizon_trading_days": 10,
        "min_cross_section_assets": 5,
        "source": "/data/mongo/fs_quality_fnew.feather + quotefac_repaired.feather"
      }
    },
    {
      "factor_id": "factor_029",
      "b_id": "B3",
      "b_name": "财务基本面因子",
      "c_id": "C3-1",
      "c_name": "财务质量因子",
      "classification_basis": "原表列为“财报／盈利”，逻辑为 ROIC；投入资本回报率衡量经营资本的盈利效率，与原目录盈利能力及财务质量指标同属财务质量因子。",
      "factor_name": "ROIC_new",
      "factor_desc": "数据表：fs_quality_fnew.feather；原表逻辑：ROIC。",
      "formula": "读取 fs_quality_fnew.feather 中的 ROIC_new 预计算字段，按 date、code 对齐；原表未提供源因子的计算公式。"
    },
    {
      "factor_id": "factor_030",
      "b_id": "B6",
      "b_name": "另类、分析师与股东因子",
      "c_id": "C6-2",
      "c_name": "分析师预期因子",
      "classification_basis": "原表说明为过去三个月分析师净利润预测的变化增长；主导机制是分析师盈利预期修正，与原目录“加权的盈余调整幅度”同属分析师预期因子。",
      "factor_name": "afr",
      "factor_desc": "数据表：forecast_pafr.feather；原表逻辑：过去3个月分析师净利润变化增长。",
      "formula": "读取 forecast_pafr.feather 中的 afr 预计算字段，按 date、code 对齐；原表未提供源因子的计算公式。"
    },
    {
      "factor_id": "factor_031",
      "b_id": "B6",
      "b_name": "另类、分析师与股东因子",
      "c_id": "C6-2",
      "c_name": "分析师预期因子",
      "classification_basis": "原表说明为 afr 对过去三个月超额收益正交后的信号；主体仍是剔除价格影响后的分析师预测修正，按主导机制归入分析师预期因子。",
      "factor_name": "pafr2",
      "factor_desc": "数据表：forecast_pafr.feather；原表逻辑：afr对过去3个月的超额做正交。",
      "formula": "读取 forecast_pafr.feather 中的 pafr2 预计算字段，按 date、code 对齐；原表未提供源因子的计算公式。"
    },
    {
      "factor_id": "factor_032",
      "b_id": "B6",
      "b_name": "另类、分析师与股东因子",
      "c_id": "C6-2",
      "c_name": "分析师预期因子",
      "classification_basis": "原表说明为 pafr2 的一年时序标准化；底层信息仍为分析师盈利预测变化，标准化属于实现变换，按主导机制归入分析师预期因子。",
      "factor_name": "forecast_inertia_factor",
      "factor_desc": "数据表：forecast_pafr.feather；原表逻辑：pafr2过去一年的时序标准化。",
      "formula": "读取 forecast_pafr.feather 中的 forecast_inertia_factor 预计算字段，按 date、code 对齐；原表未提供源因子的计算公式。"
    }
  ],
  "unimplemented_factors": [
    {
      "factor_id": "factor_012",
      "b_id": "B4",
      "b_name": "动量、风险与波动率因子",
      "c_id": "C4-3",
      "c_name": "流动性冲击因子",
      "classification_basis": "原表标注为停用的北向资金因子；为保持原目录将“北上持仓因子”置于流动性冲击维度的分类逻辑，归入 B4／C4-3，同时保留停用状态。",
      "factor_name": "HK_holdings_ratio_delta_60",
      "factor_desc": "原表逻辑：北向资金（停用）。",
      "formula": "原表未提供明确计算公式。",
      "unimplemented_reason": "原表标注“北向资金（停用）”，且未指定数据表。"
    },
    {
      "factor_id": "factor_033",
      "b_id": "B3",
      "b_name": "财务基本面因子",
      "c_id": "C3-1",
      "c_name": "财务质量因子",
      "classification_basis": "本地类型表将该旧标签列为 fundamental；按 ROIC 所指的投入资本盈利效率归入财务质量因子。同名字段缺失，且与 ROIC_new 的对应关系尚未确认。",
      "factor_name": "ROIC",
      "factor_desc": "本地因子库中的基本面标签；未找到同名 Feather 字段。",
      "formula": "暂无可核验的计算公式或同名数据字段。",
      "unimplemented_reason": "本地因子库保留该名称，但 /data/mongo 下 Feather 文件均无同名字段；与 *_new 字段的对应关系未确认。"
    },
    {
      "factor_id": "factor_034",
      "b_id": null,
      "b_name": null,
      "c_id": null,
      "c_name": null,
      "classification_basis": "本地类型表仅将该旧标签列为 fundamental，缺少同名字段和机制定义，尚不足以映射到原目录的具体 B/C 类别；与同序号 *_new 因子的对应关系未确认。",
      "factor_name": "f1",
      "factor_desc": "本地因子库中的基本面标签；未找到同名 Feather 字段。",
      "formula": "暂无可核验的计算公式或同名数据字段。",
      "unimplemented_reason": "本地因子库保留该名称，但 /data/mongo 下 Feather 文件均无同名字段；与 *_new 字段的对应关系未确认。"
    },
    {
      "factor_id": "factor_035",
      "b_id": null,
      "b_name": null,
      "c_id": null,
      "c_name": null,
      "classification_basis": "本地类型表仅将该旧标签列为 fundamental，缺少同名字段和机制定义，尚不足以映射到原目录的具体 B/C 类别；与同序号 *_new 因子的对应关系未确认。",
      "factor_name": "f2",
      "factor_desc": "本地因子库中的基本面标签；未找到同名 Feather 字段。",
      "formula": "暂无可核验的计算公式或同名数据字段。",
      "unimplemented_reason": "本地因子库保留该名称，但 /data/mongo 下 Feather 文件均无同名字段；与 *_new 字段的对应关系未确认。"
    },
    {
      "factor_id": "factor_036",
      "b_id": null,
      "b_name": null,
      "c_id": null,
      "c_name": null,
      "classification_basis": "本地类型表仅将该旧标签列为 fundamental，缺少同名字段和机制定义，尚不足以映射到原目录的具体 B/C 类别；与同序号 *_new 因子的对应关系未确认。",
      "factor_name": "f3",
      "factor_desc": "本地因子库中的基本面标签；未找到同名 Feather 字段。",
      "formula": "暂无可核验的计算公式或同名数据字段。",
      "unimplemented_reason": "本地因子库保留该名称，但 /data/mongo 下 Feather 文件均无同名字段；与 *_new 字段的对应关系未确认。"
    },
    {
      "factor_id": "factor_037",
      "b_id": null,
      "b_name": null,
      "c_id": null,
      "c_name": null,
      "classification_basis": "本地类型表仅将该旧标签列为 fundamental，缺少同名字段和机制定义，尚不足以映射到原目录的具体 B/C 类别；与同序号 *_new 因子的对应关系未确认。",
      "factor_name": "f4",
      "factor_desc": "本地因子库中的基本面标签；未找到同名 Feather 字段。",
      "formula": "暂无可核验的计算公式或同名数据字段。",
      "unimplemented_reason": "本地因子库保留该名称，但 /data/mongo 下 Feather 文件均无同名字段；与 *_new 字段的对应关系未确认。"
    },
    {
      "factor_id": "factor_038",
      "b_id": null,
      "b_name": null,
      "c_id": null,
      "c_name": null,
      "classification_basis": "本地类型表仅将该旧标签列为 fundamental，缺少同名字段和机制定义，尚不足以映射到原目录的具体 B/C 类别；与同序号 *_new 因子的对应关系未确认。",
      "factor_name": "f5",
      "factor_desc": "本地因子库中的基本面标签；未找到同名 Feather 字段。",
      "formula": "暂无可核验的计算公式或同名数据字段。",
      "unimplemented_reason": "本地因子库保留该名称，但 /data/mongo 下 Feather 文件均无同名字段；与 *_new 字段的对应关系未确认。"
    }
  ]
}
```
