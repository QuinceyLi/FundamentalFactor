"""Write an offline implementation/data-delivery review from recorded evidence."""
from pathlib import Path
from html import escape
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/proxy_inputs"


def read(path):
    path = ROOT / path
    return json.loads(path.read_text()) if path.exists() else {}


def details(title, value):
    return f"<details><summary>{escape(title)}</summary><pre>{escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre></details>"


def main():
    st = read("outputs/historical_st/staging/public_st_history_manifest.json")
    price = read("outputs/price_repair/validated/quotefac_repair_manifest.json")
    proxy = read("outputs/proxy_inputs/proxy_inputs_manifest.json")
    gaps = read("outputs/proxy_inputs/remaining_gaps.json")
    execution_checks = read("outputs/proxy_inputs/real_execution_checks.json")
    receipts = []
    for filename in ("publication_core_receipt.json", "publication_proxy_receipt.json"):
        receipts += read("outputs/proxy_inputs/" + filename) or []
    rows = "".join(f'<tr><td>{escape(Path(r["destination"]).name)}</td><td>{r["bytes"]:,}</td><td class="hash">{r["sha256"]}</td></tr>' for r in receipts)
    runs = []
    for name, label in (("factor_test_proxy", "全历史因子研究"), ("factor_test_execution_proxy", "2025上半年交易限制代理账本")):
        result = read(f"outputs/{name}/results.json")
        manifest = read(f"outputs/{name}/panel_manifest.json")
        runs.append(f'<li><a href="../{name}/report.html">{label}</a>：{escape(result.get("status", "尚未完成"))}，'
                    f'{manifest.get("weeks", "—")} 周，未解决个股周标签 {manifest.get("unresolved_labels", "—")} 条。'
                    f'质量：{escape(str(manifest.get("quality_status", "见报告")))}</li>')
    known = proxy.get("st_unknown_fraction", 0)
    html = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>因子框架、数据修复与历史ST交付</title>
<style>body{{font:16px/1.75 system-ui,sans-serif;max-width:1160px;margin:40px auto;padding:0 20px;color:#203348}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;border-bottom:1px solid #dae2e8;padding:10px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7fa;padding:15px}}.hash{{font:12px monospace;overflow-wrap:anywhere}}.notice{{background:#fff2d5;padding:18px;border-left:5px solid #b87d18}}details{{margin:18px 0}}a{{color:#09669b}}</style>
<h1>因子框架、数据修复与历史 ST 交付</h1><p>2026-10-14 · Python · 沪深A股 · 无预测模型训练 · 无持仓优化器</p>
<p class="notice">真实源记录、衍生状态、交易代理分别留痕。历史ST覆盖不完整；涨跌停和停牌采用明示代理。报告为因子诊断及交易限制敏感性，不能解释成可实现扣费策略。</p>
<h2>框架已实现</h2><p>信号和十分组先冻结，再连接未来收益；三套信号使用相同股票样本。每组直接等权，输出RankIC、Pearson IC、分组/超额收益、Q10−Q1、年度表现和漂移后双边换手。行业中性不要求行业市值或个股实际市值。</p>
<p>代理模式允许未知ST单列并保留，未知入场停牌排除；有效标签计算IC并披露覆盖。缺失标签不补零，也不删除持仓后重新等权，受影响组当周收益保持缺失。原严格模式保留。</p>
<p>附加复权单位账本实行停牌双向冻结、涨停不买/跌停不卖、先卖后买、现金不足按比例缩买，保留卖不出的旧仓和买不进的现金。复权单位不是真实股数；不重复加入分红送转。实际持仓跨过无法解决的退市，单独停止该账户收益和交易，其他账户继续。</p>
<h2>实际运行报告</h2><ul>{''.join(runs)}</ul>
<h2>价格异常修复</h2><p>旧、新、Mongo三本地源比较，原始行情重合3,121,636行，开/收/昨收/成交量一致。东方财富独立23日原始开收盘全部吻合，公开复权行情没有暴跌；新浪分红记录的邻近除息日为2025-06-12。</p>
<p>仅修复000001.SZ在2025-05-28的倍率断点：边界前3,008行全部acc价格统一乘0.6138449932017157，保留其他股票、字段和原始文件。异常日复权收盘收益由−38.401803%改为+0.348130%，边界前区间收益不变。两种公开与本地复权方法未混接。</p>
<h2>历史ST下载及核验</h2><p>当天从深交所下载{st.get('name_events_a_share', '—'):,}条A股官方简称事件，涉及{st.get('name_chains_checked', '—'):,}个代码。由变更后名称识别ST/*ST及历史S/G前缀，修复32条旧识别遗漏。逐股名称链核验连续；在34,224个已知深圳抽样中，本地名称代理与官方记录有1,465条、250代码冲突，采用官方事件。</p>
<p>首次事件前、没有事件和上海未覆盖的状态保持未知，不用今天名称倒填历史。采集日后设置失效标记，不能向未来无限延续。本次每日代理输入未知ST占{known:.2%}，股票周次实际比例见因子报告。BaoStock两次公开接口连接失败，没有生成虚构日线。</p>
<h2>仍未补齐的内容</h2><ul><li>上海完整历史ST、深圳首次可确认事件之前的ST。</li><li>历史逐股真实限价及复牌/重组等特殊例外；当前为代理，已纳入创业板2020-08-24和主板ST2026-07-06规则边界。</li><li>57股票的12,640条复权记录缺少原始行情对应键，保留缺口；9个正向特殊复权事件未证伪，未擅自抹掉。</li><li>完整公司行动与退市后真实退出价值仍未覆盖；不阻断其他有效IC，影响到的收益保持缺失。</li></ul>
<p>本次全历史实际剩余缺口：{gaps.get('unresolved_labels', '—')} 条退市个股周标签，涉及 {gaps.get('affected_benchmark_weeks', '—')} 周；标签覆盖率 {gaps.get('label_coverage', 0):.5%}。<a href="../factor_test_proxy/unresolved_labels.csv">逐条缺口清单</a>。过滤后股票周的 ST 未知比例为 {gaps.get('eligible_unknown_st_fraction', 0):.2%}。</p>
<p>当前研究仅沪深A股，不需要单独补北交所；也不需要实际流通市值，因为未启用市值中性、市值加权或微盘筛选。</p>
<h2>已写入 /data/mongo 的文件</h2><p>日期YYYYMMDD、股票代码六位加交易所后缀；原文件未覆盖。下表SHA256为发布后完整文件核对结果。</p><table><tr><th>文件</th><th>字节</th><th>SHA256</th></tr>{rows}</table>
<h2>方法参考与证据</h2><p><a href="https://github.com/quantopian/alphalens/blob/master/alphalens/performance.py">Alphalens因子统计</a>；<a href="https://github.com/microsoft/qlib/blob/main/qlib/backtest/exchange.py">Qlib交易限制代理</a>；本地GFOptim调用链中的持仓、现金和停牌处理。保留本框架周度开盘口径。</p>
<p><a href="https://www.szse.cn/api/report/ShowReport?SHOWTYPE=xlsx&amp;CATALOGID=SSGSGMXX&amp;TABKEY=tab2">深交所历史简称官方源</a>；<a href="https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20260424_10816474.shtml">上交所2026交易规则</a>；<a href="https://docs.static.szse.cn/www/lawrules/rule/stock/W020260424690713346617.pdf">深交所2026修订说明</a>。</p>
{details('价格修复完整记录', price)}{details('历史ST来源与真实性边界', st)}{details('代理输入与规则来源', proxy)}{details('真实执行账本资金、权重与限制检查', execution_checks)}
<p><a href="test_results.log">完整验收日志</a> · <a href="export_tests.log">大型账本导出回归日志</a> · <a href="final_targeted_tests.log">针对性回归日志</a> · <a href="../../README.md">运行说明</a></p></html>'''
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "review.html").write_text(html, encoding="utf-8")
    print(OUT / "review.html")


if __name__ == "__main__":
    main()
