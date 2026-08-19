# -*- coding: utf-8 -*-
"""Stream v4g prompt 模块 —— 纠纷调解「流式辅助（增量版 / diff）」。

v4g: system/user/diff 结构沿用 v4/v4b/v4d; system 由 SYS_FILE 切换(见下方 SYS_FILE)。
系谱: v4f(cd0de265/3434B) --加【★换人=新号】--> system_v4g.txt(7cfdebda/4400B)
      --逐版追加约束--> system_cprime_procfilter_v4.txt(d76c8045/13963B)。
      **当前默认 = system_cprime_procfilter_v4.txt。**
两者契约(五块结构/编号规则/闭合词表)逐字一致, 仅逐版追加约束。
- system: 单一常量 -> 由 SYS_FILE 指定(默认 system_cprime_procfilter_v4.txt; 回滚设 system_v4g.txt)
- user(每一步都带 累积状态, 首步为空):
      【当事人】{名册}\n\n【当前累积状态】\n{全量五块JSON 或 (空,这是第一步)}\n\n【新对话】\n{新增对话}
- 名册: "姓名(申请人方) ; 姓名(被申请人方)"   (分隔符 = ' ; ')
- 对话行: "turn_001 姓名: 文本"
- 无案由大类。
输出(assistant)为 diff JSON: {"新增":[...], "更新":[...], "调解方案":{...(可省)}}
  · 新增/更新 每条带“编号”, 所属块由编号前缀决定:
       claim_* -> 诉求   fact_* -> 事实   issue_* -> 争议焦点   evidence_* -> 证据
  · 新增 = 追加; 更新 = 按编号整条替换; 调解方案 = 整块替换(给了才换)。
本模块提供 apply_diff() 把 diff 合并回全量五块累积状态。
(apply_diff 逻辑与 v4d 同源; 曾对 v4d 训练数据回放校验 2342/2342 步一致。)
"""
import os
import json
import copy

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROMPTS = os.environ.get("PROMPTS_DIR", os.path.join(_HERE, os.pardir, "prompts"))
# system prompt 文件可用环境变量切换，便于换 prompt 而不动代码：
#   默认 system_cprime_procfilter_v4.txt (md5 d76c8045 / 13963B) —— 在 v4g 那份基础上逐版追加
#     约束(事实块只留已发生、程序性事项不进争议焦点、诉求金额只写一个总额等)，
#     这些约束同时在 normalize_diff_v4g.py 里有代码侧硬规则兜底(prompt 单靠模型判会残留)。
#     ⚠️ 它**不是** v4g adapter 的训练用 prompt，属训练后追加，与训练分布有偏离。
#   备选 system_v4g.txt (md5 7cfdebda / 4400B) —— v4g 训练时那份，与
#     adapters/v4g/_sys_v4g.txt 逐字一致(train==eval==serving 对齐)。
#     契约(五块结构/编号规则/闭合词表)两份逐字一致，封装层无需改动。
#     回滚: SYS_FILE=system_v4g.txt 重启。
SYS_FILE = os.environ.get("SYS_FILE", "system_cprime_procfilter_v4.txt")
with open(os.path.join(_PROMPTS, SYS_FILE), encoding="utf-8") as _f:
    SYSTEM = _f.read()

# 全量累积状态固定五块 + 固定顺序
STATE_KEYS = ["诉求", "事实", "争议焦点", "证据", "调解方案"]
_LIST_KEYS = STATE_KEYS[:4]                       # 前四块是数组
_PLAN_KEY = "调解方案"                             # 第五块是对象
_EMPTY_MARK = "(空,这是第一步)"                    # 首步累积状态占位符(逐字对齐训练)

# 编号前缀 -> 所属块
_PREFIX_BLOCK = {"claim": "诉求", "fact": "事实", "issue": "争议焦点", "evidence": "证据"}


def build_system() -> str:
    return SYSTEM


def _roster(dangshiren) -> str:
    """str -> 原样; list[{姓名,方}] -> '姓名(方) ; 姓名(方)'。"""
    if isinstance(dangshiren, str):
        return dangshiren.strip()
    parts = []
    for p in dangshiren or []:
        name = (p.get("姓名") or p.get("name") or "").strip()
        side = (p.get("方") or p.get("role") or "").strip()
        parts.append(f"{name}({side})" if side else name)
    return " ; ".join(parts)


def _dialogue(dialogue) -> str:
    """str -> 原样(strip); list[{turn_id?,speaker,text}] -> 'turn_XXX speaker: text' 逐行。"""
    if isinstance(dialogue, str):
        return dialogue.strip()
    lines = []
    for i, t in enumerate(dialogue or [], 1):
        tid = t.get("turn_id") or t.get("turn") or f"turn_{i:03d}"
        sp = t.get("speaker") or t.get("角色") or t.get("role") or ""
        tx = t.get("text") or t.get("内容") or t.get("content") or ""
        lines.append(f"{tid} {sp}: {tx}")
    return "\n".join(lines)


def _state_str(当前累积状态) -> str:
    """空 -> 首步占位符; str -> 原样; dict -> json.dumps(ensure_ascii=False)（与训练一致）。"""
    if 当前累积状态 in (None, "", {}, []):
        return _EMPTY_MARK
    if isinstance(当前累积状态, str):
        s = 当前累积状态.strip()
        return s if s else _EMPTY_MARK
    return json.dumps(当前累积状态, ensure_ascii=False)


def build_user(当事人, 新对话, 当前累积状态=None) -> str:
    roster = _roster(当事人)
    dlg = _dialogue(新对话)
    state = _state_str(当前累积状态)
    return f"【当事人】{roster}\n\n【当前累积状态】\n{state}\n\n【新对话】\n{dlg}"


# ------------------------- diff -> 全量累积状态 合并 -------------------------

def empty_state() -> dict:
    return {"诉求": [], "事实": [], "争议焦点": [], "证据": [], "调解方案": {}}


def _normalize_state(state) -> dict:
    """任意输入 -> 规整的五块 dict(固定顺序, 深拷贝, 不改调用方对象)。"""
    if state in (None, "", {}, []):
        src = {}
    elif isinstance(state, str):
        try:
            src = json.loads(state)
        except Exception:
            src = {}
    elif isinstance(state, dict):
        src = state
    else:
        src = {}
    ns = empty_state()
    for k in _LIST_KEYS:
        v = src.get(k)
        if isinstance(v, list):
            ns[k] = copy.deepcopy(v)
    v = src.get(_PLAN_KEY)
    if isinstance(v, dict):
        ns[_PLAN_KEY] = copy.deepcopy(v)
    return ns


def _block_of(item: dict):
    """按编号前缀判块; 前缀不认识时按字段特征兜底。返回块名或 None。"""
    bh = str(item.get("编号", ""))
    pre = bh.split("_", 1)[0] if "_" in bh else ""
    if pre in _PREFIX_BLOCK:
        return _PREFIX_BLOCK[pre]
    if "诉求" in item:
        return "诉求"
    if "描述" in item or "方态" in item:
        return "事实"
    if "问题" in item or "还缺什么信息" in item:
        return "争议焦点"
    if "证据" in item or "提交状态" in item:
        return "证据"
    return None


def apply_diff(当前累积状态, diff: dict) -> dict:
    """把模型 diff({新增,更新,调解方案}) 合并进累积状态, 返回**新的全量五块快照**。

    新增 = 追加到对应块; 更新 = 同编号整条替换(找不到则追加, 兜底); 调解方案 = 整块替换(present 才换)。
    未变项由累积状态自动带走(本函数从累积状态起算)。
    """
    ns = _normalize_state(当前累积状态)
    if not isinstance(diff, dict):
        return ns

    for it in diff.get("新增") or []:
        if not isinstance(it, dict):
            continue
        b = _block_of(it)
        if b in _LIST_KEYS:
            ns[b].append(it)

    for it in diff.get("更新") or []:
        if not isinstance(it, dict):
            continue
        b = _block_of(it)
        if b not in _LIST_KEYS:
            continue
        bh = it.get("编号")
        lst = ns[b]
        for i, old in enumerate(lst):
            if isinstance(old, dict) and old.get("编号") == bh:
                lst[i] = it
                break
        else:
            lst.append(it)   # 累积里没有该编号: 兜底当新增

    plan = diff.get(_PLAN_KEY)
    if isinstance(plan, dict) and plan:
        ns[_PLAN_KEY] = plan

    return ns
