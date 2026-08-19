# -*- coding: utf-8 -*-
"""封装层 diff 归一 —— 仅对**模型已生成的输出**做后处理, 不碰模型/prompt/采样/输入。

三层, 由便宜到贵, 模型判只兜灰区:
  1) 空更新过滤(纯判等, 零风险): 更新条目与上一版快照**逐字段完全相同** -> 直接删(空操作)。
  2) 描述锚定(纯规则, 仅兜"几乎一字不差"): 描述相似>=0.98 + 同金额同人 + 无否定翻转 -> 保留旧描述只吃结构变化;
       若锚定后与旧值全等 -> 退化为空更新, 删。(有改写/换说法一律不走这层, 下沉到 3 交模型。)
  3) 灰区调基座判官(mediation-raw): 凡有改写的更新(0.45<=相似<0.98, 同金额)与规则拿不准的新增,
       一次性打包问基座"是不是同一件事":
         新增·同一件事 -> 折成对已有编号的更新(保留已有描述);
         更新·同一件事 -> 锚定旧描述(或退化空更新删);
         判为不同 / 判不出 / 调用失败 -> 一律**放行照常显示**(绝不吞信息, fail-open)。

另: 列表字段(如"还缺什么信息""证据清单")内部去重 ——
  规则层去完全相同项(顺序不同/内部重复不算变化, 并入空更新判定);
  近义项(一个是另一个子串, 或高相似)并进上面**同一次**判官调用, 判"同一项"则只留更具体(更长)那条,
  冗余的删掉。删哪条由代码定, 裸模只回 true/false, 内容一个字不重写。

另: 硬规则(恒定生效, 不经判重/判官, 不受 JUDGE/DEDUP 影响) —— 调解员不是当事人:
  "诉求"若整条以调解员为主语(调解员提出/称/要求/建议/主张/认为/指出) -> 整条丢弃, 不进 新增/更新;
  "方态"(事实等条目用于追踪各方表态的字段)若含"调解员"这个键 -> 就地摘除该键, 其余不动。

另: 硬规则(同上, 恒定生效) —— 事实块只留"已发生", 命中即整条丢弃(不转存, 不在任何输出块里出现):
  "事实"描述命中将来安排/承诺/待办高置信信号(次日/届时/先行恢复/X日内完成…, prompt 铁律五交给
    模型判但多版收效有限, 代码兜底) -> 直接丢弃, 不进 新增/更新;
  "事实"描述与本次 diff 里模型自己给的 调解方案.内容/待确认事项 高度重合(同金额+高包含) -> 直接丢弃
    (方案条款不该在事实块里重复登记一遍)。
  以上两条在"更新"里对新老编号一视同仁(判断先于 old 是否存在) —— 否则模型会先合法开一条事实,
    等方案定型后再借"更新"改写老编号描述, 把方案条款绕道塞回事实块(实测过, 命中率100%, 已堵)。

另: 硬规则(同上, 恒定生效) —— 争议焦点只留"实体权利义务的对立", 命中即整条丢弃:
  "争议焦点"的"问题"命中推进流程安排高置信信号(是否进入诉讼/仲裁/信访、要不要起诉、调解本身的
    延期/下次时间地点/出席人/审批权/材料提交/验收安排…, prompt 铁律六交给模型判但实测残留约1/3,
    代码兜底) -> 直接丢弃, 不进 新增/更新; 同样在"更新"里对新老编号一视同仁(判断先于 old 是否存在)。

另: 硬规则(同上, 恒定生效) —— "诉求"金额只留一个干净总额, 命中即就地摘除括注(不丢整条):
  "诉求"里总额数字后面若带"(含A元、B元…)"这种分类明细括注(括注里还能再找到至少一个金额数字才算,
    避免误伤"(不含税)"这类非金额括注) -> 就地删掉整个括注、只留前面的总额数字(prompt 铁律七交给
    模型判但两版收效有限, 代码兜底); 与新增/更新无关, 在两条通道产出的最终条目上统一收尾摘除
    (同 _strip_mediator_fangtai, 合并/锚定把括注带回来也照样摘)。

判重的廉价信号只用程序算(同块 + 金额/当事人 硬键 + 字符 bigram Jaccard), 灰区才升级到基座模型。
返回 (norm_diff, actions), 不修改入参。主调用方应 try/except: 本模块任何异常都回退原始 diff。
DEDUP=0 时上游根本不调本模块; JUDGE=0 时只跑 1)、2) 两层纯规则、不调模型。
"""
import os
import re
import json
import copy
import urllib.request

import conv_prompts_stream_v4g as conv

# ---- 阈值(可调) ----
STRONG = 0.85         # 文本相似度 >= 此 -> 判重(合并)
HARD_STRONG = 0.60    # 硬键命中(同金额+同当事人)时, 合并的较低相似度门槛
GRAY = 0.55           # 相似度 >= 此(且未达合并) -> 新增进灰区
GRAY_SHORT = 0.28     # 短文本(争议焦点的「问题」常只 7~16 字)的灰区门槛。
#   为什么要分开: bigram Jaccard 在短文本上被虚词稀释 —— 同一争议的两种问法
#   ("是否应砍掉槐树" vs "是否应保留槐树")实测只有 0.333, 长文本上同等语义能到 0.7+。
#   用统一的 0.55 会让这类重复**连灰区都进不了、直接放行**(线上争议焦点重复的主因)。
#   ⚠️ 只放宽「够不够格送模型判」的门槛, STRONG/HARD_STRONG 合并门槛一律不动 ——
#   低相似度绝不由规则直接合并, 一律由模型裁决, 避免误合并。
SHORT_LEN = 20        # 归一化后较短者 <= 此长度, 视为短文本
ANCHOR_SIM = 0.98     # 更新: 描述几乎一字不差(>=此) + 同金额同人 + 无否定翻转 -> 廉价规则锚定, 不劳模型
JUDGE_UPD_LOW = 0.45  # 更新: 描述相似度 >= 此(且同金额)且 < ANCHOR_SIM -> 一律下沉给模型判(有改写就交模型)
SHRINK_CONTAIN = 0.70 # 更新: 新描述的字词 >=此比例落在旧描述内(即新是旧的子集/缩水), 且更短、金额只减不增、否定不变 -> 保留更全旧描述
_LIST_DUP_SIM = 0.80  # 列表字段(如"还缺什么信息")内两条目相似度 >= 此 -> 近义候选, 交判官(子串关系也算候选)

# 视作"自由文本描述"的字段(判结构时忽略这些, 只比其余结构字段)
_DESC_KEYS = ("描述", "诉求", "内容", "问题", "证据", "说明")
# 归一时产生的内部标记(判等/判结构时都要剥掉)
_MARK_KEYS = ("疑似重复", "仅措辞", "_合并自", "描述锚定")

# 极小同义词表: 归一时替换, 让"承担/赔偿"这类同义拉高相似度
_SYN = [("承担", "赔偿"), ("补偿", "赔偿"), ("赔付", "赔偿"),
        ("拆除", "拆"), ("拆掉", "拆"), ("返修", "返工")]

# 否定/翻转标记: 规则锚定前兜底 —— old/new 的否定标记集不同 => 可能意思反了(如"赔"→"不赔"),
# 不敢用规则锚定, 一律下沉给模型判。宁可多问一次, 不可锚错。
_NEG = ("不", "未", "没", "拒绝", "否认", "无需", "无法", "并非", "不予")

# 硬规则: "诉求"整条以调解员为主语 -> 不是诉求(调解员不是当事人, 普法/程序性发言不构成诉求)。
# 只认"开头即调解员+这几个动词"这种主语式表达, 不碰"XX要求按调解员建议执行"这类调解员只是宾语/被提及的正常诉求。
_MEDIATOR_CLAIM_RE = re.compile(r"^\s*调解员(提出|称|要求|建议|主张|认为|指出)")

# 硬规则: "事实"描述若命中"将来才会发生的安排/承诺/待办事项", 不算"已发生的事实"(prompt 铁律五
# 交给模型判但 4 版都收效有限, 这里用代码兜底, 与判重/判官无关, 恒定生效)。
# 只认这几类高置信度的"将来"信号, 宁可漏、不可错杀已发生的历史陈述(与本文件其余硬规则同一克制风格)。
_FUTURE_ARRANGE_RE = re.compile(
    r"次日|明日|明天|后天|届时|下次(?:调解|沟通|见面|开庭|会议)?|二次调解|再次调解"
    r"|将于|拟于|拟在|计划于|表示将|承诺(?:将|届时|会)|保证(?:将|届时|会)"
    r"|先行(?:恢复|启动|办理|执行|垫付)|暂先(?:恢复|启动|垫付)"
    r"|将带|携带.{0,6}(?:批复|方案|材料)(?:到场|前来)"
    r"|\d+\s*(?:个)?(?:工作日|日|天|周|个月)内\s*"
    r"(?:完成|办理|到账|检修|修复|整改|处理|答复|回复|转账|支付|赔付|交付|验收)"
)

# 硬规则: "争议焦点"若命中"推进流程安排"(而非实体权利义务), 不算正式争议焦点(prompt 铁律六
# 交给模型判, 实测残留约1/3, 这里用代码兜底, 与判重/判官无关, 恒定生效)。
# 只认铁律六本身列出的这几类高置信度程序性信号, 同一克制风格: 宁可漏、不可错杀实体分歧。
_PROC_ISSUE_RE = re.compile(
    r"是否(?:进入|提起|启动|走)?(?:诉讼|仲裁|信访|上访|投诉)(?:程序|途径)?"
    r"|(?:要不要|该不该|是否|要)(?:走|提起|申请|启动)?起诉"
    r"|(?:是否|要不要)(?:延期|中止|终止|另行安排)(?:调解|沟通|会面|开庭)?"
    r"|(?:下|二|再)次调解(?:的)?(?:时间|地点|安排)?"
    r"|由谁出席|(?:谁|由谁)(?:有|享有)?审批权"
    r"|材料(?:何时|什么时候)?提交(?:时间|安排)?"
    r"|验收(?:如何|怎么|怎样)安排"
    r"|向.{0,8}(?:部门|机构)反映"
)

# 硬规则: "诉求"总额后面带"(含A元、B元…)"分类明细括注的, 就地摘掉括注只留总额(prompt 铁律七
# 交给模型判, 实测两版都收效有限, 这里用代码兜底, 与判重/判官无关, 恒定生效)。
# 要求括注里还能再找到至少一个金额数字才判定命中(避免误伤"(不含税)""(总部让步前的立场)"这类
# 非金额括注), 同一克制风格: 宁可漏、不可错杀正常说明性括注。
_CLAIM_ITEMIZE_RE = re.compile(
    r"(\d+(?:\.\d+)?\s*元)\s*[\(（]\s*(?:含)?\s*[^()（）]*\d+(?:\.\d+)?\s*元[^()（）]*[\)）]"
)

# ---- 灰区判官(基座模型)配置 ----
JUDGE = os.environ.get("JUDGE", "1") not in ("0", "false", "False", "no", "")
JUDGE_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000/v1/chat/completions")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "mediation-raw")   # 用基座, 不用 v4g LoRA
JUDGE_KEY = os.environ.get("BACKEND_API_KEY", "")
JUDGE_TIMEOUT = int(os.environ.get("JUDGE_TIMEOUT", "60"))
JUDGE_MAX_PAIRS = int(os.environ.get("JUDGE_MAX_PAIRS", "24"))

_JUDGE_FMT = ("只输出一个JSON对象, 键必须是给定的id本身, 值是true/false, 不要任何解释, "
              "不要套一层再写{\"id\":...,\"value\":...}。"
              "例如待判id为\"x1\"且判同一件事, 应输出{\"x1\": true}, 不是{\"id\":\"x1\",\"value\":true}。")

_JUDGE_SYS = ("你是文本判等助手。判断每一对文本是否表达同一件事"
              "(仅措辞、详略、语序不同视为同一件事; 事实、金额、时间、责任、主体不同视为不同的事)。"
              + _JUDGE_FMT)

# 争议焦点判等: 焦点是「待解决的问题/议题」, 与描述级判等语义不同 ——
# 同一个议题的**正反两种问法**(砍掉/保留、合法/违法、生效/无效、承担/分担)问的是同一件事,
# 必须判 true; 描述级那套"责任/主体不同即不同"会把正反问法误判成两回事。
_JUDGE_SYS_ISSUE = ("你是争议焦点判等助手。每一对是同一起纠纷里的两个「争议焦点」(待解决的议题)。"
                    "判断两者是否指向**同一个待解决议题**:\n"
                    "- 同一议题(判 true): 仅措辞/详略/语序不同; 一方带金额一方不带但指同一笔; "
                    "同一议题的正反问法(如 是否应砍掉X / 是否应保留X, 程序是否合法 / 程序是否违法, "
                    "是否生效 / 是否无效); 同义词替换(承担/分担, 补偿/补助, 清除/清理)。\n"
                    "- 不同议题(判 false): 针对**不同标的或不同事项**(如 拆除 vs 赔偿, 雨棚清洗 vs 四件套赔偿, "
                    "移机 vs 噪音, 退款金额 vs 涨价程序); 金额指向明显不同的两笔费用。\n"
                    + _JUDGE_FMT)

# 列表条目(如"还缺什么信息""证据清单"里的项)判等: 语义和描述级不同 ——
# 加了人名/时间/编号等限定词只是更具体, 仍是同一项(不像描述级把'主体不同'当作不同事)。
_JUDGE_SYS_LIST = ("你是清单条目判等助手。每一对是某清单(如'还缺的材料''证据清单')里的两个条目。"
                   "判断它们是否指向同一项: 若一条是另一条的更具体表述(加了人名、时间、编号等限定词, "
                   "例如'施工日志'与'马建军施工日志'、'损失评估单'与'401损失评估单'), 视为同一项(true); "
                   "只有指向不同材料/不同事项时才算不同(false)。"
                   + _JUDGE_FMT)


def _norm_text(s):
    s = str(s or "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[，。、,\.;；:：!！?？\"'“”‘’（）()【】\[\]]", "", s)
    for a, b in _SYN:
        s = s.replace(a, b)
    return s


def _amounts(s):
    """只收带 元/万 单位的金额, 归一成整数集合。房号/日期等裸数字不算, 避免误判。"""
    s = str(s or "")
    out = set()
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*万", s):
        out.add(int(round(float(m.group(1)) * 10000)))
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*元", s):
        out.add(int(round(float(m.group(1)))))
    return out


def _pcts(s):
    """只收"X折"这种折扣数(如 6.5折 -> "6.5"), 归一成字符串集合。_amounts()只认元/万,
    对纯折扣条款(不带元数的那半句)是瞎的, 这里单独给 _dup_of_plan_content 补一个硬键维度。"""
    s = str(s or "")
    return set(m.group(1) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*折", s))


def _bigrams(s):
    if len(s) >= 2:
        return set(s[i:i + 2] for i in range(len(s) - 1))
    return {s} if s else set()


def _sim(a, b):
    A, B = _bigrams(a), _bigrams(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _contain(a, b):
    """a 的 bigram 有多大比例落在 b 内(= a 是不是 b 的子集/缩水版)。缩水判定用, 不受 b 多出的内容拖累。"""
    A = _bigrams(a)
    if not A:
        return 0.0
    return len(A & _bigrams(b)) / len(A)


def _neg_set(s):
    """描述里出现的否定标记集合。规则锚定前比 old/new 是否一致, 防"不"字翻转被误锚。"""
    s = str(s or "")
    return frozenset(m for m in _NEG if m in s)


def _desc_key(item):
    for k in _DESC_KEYS:
        v = item.get(k)
        if isinstance(v, str) and v:
            return k
    return None


def _desc_of(item):
    k = _desc_key(item)
    if k is not None:
        return item.get(k)
    # 兜底: 除 编号/方态/当前状态 外的字符串字段拼起来
    return " ".join(str(v) for k, v in item.items()
                    if k not in ("编号", "方态", "当前状态") and isinstance(v, str))


def _roster_persons(dangshiren):
    """'姓名(申请人方) ; 姓名(被申请人方)' -> {姓名, ...}"""
    roster = conv._roster(dangshiren)
    names = set()
    for seg in re.split(r"[;；]", str(roster or "")):
        m = re.match(r"\s*([^\(（]+)", seg)
        if m:
            n = m.group(1).strip()
            if n:
                names.add(n)
    return names


def _persons_in(item, roster_persons):
    txt = json.dumps(item, ensure_ascii=False)
    return {p for p in roster_persons if p and p in txt}


def _struct_only(item):
    """剥掉描述类字段与编号/内部标记, 留下结构字段(状态/方态/…)。"""
    return {k: v for k, v in item.items()
            if k not in _DESC_KEYS and k not in ("编号",) and k not in _MARK_KEYS}


def _strip_marks(item):
    return {k: v for k, v in item.items() if k not in _MARK_KEYS}


def _canon(x):
    """规范化用于判等: 剥内部标记 + 列表按'集合'处理(去重 + 忽略顺序)。
    这样'还缺什么信息'这类清单字段, 仅重排 / 仅内部重复不算变化(空更新)。"""
    if isinstance(x, list):
        uniq = []
        for e in x:
            ce = _canon(e)
            if ce not in uniq:
                uniq.append(ce)
        return sorted(uniq, key=lambda z: json.dumps(z, ensure_ascii=False, sort_keys=True))
    if isinstance(x, dict):
        return {k: _canon(v) for k, v in x.items() if k not in _MARK_KEYS}
    return x


def _items_equal(a, b):
    """整条判等: 忽略内部标记, 且列表字段按集合比(去重 + 无视顺序)。用于空更新过滤。"""
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return a == b
    return _canon(a) == _canon(b)


def _list_near(a, b):
    """列表内两条目是否'近义候选'(子串关系或高相似, 交判官定夺)。完全相同的由规则层先去掉。"""
    na, nb = _norm_text(a), _norm_text(b)
    if not na or not nb or na == nb:
        return False
    if na in nb or nb in na:                     # 一个是另一个子串(如"施工日志"⊂"马建军施工日志")
        return True
    return _sim(na, nb) >= _LIST_DUP_SIM


def _dedupe_list_exact(item, actions):
    """就地去掉 item 各 list 字段内 norm 后完全相同的条目(保留首次出现)。纯规则, 幂等。"""
    if not isinstance(item, dict):
        return
    for k, v in list(item.items()):
        if k in _MARK_KEYS or not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            continue
        seen, nl = set(), []
        for x in v:
            nk = _norm_text(x)
            if nk in seen:
                actions.append({"动作": "列表去重", "编号": item.get("编号"), "字段": k, "删": x})
                continue
            seen.add(nk)
            nl.append(x)
        if len(nl) != len(v):
            item[k] = nl


def _find_by_bh(ns, bh):
    if not bh:
        return None
    for b in conv._LIST_KEYS:
        for it in ns.get(b, []):
            if isinstance(it, dict) and it.get("编号") == bh:
                return it
    return None


def _gray_thr(a, b):
    """灰区门槛按文本长度自适应(见 GRAY_SHORT 注释)。只影响"是否送模型判"。"""
    return GRAY_SHORT if min(len(a), len(b)) <= SHORT_LEN else GRAY


def _find_dup(item, cand, roster_persons):
    """在同块已有条目 cand 里找最相似者; 返回 (exist, score, hard) 或 None。"""
    d_new = _norm_text(_desc_of(item))
    amt_new = _amounts(_desc_of(item))
    per_new = _persons_in(item, roster_persons)
    best = None
    for ex in cand:
        if not isinstance(ex, dict):
            continue
        if ex.get("编号") == item.get("编号"):   # 同编号是"更新"的事, 不算重
            continue
        score = _sim(d_new, _norm_text(_desc_of(ex)))
        amt_ex = _amounts(_desc_of(ex))
        per_ex = _persons_in(ex, roster_persons)
        hard = bool(amt_new) and amt_new == amt_ex and bool(per_new) and per_new == per_ex
        if best is None or score > best[1]:
            best = (ex, score, hard)
    if best and (best[1] >= _gray_thr(d_new, _norm_text(_desc_of(best[0]))) or best[2]):
        return best
    return None


def _fold(exist, newit):
    """把新增条目折进已有: 以已有为底, 状态/方态等结构字段用新值覆盖, 描述保留已有(防抖)。"""
    merged = copy.deepcopy(exist)
    for k, v in newit.items():
        if k == "编号" or k in _DESC_KEYS:
            continue
        merged[k] = v
    merged["_合并自"] = newit.get("编号")
    return merged


def _anchor_desc(newit, old):
    """描述锚定: 保留旧描述, 其余结构字段用新值。返回新 dict。"""
    m = dict(newit)
    k = _desc_key(newit)
    if k is not None:
        ov = old.get(k)
        m[k] = ov if (isinstance(ov, str) and ov) else _desc_of(old)
    m["描述锚定"] = True
    return m


def _extract_json(txt):
    """从模型输出里抠出最外层 JSON(对象或数组, 就近取第一个左括号对应的那一层); 抠不出来返回 None。"""
    starts = [i for i in (txt.find("{"), txt.find("[")) if i >= 0]
    if not starts:
        return None
    i = min(starts)
    close = "}" if txt[i] == "{" else "]"
    j = txt.rfind(close)
    if j <= i:
        return None
    try:
        return json.loads(txt[i:j + 1])
    except Exception:
        return None


def _coerce_verdict(out, ids):
    """把模型输出规整成 {id: bool}。指令要求扁平 {id: true/false, ...}, 但实测:
    只送一对待判时, 模型有时会滑向常见的 {"id": "<id>", "value": true} 包装(把"id"当
    字面字段名而不是我们给的动态键)。两种形状都认, 数组套一层也认; 都不像 -> {}(fail-open)。"""
    idset = set(ids)
    if isinstance(out, dict):
        direct = {k: bool(v) for k, v in out.items() if k in idset}
        if direct:
            return direct
        if "id" in out:
            vid = out.get("id")
            for vk in ("value", "same", "同一件事", "同一项", "是否同一", "result", "答案", "verdict"):
                if vk in out:
                    return {vid: bool(out[vk])}
        return {}
    if isinstance(out, list):
        merged = {}
        for e in out:
            merged.update(_coerce_verdict(e, ids))
        return merged
    return {}


def _llm_judge_same(pairs, system=None):
    """pairs: [{"id","a","b"}]. 问基座'每对是不是同一件事'。返回 {id: bool}。
    system: 判等语义提示(默认描述级'同一件事'; 列表条目传 _JUDGE_SYS_LIST 用'同一项'语义)。
    任何异常/超时/解析失败 -> {} (调用方 fail-open, 一律放行)。"""
    if not pairs:
        return {}
    try:
        usr = json.dumps({"待判": pairs}, ensure_ascii=False)
        payload = {"model": JUDGE_MODEL,
                   "messages": [{"role": "system", "content": system or _JUDGE_SYS},
                                {"role": "user", "content": usr}],
                   "chat_template_kwargs": {"enable_thinking": False},
                   "temperature": 0, "max_tokens": 512}
        headers = {"Content-Type": "application/json"}
        if JUDGE_KEY:
            headers["Authorization"] = "Bearer " + JUDGE_KEY
        req = urllib.request.Request(JUDGE_URL, data=json.dumps(payload).encode(),
                                     headers=headers, method="POST")
        r = json.load(urllib.request.urlopen(req, timeout=JUDGE_TIMEOUT))
        msg = r["choices"][0]["message"]
        txt = (msg.get("content") or msg.get("reasoning_content") or "").strip()
        if "```" in txt:
            mm = re.search(r"```(?:json)?\s*(.*?)```", txt, re.S)
            if mm:
                txt = mm.group(1).strip()
        out = _extract_json(txt)
        return _coerce_verdict(out, [p["id"] for p in pairs]) if out is not None else {}
    except Exception:
        return {}


def _is_mediator_claim(item):
    """诉求条目是否整条以"调解员"为主语提出——调解员不是当事人, 这种条目不该作为诉求存在。"""
    if not isinstance(item, dict) or conv._block_of(item) != "诉求":
        return False
    txt = item.get("诉求")
    return isinstance(txt, str) and bool(_MEDIATOR_CLAIM_RE.match(txt))


def _strip_mediator_fangtai(item, actions):
    """就地摘除"方态"里的"调解员"键——方态只追踪当事人表态, 调解员不该出现在里面。"""
    if not isinstance(item, dict):
        return
    ft = item.get("方态")
    if isinstance(ft, dict) and "调解员" in ft:
        ft.pop("调解员", None)
        actions.append({"动作": "方态过滤(调解员)", "编号": item.get("编号")})


def _strip_claim_itemization(item, actions):
    """就地摘除"诉求"总额后面的分类明细括注(如"2860元(含A元、B元)"->"2860元")——诉求只留一个
    干净总额, 分项名目/金额/凭证等细节该放"事实"分别登记, 不该在诉求里替当事人拼计算过程/秀加总。"""
    if not isinstance(item, dict) or conv._block_of(item) != "诉求":
        return
    t = item.get("诉求")
    if not isinstance(t, str):
        return
    new_t = _CLAIM_ITEMIZE_RE.sub(lambda m: m.group(1), t)
    if new_t != t:
        item["诉求"] = new_t
        actions.append({"动作": "诉求金额去分类明细", "编号": item.get("编号"), "原诉求": t, "新诉求": new_t})


def _is_future_arrangement(item):
    """"事实"条目是否命中"将来才会发生的安排/承诺/待办事项"高置信度信号(见 _FUTURE_ARRANGE_RE)。"""
    if not isinstance(item, dict) or conv._block_of(item) != "事实":
        return False
    d = item.get("描述")
    return isinstance(d, str) and bool(_FUTURE_ARRANGE_RE.search(d))


def _dup_of_plan_content(item, nd):
    """"事实"描述是否与本次 diff 里模型刚给的 调解方案.内容/待确认事项 高度重合——
    要求至少共享一个具体金额或折扣数(硬键, 避免泛泛相似误伤)且字词大半落在方案文本内(_contain), 才判定
    "这条本质是方案条款, 不该在事实块里重复登记一遍"。两种硬键都没提的一律不判(宁可漏, 不可错杀)。"""
    if not isinstance(item, dict) or conv._block_of(item) != "事实":
        return False
    d = _desc_of(item)
    plan = nd.get("调解方案")
    if not isinstance(d, str) or not d or not isinstance(plan, dict):
        return False
    blob = str(plan.get("内容") or "")
    pend = plan.get("待确认事项")
    if isinstance(pend, list):
        blob += " " + " ".join(x for x in pend if isinstance(x, str))
    if not blob.strip():
        return False
    hard_d = _amounts(d) | _pcts(d)
    if not hard_d or not (hard_d & (_amounts(blob) | _pcts(blob))):
        return False
    return _contain(_norm_text(d), _norm_text(blob)) >= 0.55


def _is_procedural_issue(item):
    """"争议焦点"条目是否命中"推进流程安排"高置信度信号(见 _PROC_ISSUE_RE)——
    是否进入诉讼/仲裁/信访、要不要起诉、调解本身的延期/下次时间地点/出席人/审批权/材料提交/验收安排
    等程序性事项, 不是实体权利义务的对立, 不该占争议焦点名额。"""
    if not isinstance(item, dict) or conv._block_of(item) != "争议焦点":
        return False
    q = item.get("问题")
    return isinstance(q, str) and bool(_PROC_ISSUE_RE.search(q))


def normalize(state, diff, dangshiren=""):
    """对模型 diff 做归一。返回 (norm_diff, actions)。不修改入参。"""
    if not isinstance(diff, dict):
        return diff, []
    ns = conv._normalize_state(state)              # 复用 conv: 规整成五块 dict
    roster_persons = _roster_persons(dangshiren)
    actions = []
    nd = copy.deepcopy(diff)

    # ---- 0) 硬规则: 调解员不是当事人/事实块只留"已发生"(与判重/判官无关, 恒定生效) ----
    _new_kept = []
    for _it in nd.get("新增") or []:
        if isinstance(_it, dict) and _is_mediator_claim(_it):
            actions.append({"动作": "诉求过滤(调解员)", "编号": _it.get("编号"), "诉求": _it.get("诉求")})
            continue
        if isinstance(_it, dict) and _is_future_arrangement(_it):
            actions.append({"动作": "事实丢弃(将来安排)", "编号": _it.get("编号"), "描述": _desc_of(_it)})
            continue
        if isinstance(_it, dict) and _dup_of_plan_content(_it, nd):
            actions.append({"动作": "事实丢弃(与方案条款重复)", "编号": _it.get("编号"), "描述": _desc_of(_it)})
            continue
        if isinstance(_it, dict) and _is_procedural_issue(_it):
            actions.append({"动作": "争议焦点丢弃(程序性事项)", "编号": _it.get("编号"), "问题": _it.get("问题")})
            continue
        _new_kept.append(_it)
    nd["新增"] = _new_kept

    # ---- 1) 新增查重 ----
    kept_add = []
    folded = []                # 强命中 -> 折成对已有编号的更新
    gray_add = []              # 灰区待判: (jid, it, exist)
    # 候选池按块起于已有累积状态, 边扫边把"确认为新概念"的条目加进去 ——
    # 这样同一批"新增"里后到的重复项能比对到**先到的那条**, 而不只是比对累积状态
    # (原来的 bug: 同批次两条重复互相从不比较, 各自都查不到"已有"而全部放行)。
    # 只把 hit is None(确认无匹配)的条目入池: 强合并/灰区待判的条目最终身份未定
    # (灰区要等模型裁决, 折叠的条目本身会消失), 提前入池可能让后续条目把"折叠目标"
    # 当更新对象、而该对象最终并不作为独立条目落地。三条以上连环重复挤在同一批
    # 目前没有实例, 暂不处理。
    batch_pool = {b: list(ns.get(b, [])) for b in conv._LIST_KEYS}
    for it in nd.get("新增") or []:
        if not isinstance(it, dict):
            kept_add.append(it)
            continue
        b = conv._block_of(it)
        cand = batch_pool.get(b, []) if b in conv._LIST_KEYS else []
        hit = _find_dup(it, cand, roster_persons)
        if hit is None:
            kept_add.append(it)
            if b in conv._LIST_KEYS:
                batch_pool[b].append(it)
            continue
        exist, score, hard = hit
        if score >= STRONG or (hard and score >= HARD_STRONG):
            folded.append(_fold(exist, it))
            actions.append({"动作": "合并", "新增编号": it.get("编号"),
                            "并入": exist.get("编号"), "相似度": round(score, 3), "硬键": hard})
        elif JUDGE:
            gray_add.append(("A:" + str(it.get("编号")), it, exist))
        else:
            it2 = dict(it)
            it2["疑似重复"] = exist.get("编号")
            kept_add.append(it2)
            actions.append({"动作": "疑似重复", "编号": it.get("编号"),
                            "疑似": exist.get("编号"), "相似度": round(score, 3)})

    # ---- 2) 更新: 并入 folded -> 空更新过滤 -> 描述锚定 -> 灰区待判 ----
    upd = [u for u in (nd.get("更新") or [])]

    def _merge_into(lst, m):
        bh = m.get("编号")
        for u in lst:
            if isinstance(u, dict) and u.get("编号") == bh:
                for k, v in m.items():
                    if k != "编号":
                        u[k] = v
                return
        lst.append(m)

    for m in folded:
        _merge_into(upd, m)

    kept_upd = []
    gray_upd = []              # 灰区待判: (jid, it, old)
    for it in upd:
        if not isinstance(it, dict):
            kept_upd.append(it)
            continue
        if "_合并自" in it:                      # 折叠来的必须保留(承载新增的状态)
            kept_upd.append(it)
            continue
        bh = it.get("编号")
        if _is_mediator_claim(it):                # 同 0): "更新"通道混进来的调解员诉求也要拦(不论新老编号)
            actions.append({"动作": "诉求过滤(调解员)", "编号": bh, "诉求": it.get("诉求")})
            continue
        if _is_future_arrangement(it):            # 同 0): 将来安排事实也要拦——不提前到 old 分支前,
            actions.append({"动作": "事实丢弃(将来安排)", "编号": bh, "描述": _desc_of(it)})  # 老编号借"更新"改写回填方案条款这条路就堵不住
            continue
        if _dup_of_plan_content(it, nd):          # 同 0): 与本次方案条款重复的事实也要拦(不论新老编号)
            actions.append({"动作": "事实丢弃(与方案条款重复)", "编号": bh, "描述": _desc_of(it)})
            continue
        if _is_procedural_issue(it):              # 同 0): 程序性争议焦点也要拦(不论新老编号,
            actions.append({"动作": "争议焦点丢弃(程序性事项)", "编号": bh, "问题": it.get("问题")})  # 防止先合法开一条再借"更新"改写回程序性问法
            continue
        old = _find_by_bh(ns, bh)
        if old is None:                          # 新编号的"更新" -> 交 apply_diff 追加
            kept_upd.append(it)
            continue
        if _items_equal(it, old):                # (1) 空更新 -> 删
            actions.append({"动作": "空更新", "编号": bh})
            continue
        if _desc_of(it) == _desc_of(old):        # 描述没变, 只有结构变 -> 真更新
            kept_upd.append(it)
            continue
        sim = _sim(_norm_text(_desc_of(it)), _norm_text(_desc_of(old)))
        same_amt = _amounts(_desc_of(it)) == _amounts(_desc_of(old))
        same_per = _persons_in(it, roster_persons) == _persons_in(old, roster_persons)
        same_neg = _neg_set(_desc_of(it)) == _neg_set(_desc_of(old))
        new_n, old_n = _norm_text(_desc_of(it)), _norm_text(_desc_of(old))
        # 缩水: 新描述是旧的变短版(高包含 + 更短 + 金额只减不增 + 否定完全不变) -> 保留更全旧描述, 不丢 grounding
        shrink = (len(new_n) < len(old_n)
                  and _contain(new_n, old_n) >= SHRINK_CONTAIN
                  and _amounts(_desc_of(it)) <= _amounts(_desc_of(old))
                  and _neg_set(_desc_of(it)) == _neg_set(_desc_of(old)))
        if sim >= ANCHOR_SIM and same_amt and same_per and same_neg:  # (2) 几乎一字不差且无否定翻转 -> 规则锚定
            anch = _anchor_desc(it, old)
            if _items_equal(anch, old):
                actions.append({"动作": "仅措辞", "编号": bh})    # 锚定后全等 -> 空更新, 删
            else:
                actions.append({"动作": "描述锚定", "编号": bh})
                kept_upd.append(anch)
        elif shrink:                                          # (2b) 缩水 -> 锚定旧描述(更全), 只吃结构变化
            anch = _anchor_desc(it, old)
            if _items_equal(anch, old):
                actions.append({"动作": "仅措辞(缩水)", "编号": bh})
            else:
                actions.append({"动作": "缩水保旧描述", "编号": bh})
                kept_upd.append(anch)
        elif JUDGE and same_amt and sim >= JUDGE_UPD_LOW:    # (3) 灰区(金额一致) -> 判官
            gray_upd.append(("U:" + str(bh), it, old))
        else:
            kept_upd.append(it)                  # 描述差得多/金额变了 -> 真变化, 显示

    # ---- 2.5) 列表字段内部去重: 规则先去精确重复; 近义候选(子串/高相似)收集, 交判官 ----
    out_items = ([x for x in kept_add if isinstance(x, dict)]
                 + [x for x in kept_upd if isinstance(x, dict)])
    for _it in out_items:
        _dedupe_list_exact(_it, actions)         # 纯规则: 列表内完全相同的条目去掉
    gray_list = []                               # 列表近义待判: (jid, item, field, va, vb)
    if JUDGE:
        _lc = 0
        for _it in out_items:
            for _f, _v in list(_it.items()):
                if _f in _MARK_KEYS or not isinstance(_v, list) or not all(isinstance(x, str) for x in _v):
                    continue
                for _i in range(len(_v)):
                    for _j in range(_i + 1, len(_v)):
                        if _list_near(_v[_i], _v[_j]):
                            gray_list.append(("L%d" % _lc, _it, _f, _v[_i], _v[_j]))
                            _lc += 1

    # ---- 3) 灰区调基座判官: 描述近义用'同一件事'语义, 列表条目用'同一项'语义(加限定=同一项), 各一次打包 ----
    if gray_add or gray_upd or gray_list:
        combined = (gray_add + gray_upd)[:JUDGE_MAX_PAIRS]
        _room = max(0, JUDGE_MAX_PAIRS - len(combined))
        list_pairs = gray_list[:_room]           # 描述对优先占额度, 列表对用剩余额度; 超出的不判(fail-open)
        desc_qs = [{"id": jid, "a": _desc_of(it), "b": _desc_of(ex)} for (jid, it, ex) in combined]
        # 争议焦点走专用判等语义(正反问法/同义议题=同一件事); 其余块仍用默认"同一件事"语义
        _issue_ids = set(jid for (jid, it, ex) in combined if conv._block_of(it) == "争议焦点")
        list_qs = [{"id": jid, "a": va, "b": vb} for (jid, _im, _f, va, vb) in list_pairs]
        verdict = {}
        if desc_qs:
            _other_qs = [q for q in desc_qs if q["id"] not in _issue_ids]
            _issue_qs = [q for q in desc_qs if q["id"] in _issue_ids]
            if _other_qs:
                verdict.update(_llm_judge_same(_other_qs))                    # 描述级: 主体/金额不同=不同
            if _issue_qs:
                verdict.update(_llm_judge_same(_issue_qs, _JUDGE_SYS_ISSUE))  # 争议焦点: 正反问法/同义=同一议题
        if list_qs:
            verdict.update(_llm_judge_same(list_qs, _JUDGE_SYS_LIST))  # 清单级: 加限定词=同一项
        judged = set(p["id"] for p in desc_qs) | set(p["id"] for p in list_qs)
        # 3a) 新增灰区
        for jid, it, exist in gray_add:
            same = verdict.get(jid)
            if jid not in judged:                # 超出打包上限, 没送判 -> 保留提示
                it2 = dict(it); it2["疑似重复"] = exist.get("编号")
                kept_add.append(it2)
                actions.append({"动作": "疑似重复(未送判)", "编号": it.get("编号"), "疑似": exist.get("编号")})
            elif same is True:                   # 同一件事 -> 折成更新
                _merge_into(kept_upd, _fold(exist, it))
                actions.append({"动作": "合并(模型判)", "新增编号": it.get("编号"), "并入": exist.get("编号")})
            elif same is False:                  # 不同 -> 保留为新增
                kept_add.append(it)
                actions.append({"动作": "非重复(模型判)", "新增编号": it.get("编号"), "对比": exist.get("编号")})
            else:                                # 判不出 -> 放行, 但留提示
                it2 = dict(it); it2["疑似重复"] = exist.get("编号")
                kept_add.append(it2)
                actions.append({"动作": "疑似重复(判官未定)", "编号": it.get("编号"), "疑似": exist.get("编号")})
        # 3b) 更新灰区
        for jid, it, old in gray_upd:
            same = verdict.get(jid)
            if jid in judged and same is True:   # 同一件事 -> 锚定旧描述
                anch = _anchor_desc(it, old)
                if _items_equal(anch, old):
                    actions.append({"动作": "仅措辞(模型判)", "编号": it.get("编号")})
                else:
                    actions.append({"动作": "描述锚定(模型判)", "编号": it.get("编号")})
                    kept_upd.append(anch)
            elif jid not in judged:              # 超出打包上限, 没送判 -> 原样放行
                kept_upd.append(it)
                actions.append({"动作": "更新(未送判)", "编号": it.get("编号")})
            elif same is False:                  # 模型判非仅措辞变化 -> 原样放行
                kept_upd.append(it)
                actions.append({"动作": "更新(模型判-非仅措辞)", "编号": it.get("编号")})
            else:                                # 判不出 -> 显示新描述(fail-open)
                kept_upd.append(it)
                actions.append({"动作": "更新(判官未定)", "编号": it.get("编号")})

    # 3c) 列表近义: 判"同一项" -> 只留更具体(更长)的那条, 删冗余(裸模只投票, 内容一字不重写)
    if gray_list:
        for (jid, _im, _f, va, vb) in list_pairs:
            if verdict.get(jid) is True:
                lst = _im.get(_f)
                if isinstance(lst, list):
                    drop, keep = (va, vb) if len(_norm_text(va)) <= len(_norm_text(vb)) else (vb, va)
                    if drop in lst:
                        _im[_f] = [x for x in lst if x != drop]
                        actions.append({"动作": "列表近义合并(模型判)", "编号": _im.get("编号"),
                                        "字段": _f, "删": drop, "留": keep})

    nd["新增"] = kept_add
    nd["更新"] = kept_upd
    for _it in nd["新增"] + nd["更新"]:            # 收尾: 覆盖 Part3 折叠/锚定新加的条目(幂等)
        _dedupe_list_exact(_it, actions)
        _strip_mediator_fangtai(_it, actions)      # 硬规则: 方态里不该有调解员(合并/锚定可能把它带回来)
        _strip_claim_itemization(_it, actions)     # 硬规则: 诉求总额不带分类明细括注(合并/锚定可能把它带回来)
    # 收尾: 剥掉内部标记(描述锚定/疑似重复/仅措辞/_合并自) —— 它们只用于本模块内部流转与
    # actions 审计, 绝不能出现在对外 diff/data 里(否则前端把它当业务字段渲染成"(本批新增)")。
    nd["新增"] = [_strip_marks(_it) for _it in nd["新增"]]
    nd["更新"] = [_strip_marks(_it) for _it in nd["更新"]]
    return nd, actions
