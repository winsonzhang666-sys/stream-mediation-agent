#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""流式调解 diff 模型 · 极简推理 demo(纯标准库,零 pip 依赖)。

演示一次「按轮增量」推理:喂 名册 + 当前累积状态 + 本轮新对话 → 模型只吐本轮 diff。
用例是一个合成案(张三 vs 某物业公司,与 docs/数据格式_输入输出_纯格式版.md 完全一致),
不依赖任何金标数据,起好服务即可跑通。

用法:
    # 先起服务(见 serve/serve_vllm_lora.sh),再:
    python serve/demo_infer.py
环境变量(都有默认,绝不硬编真实凭证):
    VLLM_BASE     OpenAI 兼容端点,默认 http://localhost:8000/v1
    MODEL         served model 名,默认 mediation-stream-v4g
    VLLM_API_KEY  鉴权 token,默认哑值 "x"(本地裸 vLLM 通常不校验)
"""
import os, json, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SYS_PATH = os.path.join(HERE, "..", "prompts", "system_v4g.txt")

VLLM_BASE = os.environ.get("VLLM_BASE", "http://localhost:8000/v1")
MODEL     = os.environ.get("MODEL", "mediation-stream-v4g")
API_KEY   = os.environ.get("VLLM_API_KEY", "x")

# ── 输入三块(与 build_sft 训练时的拼装格式逐字一致)────────────────────────────
# 名册:一行,分号隔开 "姓名(方) ; 姓名(方)",以 【当事人】 开头
ROSTER = "【当事人】张三(申请人方) ; 某物业公司(被申请人方)"

# 上一步累积出的全量五块状态(首步应为字符串 "(空,这是第一步)")
PREV_STATE = {
    "诉求": [{"编号": "claim_001", "诉求": "张三要求先修好楼道灯再补交物业费", "当前状态": "对方提出反方案"}],
    "事实": [{"编号": "fact_001", "描述": "张三近两年未缴物业费，共2400元", "当前状态": "双方确认",
              "方态": {"张三": "认可", "某物业公司": "认可"}}],
    "争议焦点": [{"编号": "issue_001", "问题": "是否应先修好楼道灯再补缴物业费", "当前状态": "争议中",
                  "还缺什么信息": ["楼道灯损坏的责任与时间"]}],
    "证据": [{"编号": "evidence_001", "证据": "某物业公司提供的欠费明细，用以证明欠费2400元", "提交状态": "已提及未上传"}],
    "调解方案": {"内容": "", "待确认事项": [], "当前状态": "内部草案",
                 "策略话术": {"当前障碍": "双方卡在先修还是先补", "策略目标": "用限期修缮换补缴承诺",
                              "话术建议": [{"维度": "利", "参考话术": "灯一周修好，修好你再补，不吃亏。"}]}},
}

# 本轮新增对话:每行 "turn_编号 说话人: 文本"
NEW_DIALOGUE = "\n".join([
    "turn_007 某物业公司: 灯我们一周内修好。修好之后，欠的2400能不能三天内补上？",
    "turn_008 张三: 只要灯真修好，我三天内补2400，没问题。",
    "turn_009 调解员: 那就这么定：物业一周内修好楼道灯，张三在修好后三天内补缴2400元，再定个验收方式。",
])


def build_user(roster, prev_state, new_dialogue, first_step=False):
    """复刻训练侧 build_sft_v4.convert 的 user 拼装:名册 + 当前累积状态 + 新对话。"""
    if first_step:
        state_block = "(空,这是第一步)"
    else:
        state_block = json.dumps(prev_state, ensure_ascii=False)
    return f"{roster}\n\n【当前累积状态】\n{state_block}\n\n【新对话】\n{new_dialogue}"


def call_model(system, user):
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": 2048,
        # ★ nothink 模型:每次请求必带,缺失即与训练分布失配
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{VLLM_BASE}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]


def extract_json(text):
    """从模型输出里取出 diff JSON(容错:剥掉可能的 ``` 围栏 / 前后杂字)。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        return json.loads(t[i:j + 1])


_PREFIX2ARR = {"claim": "诉求", "fact": "事实", "issue": "争议焦点", "evidence": "证据"}


def apply_diff(state, diff):
    """封装层职责的最小演示:把本步 diff 累加进全量态(新增追加 / 更新就地改 / 方案整块换)。
    真实系统里这一步还要做 空更新过滤 + 一致性回归(见 README §三),此处从简。"""
    for item in diff.get("新增", []):
        arr = _PREFIX2ARR.get(str(item.get("编号", "")).split("_")[0])
        if arr:
            state.setdefault(arr, []).append(item)
    for item in diff.get("更新", []):
        arr = _PREFIX2ARR.get(str(item.get("编号", "")).split("_")[0])
        if not arr:
            continue
        lst = state.setdefault(arr, [])
        for k, old in enumerate(lst):
            if old.get("编号") == item.get("编号"):
                lst[k] = item
                break
        else:
            lst.append(item)  # 更新了一个不存在的编号(异常):兜底追加
    if "调解方案" in diff:
        state["调解方案"] = diff["调解方案"]
    return state


def main():
    system = open(SYS_PATH, encoding="utf-8").read()
    user = build_user(ROSTER, PREV_STATE, NEW_DIALOGUE, first_step=False)

    print("=" * 70)
    print("【system】(prompts/system_v4g.txt,前 120 字)\n" + system[:120] + " ...")
    print("=" * 70)
    print("【user】\n" + user)
    print("=" * 70)

    raw = call_model(system, user)
    print("【模型原始输出(本轮 diff)】\n" + raw)
    print("=" * 70)

    diff = extract_json(raw)
    print("【解析后的 diff】\n" + json.dumps(diff, ensure_ascii=False, indent=2))
    print("=" * 70)

    import copy
    full = apply_diff(copy.deepcopy(PREV_STATE), diff)
    print("【封装层回灌后的全量态(作为下一步的【当前累积状态】)】\n"
          + json.dumps(full, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
