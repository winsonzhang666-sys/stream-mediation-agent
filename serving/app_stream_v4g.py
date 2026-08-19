# -*- coding: utf-8 -*-
"""Mediation Stream v4g 业务封装 —— 纠纷调解「流式辅助（增量版 / diff）」。

v4g: 契约、diff 合并逻辑、字段沿用 v4→v4d 那套; **prompt 用 v4g 专属的一份**
(import conv_prompts_stream_v4g -> stream_system_v4g.txt, md5 cd0de265 / 3434B;
 与 v4d/v4b 共用的 98d28c33 不是同一份)。
后端 model=mediation-stream-v4g (base + LoRA v4g adapter, best checkpoint-200)。

输入 (无案由大类):
  当事人:        名册。str("姓名(申请人方) ; 姓名(被申请人方)") 或 list[{"姓名","方"}]
  新对话:        自上次以来**新增**的对话。str(逐行 "turn_001 姓名: 文本") 或 list[{...}]
                 (兼容别名: 对话)
  当前累积状态:  可选。上次本接口返回的 data(全量五块快照); 首次不传/传空。
                 (兼容别名: 上一版当前状态 / data —— 可把上次返回的 data 原样回传)
输出:
  data: 合并 diff 后的**全量五块快照**(诉求/事实/争议焦点/证据/调解方案) —— 直接渲染, 下轮原样回传当 当前累积状态。
  diff: 模型本轮产出的变化({新增,更新,调解方案}) —— 需要“只看这轮变了什么”时用。
采样对齐 eval: temperature=0。
"""
import os
import json
import time
import logging
from typing import Optional, List, Union

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

import conv_prompts_stream_v4g as conv
import normalize_diff_v4g as normz
try:                              # 请求留存模块(可选; 不随仓发布——落盘内容为真实调解对话)
    import reqlog                 # 自备实现需提供 save(payload, result) / 同名签名; SAVE_LOG=0 可关
except ImportError:               # 缺失时降级为 no-op, 不影响主流程
    class _NoReqLog:
        def __getattr__(self, _name):
            return lambda *a, **kw: None
    reqlog = _NoReqLog()

logger = logging.getLogger("mediation-stream-v4g")

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000/v1/chat/completions")
MODEL_NAME = os.environ.get("MODEL_NAME", "mediation-stream-v4g")
API_KEY = os.environ.get("MEDIATION_API_KEY", "")
BACKEND_API_KEY = os.environ.get("BACKEND_API_KEY", "")   # 后端裸口带 --api-key 时用它鉴权
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "180"))

# 对齐 eval: temperature=0 (贪心, 确定性). diff 通常较短, 首步接近全量 -> 给足防截断。
SAMPLING = {"temperature": 0, "max_tokens": 3072}
DIFF_KEYS = ("新增", "更新", "调解方案")

# 封装层 diff 归一(新增查重+更新查措辞); 纯后处理, 不碰模型。DEDUP=0 可退回原始行为。
DEDUP = os.environ.get("DEDUP", "1") not in ("0", "false", "False", "no", "")

app = FastAPI(title="Mediation Stream v4g API", version="v4g")

# 允许浏览器测试页跨源调用(演示端点无鉴权; 正式端点仍需 Bearer, CORS 不绕过鉴权)。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# 测试台静态页(同源直连本封装, 免 CORS/免 Key); 请求时读盘, 改 HTML 无需重启。
DEMO_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_v4g.html")
# 研判台静态页(参考 index-source 版式; 同一套 /demo/generate 契约, 同样读盘不用重启)。
INDEX_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index_v4g.html")
# 裸模调试台(纯静态页; 浏览器**直连 8000 vLLM**, 不经本封装的任何逻辑,
# 本服务只负责把这个 HTML 吐出去 —— 相当于顺带当了回静态文件服务器)。
BARE_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bare_llm.html")


def _read_page(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="page not found: %s" % os.path.basename(path))


@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    return _read_page(DEMO_HTML)


@app.get("/index", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
def index_page():
    """实时研判台(新版前端, 参考 index-source 版式)。/ 仍保持原 JSON 服务说明不变。"""
    return _read_page(INDEX_HTML)


@app.get("/bare", response_class=HTMLResponse)
def bare_page():
    """裸模调试台。⚠️ 注意: 本路由只是把静态 HTML 发给浏览器;
    页面里的推理请求由**浏览器直接打 8000 vLLM**, 完全不经过本封装
    (不拼 prompt / 不累积状态 / 不做 diff 归一)。"""
    return _read_page(BARE_HTML)


class GenerateRequest(BaseModel):
    当事人: Union[str, List[dict]]
    新对话: Union[str, List[dict]] = ""
    对话: Union[str, List[dict]] = ""                      # 兼容别名
    当前累积状态: Optional[Union[str, dict]] = None
    上一版当前状态: Optional[Union[str, dict]] = None      # 兼容别名(v3 习惯)
    data: Optional[Union[str, dict]] = None                # 允许把上次返回的 data 直接回传


def _check_auth(authorization: Optional[str]) -> None:
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def _extract_text(message: dict) -> str:
    return (message.get("content")
            or message.get("reasoning_content")
            or message.get("reasoning")
            or "")


def _pick_dialogue(req: GenerateRequest):
    for v in (req.新对话, req.对话):
        if v not in (None, "", []):
            return v
    return ""


def _pick_state(req: GenerateRequest):
    for v in (req.当前累积状态, req.上一版当前状态, req.data):
        if v not in (None, "", {}, []):
            return v
    return None


@app.get("/")
def root():
    return {"service": "mediation-stream-v4g",
            "usage": "POST /api/mediation/generate  body: {当事人, 新对话, 当前累积状态?}"}


@app.get("/health")
def health():
    return {"status": "ok"}


def _core(req: GenerateRequest) -> dict:
    state = _pick_state(req)
    system = conv.build_system()
    user = conv.build_user(req.当事人, _pick_dialogue(req), state)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        **SAMPLING,
    }
    try:
        headers = {"Authorization": f"Bearer {BACKEND_API_KEY}"} if BACKEND_API_KEY else None
        resp = requests.post(BACKEND_URL, json=payload, timeout=REQUEST_TIMEOUT, headers=headers)
        resp.raise_for_status()
        result = _extract_text(resp.json()["choices"][0]["message"])
    except Exception as e:
        logger.error("backend call failed: %s", e)
        raise HTTPException(status_code=502, detail="模型服务调用失败")

    # 模型输出应为 diff JSON
    try:
        diff = json.loads(result)
    except Exception:
        return {"code": 502, "message": "模型输出非合法JSON", "result": result}
    if not isinstance(diff, dict) or not any(k in diff for k in DIFF_KEYS):
        return {"code": 502, "message": "模型输出不含 新增/更新/调解方案", "result": result}

    # 封装层归一(仅后处理模型输出, 不影响模型判断): 新增查重 + 更新查措辞。
    diff_raw = diff
    actions = []
    if DEDUP:
        try:
            diff, actions = normz.normalize(state, diff, req.当事人)
        except Exception as e:                       # 归一失败 -> 回退原始 diff, 主流程不受影响
            logger.error("normalize failed: %s", e)
            diff, actions = diff_raw, []

    # 合并 diff -> 全量五块快照
    data = conv.apply_diff(state, diff)
    resp = {"code": 0, "message": "success", "data": data, "diff": diff}
    if actions:                                       # 有归一动作时才附带, 保持常规响应形状不变
        resp["_normalize"] = actions
        resp["diff_raw"] = diff_raw
    return resp


def _core_logged(req: GenerateRequest, endpoint: str, request: Request = None):
    """_core 外面套一层留存：记输入 + 输出 + 耗时 + 来源IP。
    留存失败绝不影响接口本身（reqlog.save 内部已吞异常）。"""
    t0 = time.time()
    # pydantic v2 用 model_dump()，v1 用 dict()，两边都兼容
    body = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    ip = ""
    try:
        if request is not None and request.client:
            ip = request.client.host
    except Exception:
        pass
    try:
        resp = _core(req)
    except Exception as e:
        reqlog.save(endpoint, body, None, time.time() - t0, ip, err=str(e))
        raise
    reqlog.save(endpoint, body, resp, time.time() - t0, ip)
    return resp


@app.post("/api/mediation/generate")
def generate(req: GenerateRequest, request: Request,
             authorization: Optional[str] = Header(default=None)):
    _check_auth(authorization)
    return _core_logged(req, "/api/mediation/generate", request)


@app.post("/demo/generate")
def demo_generate(req: GenerateRequest, request: Request):
    return _core_logged(req, "/demo/generate", request)
