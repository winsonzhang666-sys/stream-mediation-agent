# v4g LoRA adapter · provenance

流式纠纷调解辅助模型的 LoRA 适配器(**diff 增量版**,v4g)。

| 项 | 值 |
|---|---|
| 基座 | `Qwen3.6-35B-A3B`(MoE,架构 `qwen3_5_moe`,~35B 总参 / ~3B 激活;需自备,见仓库 README「前置依赖」) |
| 微调方式 | LoRA(PEFT 0.18.1),`r=64` / `alpha=128` / `dropout=0.1`,`lora_target=all` |
| 训练 | SFT,3 epoch,lr 1e-4 cosine,有效 batch 32,ZeRO-3;template `qwen3_5_nothink` |
| best checkpoint | 由 eval_loss 选出(`load_best_model_at_end`),**best eval_loss ≈ 0.4625** |
| 是否 merge | **否** —— 直接以 base + LoRA 热挂方式 serve(见 `serve/serve_vllm_lora.sh`) |
| 配套 SYS | `prompts/system_v4g.txt` —— **训练 / 推理必须逐字节一致**(train==serve 单一事实源) |

## 文件
- `adapter_config.json` —— PEFT 配置。`base_model_name_or_path` 已改为裸名 `Qwen3.6-35B-A3B`;vLLM 实际以 `--model` 指定 base 路径,此字段仅信息性,不影响加载。
- `adapter_model.safetensors` —— 适配器权重(~172 MiB,经 Git LFS 分发,见仓库根 `.gitattributes`)。

## 与 v4f 的差异(为什么是 v4g)
v4f 上线暴露一个覆盖缺口:**背靠背 / 一对一单独沟通场景下,同阵营第二个当事人提出与第一人高度同构的诉求时,会被误当成"已有诉求的更新"而吞掉**(前端表现为"什么都没发生")。v4g 在数据侧补入这类「同阵营多人-同构诉求」反例,并把「**换人 = 新号**」这条铁律焊进 SYS 契约(数据 + prompt 双管),使各当事人的同类诉求各自独立成条、各自新编号,不再并入他人编号。

> ⚠️ 推理时**必须**使用 `prompts/system_v4g.txt` 作为 system prompt,并对每次请求传 `chat_template_kwargs={"enable_thinking": false}`(nothink 模型)。二者任一缺失都会与训练分布失配。
