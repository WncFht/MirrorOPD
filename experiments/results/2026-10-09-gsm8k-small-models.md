# GSM8K 小模型基线

## 测试协议

使用 `openai/gsm8k` 的 `test` split 全部 1319 个题目，不划分 validation。每个题目生成一条轨迹，temperature 为 0，`max_new_tokens=1024`，`max_model_len=2048`，vLLM 使用 bfloat16。Qwen2.5-0.5B 的 base 模型使用脚本对无 chat template 的 tokenizer 的 fallback prompt；其余 instruct 模型使用各自的 chat template。

严格准确率只接受模型输出中的 `#### number` 与数据集答案一致。由于不同 instruct 模型的答案格式不同，另报告宽松准确率：优先取 `####`，其次取最后一个 `\\boxed{number}`，最后取文本中最后一个数字。宽松规则会受到重复生成和解释末尾数字影响，不能替代严格指标。

## 结果

| 模型                  | 参数规模 | 严格准确率 | 宽松准确率 | 宽松正确数 | 解析率（`####`） | 平均生成 token |  P50 |  P95 | 1024 截断率 |
| --------------------- | -------: | ---------: | ---------: | ---------: | ---------------: | -------------: | ---: | ---: | ----------: |
| Qwen2.5-0.5B base     |    0.49B |     0/1319 |      28.6% |        286 |            0.30% |          826.3 | 1024 | 1024 |       75.1% |
| Qwen2.5-0.5B-Instruct |    0.49B |     0/1319 |      49.4% |        651 |            0.00% |          308.1 |  287 |  494 |        1.3% |
| SmolLM2-1.7B-Instruct |     1.7B |    30/1319 |      43.2% |        570 |            3.34% |          194.2 |  169 |  380 |        0.8% |

SmolLM2-135M/360M-Instruct 的同协议结果分别为严格 0.23%/1.29%，宽松 1.59%/9.02%，用于说明参数规模对当前任务的影响。

严格指标对 Qwen2.5-Instruct 不适用，因为它主要输出 `\\boxed{}` 而不是 `####`；因此该行的 49.4% 是当前最有参考价值的 GSM8K 端到端数字匹配结果，但仍应在后续使用官方 GSM8K extractor 复核。

## 运行产物

- `~/src/mirroropd/runs/gsm8k-qwen25-0p5b-test-greedy-1024`
- `~/src/mirroropd/runs/gsm8k-qwen25-0p5b-instruct-test-greedy-1024`
- `~/src/mirroropd/runs/gsm8k-smollm2-1p7b-instruct-test-greedy-1024`

## 解读

Qwen2.5-0.5B base 在没有 instruction tuning 的情况下出现长循环，因此不能作为当前 OPD rollout 的可用 student。Qwen2.5-0.5B-Instruct 与 SmolLM2-1.7B-Instruct 都能稳定结束生成；前者更适合在 1B 教师上限下做同 tokenizer 的 base→instruct 对照，后者的模型能力更强但 tokenizer 与 Qwen 系列不兼容。
