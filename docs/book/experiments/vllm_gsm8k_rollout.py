#!/usr/bin/env python3
"""Full one-trajectory-per-prompt GSM8K rollout with vLLM."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-1.7B')
    ap.add_argument('--output', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--max-new-tokens', type=int, default=512)
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    ap.add_argument('--max-model-len', type=int, default=4096)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    ds = load_dataset('openai/gsm8k', 'main', split=args.split)
    prompts = [tok.apply_chat_template([{'role':'user','content':x['question']}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for x in ds]
    llm = LLM(model=args.model, dtype='bfloat16', gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len, trust_remote_code=True)
    params = SamplingParams(temperature=args.temperature, top_p=1.0, max_tokens=args.max_new_tokens, min_p=0.0, stop=None)
    results = llm.generate(prompts, params)
    rows=[]
    for i,(ex,res) in enumerate(zip(ds,results)):
        o=res.outputs[0]
        rows.append({'example_id': i, 'question': ex['question'], 'answer': ex['answer'], 'prompt_token_ids': res.prompt_token_ids, 'generated_token_ids': o.token_ids, 'generated_text': o.text})
    (out/'rollouts.jsonl').write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in rows)+'\n')
    (out/'metadata.json').write_text(json.dumps({'model':args.model,'split':args.split,'n_examples':len(rows),'temperature':args.temperature,'top_p':1.0,'max_new_tokens':args.max_new_tokens,'engine':'vllm','definition':'one sampled trajectory per GSM8K test prompt; generation is complete up to EOS or max_new_tokens'}, indent=2))
    print(json.dumps({'n_examples':len(rows),'tokens':sum(len(x['generated_token_ids']) for x in rows)}, indent=2))
if __name__=='__main__': main()
