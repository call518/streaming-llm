import warnings
import torch
import argparse
import json
import os
import re
import sys
import subprocess
import time
from tqdm import tqdm
from streaming_llm.utils import load
from streaming_llm.enable_streaming_llm import enable_streaming_llm

warnings.filterwarnings("ignore")

#LOGFILE = "/var/log/messages"
#LOGFILE = "/var/log/ambari-agent/ambari-agent.log"
#LOGFILE = "/var/log/*.log /var/log/*/*.log /var/log/*/*/*.log"
LOGFILE = "/var/log/messages"

BATCH_SIZE = 1  # 30줄씩 배치 처리

# TCPDUMP 실시간 패킷 캡처 및 분석
def get_tail_data():
    process = subprocess.Popen(
        f"sudo tail -n0 -q -F {LOGFILE}", shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1
    )

    batch = []
    for line in iter(process.stdout.readline, ""):
        line = line.strip()
        if line:
            batch.append(line)  # JSON 변환 없이 원본 데이터 저장

        if len(batch) >= BATCH_SIZE:
            yield batch
            batch = []  # 배치 초기화

@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len):
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0

    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())

        generated_text = tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip().split(" ")

        now = len(generated_text) - 1
        if now > pos:
            print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break

    print(" ".join(generated_text[pos:]), flush=True)
    return past_key_values

system_prompt = f"The following is a recent part of the Log file ({LOGFILE}). Analyze the content, and if there is an Error, Failed, or Warning related content, output 'ISSUE' and the review result of the content briefly, otherwise output only 'NORMAL'. Do not output any other response."

@torch.no_grad()
def streaming_inference(model, tokenizer, kv_cache=None, max_gen_len=1000):
    past_key_values = None
    system_prompt_sent = False  # system_prompt가 이미 전송되었는지 여부 플래그
    for batch in get_tail_data():
        # � LLM에 전달할 프롬프트를 하나의 JSON으로 생성
        batch_text = "\n".join(batch)  # 30줄을 하나의 문자열로 변환

        if not system_prompt_sent:
            prompt = f"\n[USER]\n{system_prompt}\n{batch_text}\n\n[ASSISTANT]\n"
            #system_prompt_sent = True  # 첫 배치 후에는 더 이상 시스템 프롬프트를 포함하지 않음
        else:
            prompt = f"\n[USER]\n{batch_text}\n\n[ASSISTANT]\n"

        print("\n" + prompt, end="")

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        if kv_cache is not None:
            past_key_values = kv_cache.evict_for_space(past_key_values, input_ids.shape[1] + max_gen_len)

        past_key_values = greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len)

def main(args):
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load(model_name_or_path)

    if args.enable_streaming:
        kv_cache = enable_streaming_llm(model, start_size=args.start_size, recent_size=args.recent_size)
    else:
        kv_cache = None

    streaming_inference(model, tokenizer, kv_cache)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="lmsys/vicuna-13b-v1.3")
    #parser.add_argument("--model_name_or_path", type=str, default="lmsys/vicuna-7b-v1.3")
    #parser.add_argument("--model_name_or_path", type=str, default="Jiayi-Pan/Tiny-Vicuna-1B")
    #parser.add_argument("--model_name_or_path", type=str, default="TheBloke/Vicuna-13B-1-3-SuperHOT-8K-GPTQ")
    #parser.add_argument("--model_name_or_path", type=str, default="togethercomputer/LLaMA-2-7B-32K")
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=2000)
    args = parser.parse_args()

    main(args)
