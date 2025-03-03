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

BATCH_SIZE = 10  # 30줄씩 배치 처리

# TCPDUMP 실시간 패킷 캡처 및 분석
def capture_packets():
    # sudo tcpdump -enni any not port 22 and not arp and not host 127.0.0.1
    process = subprocess.Popen(
        ["sudo", "tcpdump", "-enni", "any", "not", "port", "22", "and", "not", "arp", "and", "not", "host", "127.0.0.1"],
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

system_prompt = f"The following is a part of a recent tcpdump. Analyze the content, and output 'Result=1' only if it is certain that there are anomalies, otherwise ouput 'Result=0'."

@torch.no_grad()
def streaming_inference(model, tokenizer, kv_cache=None, max_gen_len=1000):
    past_key_values = None
    for batch in capture_packets():
        # � LLM에 전달할 프롬프트를 하나의 JSON으로 생성
        batch_text = "\n".join(batch)  # 30줄을 하나의 문자열로 변환
        #prompt = f"[USER] Analyze the following network packet for potential security threats:\n{batch_text}\n\n[ASSISTANT]\n"
        #prompt = f"[USER] Review the following tcpdump contents, and if you find any suspicious part from a security perspective, output 'Result=1', otherwise output 'Result=0' and exit. However, only describe the details that apply if it is Result=1, and absolutely do not respond at all if it is Result=0.\n{batch_text}\n\n[ASSISTANT]\n"
        #prompt = f"[USER] The following is a portion of the results of a recent tcpdump run. Review the flow and content, and if you suspect a malicious hacking attempt, write the details with the Result=1 symbol, and in all other cases, just output Result=0 and exit. (You must not write any additional answers when Result=0 is present.)\n{batch_text}\n\n[ASSISTANT]\n"
        #prompt = f"[USER] The following is a part of a recent tcpdump. Analyze the content and if it is certain that there is a malicious hacking attempt, write the details with the Result=1 symbol, and in all other cases, just output Result=0 and exit. (Do not write any additional answers when Result=0 is present.)\n{batch_text}\n\n[ASSISTANT]\n"
        #prompt = f"[USER] The following is a part of a recent tcpdump. Analyze the content and only describe the details with the Result=1 symbol if it is certain that there is a malicious hacking attempt. In all other cases, just output Result=0 and exit. However, you must not write any additional answers when Result=0 is in the state.\n{batch_text}\n\n[ASSISTANT]\n"
        #prompt = f"USER: Here are the last {BATCH_SIZE} lines of the tcpdump -enni any output. Based on the packet flow, if a serious security risk is expected, describe the content, otherwise, output only \"n/a\".\n{batch_text}\n\nASSISTANT: "
        #prompt = f"USER: Here are the last {BATCH_SIZE} lines of the tcpdump -enni any output. Based on the packet flow, if a serious security risk is expected, describe the content in Korean, otherwise, output only \"n/a\".\n{batch_text}\n\nASSISTANT: "
        #prompt = f"[USER] Here are the last {BATCH_SIZE} lines of the tcpdump -enni any output. Based on the packet flow, if a serious security risk is expected, describe the content, otherwise, output only \"n/a\".\n{batch_text}\n\n[ASSISTANT]\n"
        #prompt = f"[USER] The following is a part of a recent tcpdump. Analyze the content, and output 'Result=1' only if it is certain that there is a malicious hacking attempt, otherwise output 'Result=0'. (Never write any response other than 'Resutl=1' or 'Result=0')\n{batch_text}\n\n[ASSISTANT]\n"
        prompt = f"""[USER]
{system_prompt}

{batch_text}

[ASSISTANT]
"""

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
    #parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=2000)
    args = parser.parse_args()

    main(args)
