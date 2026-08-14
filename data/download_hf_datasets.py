"""
从 HF 已缓存数据 + 在线下载处理训练语料
"""
import os, re, json, sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "train_corpus_hf.txt")
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".hf_cache")
MAX_CHARS = 500
MAX_LINES = 20000


def clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-z]+;', ' ', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303fa-zA-Z0-9\s.,!?;:()\[\]{}\'\"\-_+=@#$%^&*|/~`，。！？；：""''（）【】《》、……—·￥\n]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_lines(text: str, max_chars: int = MAX_CHARS) -> list:
    lines = []
    for para in text.split('\n'):
        para = para.strip()
        if not para or len(para) < 15:
            continue
        sentences = re.split(r'([。！？.!?\n])', para)
        current = ""
        for piece in sentences:
            if not piece.strip():
                continue
            if len(current) + len(piece) <= max_chars:
                current += piece
            else:
                if len(current.strip()) >= 15:
                    lines.append(current.strip())
                current = piece
        if len(current.strip()) >= 15:
            lines.append(current.strip())
    return lines


def process_alpaca(filepath: str, max_items: int = 15000) -> list:
    """处理 alpaca 格式数据"""
    lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    count = 0
    for item in data:
        inst = clean_text(item.get("instruction", ""))
        inp = clean_text(item.get("input", ""))
        out = clean_text(item.get("output", ""))
        if not inst or not out:
            continue
        if inp:
            text = f"问：{inst}\n{inp}\n答：{out}"
        else:
            text = f"问：{inst}\n答：{out}"
        lines.extend(split_lines(text))
        count += 1
        if count >= max_items:
            break
    print(f"    提取 {count} 条 -> {len(lines)} 行")
    return lines


def process_generic_json(filepath: str, max_items: int = 5000) -> list:
    """处理通用 JSON 数据"""
    lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except:
            return lines
    
    if isinstance(data, list):
        for item in data[:max_items]:
            # alpaca format
            if "instruction" in item and "output" in item:
                inst = clean_text(item.get("instruction", ""))
                inp = clean_text(item.get("input", ""))
                out = clean_text(item.get("output", ""))
                if inst and out:
                    text = f"问：{inst}\n答：{out}" if not inp else f"问：{inst}\n{inp}\n答：{out}"
                    lines.extend(split_lines(text))
            else:
                for key in ["text", "content", "answer", "output", "sentence"]:
                    if key in item:
                        t = clean_text(str(item[key]))
                        if t and len(t) >= 15:
                            lines.extend(split_lines(t))
                            break
    elif isinstance(data, dict):
        for key, item in list(data.items())[:max_items]:
            if isinstance(item, str):
                t = clean_text(item)
                if t and len(t) >= 15:
                    lines.extend(split_lines(t))
    
    return lines


def find_cached_files():
    """查找所有已缓存的JSON文件"""
    result = []
    if not os.path.exists(CACHE_DIR):
        return result
    for root, dirs, files in os.walk(CACHE_DIR):
        for f in files:
            if f.endswith(('.json', '.jsonl')):
                result.append(os.path.join(root, f))
    return result


def try_download(repo_id: str, filename: str) -> str | None:
    """尝试下载单个文件"""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo_id, filename=filename,
                               repo_type="dataset", cache_dir=CACHE_DIR)
        return path
    except Exception as e:
        return None


def main():
    print("=" * 60)
    print("StellarAI HF 数据集处理器")
    print("=" * 60)

    os.makedirs(CACHE_DIR, exist_ok=True)
    all_lines = []

    # ====== 1. 处理已缓存的 alpaca 数据 (48K条) ======
    print("\n[1/4] 中文 Alpaca 指令数据 (缓存)...")
    cached = find_cached_files()
    alpaca_files = [f for f in cached if 'alpaca' in f.lower() and f.endswith('.json')]
    if alpaca_files:
        for fpath in alpaca_files:
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            print(f"    缓存: {os.path.basename(fpath)} ({size_mb:.1f} MB)")
            lines = process_alpaca(fpath, max_items=15000)
            all_lines.extend(lines)
    else:
        print("    未找到缓存，尝试下载...")
        path = try_download("shibing624/alpaca-zh", "alpaca_gpt4_data_zh.json")
        if path:
            print(f"    已下载: {os.path.getsize(path)/1024/1024:.1f} MB")
            lines = process_alpaca(path)
            all_lines.extend(lines)

    # ====== 2. 尝试下载其他中文数据集 ======
    print("\n[2/4] 尝试下载其他数据集...")
    
    # 医疗问答
    extra_repos = [
        ("shibing624/medical", ["medical_qa.json", "train.json"]),
        ("shibing624/AdvertiseGen", ["advertise_gen.json", "train.json"]),
    ]
    
    for repo_id, filenames in extra_repos:
        for fname in filenames:
            path = try_download(repo_id, fname)
            if path:
                size_mb = os.path.getsize(path) / 1024 / 1024
                print(f"    已下载: {repo_id}/{fname} ({size_mb:.1f} MB)")
                lines = process_generic_json(path)
                all_lines.extend(lines)
                break
            else:
                print(f"    跳过: {repo_id}/{fname}")

    # ====== 3. 尝试 sharegpt (可能失败) ======
    print("\n[3/4] 尝试中文对话数据...")
    sharegpt_files = ["sharegpt_zh_38K_format.jsonl", "sharegpt_gpt4.jsonl", "sharegpt_V3_format.jsonl"]
    for fname in sharegpt_files:
        path = try_download("shibing624/sharegpt_gpt4", fname)
        if path:
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"    已下载: {fname} ({size_mb:.1f} MB)")
            # 处理 jsonl
            lines = []
            with open(path, 'r', encoding='utf-8') as f:
                count = 0
                for raw in f:
                    try:
                        item = json.loads(raw)
                        conv = item.get("conversation", item.get("conversations", []))
                        for turn in conv:
                            if isinstance(turn, dict):
                                content = turn.get("content", turn.get("value", ""))
                                if content:
                                    lines.extend(split_lines(clean_text(content)))
                    except:
                        pass
                    count += 1
                    if count >= 5000:
                        break
            print(f"    提取 {count} 条 -> {len(lines)} 行")
            all_lines.extend(lines)
            break
        else:
            print(f"    跳过: {fname}")

    # ====== 4. 处理其他缓存文件 ======
    print("\n[4/4] 处理其他缓存文件...")
    for fpath in cached:
        if 'alpaca' in fpath.lower() or 'sharegpt' in fpath.lower():
            continue
        if os.path.getsize(fpath) < 50 * 1024 * 1024:
            try:
                lines = process_generic_json(fpath, max_items=3000)
                if lines:
                    print(f"    缓存: {os.path.basename(fpath)} -> {len(lines)} 行")
                    all_lines.extend(lines)
            except:
                pass

    # 去重
    print(f"\n--- 去重处理 ---")
    seen = set()
    unique = []
    for line in all_lines:
        key = line[:80]
        if key not in seen and 15 <= len(line) <= MAX_CHARS:
            seen.add(key)
            unique.append(line)

    if len(unique) > MAX_LINES:
        unique = unique[:MAX_LINES]

    # 保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"# StellarAI HF 数据集训练语料\n")
        f.write(f"# 来源: Hugging Face Hub\n\n")
        for line in unique:
            f.write(line + '\n')

    print(f"\n{'=' * 60}")
    print(f"处理完成!")
    print(f"  原始行数: {len(all_lines)}")
    print(f"  去重后:   {len(unique)}")
    print(f"  输出:     {OUTPUT_FILE}")
    if unique:
        print(f"  大小:     {os.path.getsize(OUTPUT_FILE)/1024:.1f} KB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()