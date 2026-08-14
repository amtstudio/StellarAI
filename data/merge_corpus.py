"""合并原始语料和HF数据集"""
import os

base = os.path.dirname(__file__)

# 读取现有语料
with open(os.path.join(base, 'train_corpus.txt'), 'r', encoding='utf-8') as f:
    old_lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

# 读取HF新语料
with open(os.path.join(base, 'train_corpus_hf.txt'), 'r', encoding='utf-8') as f:
    new_lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

# 合并去重
seen = set()
merged = []
for line in old_lines + new_lines:
    key = line[:80]
    if key not in seen and 15 <= len(line) <= 500:
        seen.add(key)
        merged.append(line)

# 保存
out = os.path.join(base, 'train_combined.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('# StellarAI 合并训练语料 (原始 + HF数据集)\n')
    f.write(f'# 原始: {len(old_lines)} 行, HF: {len(new_lines)} 行\n\n')
    for line in merged:
        f.write(line + '\n')

print(f'原始语料: {len(old_lines)} 行')
print(f'HF新语料: {len(new_lines)} 行')
print(f'合并去重: {len(merged)} 行')
print(f'大小: {os.path.getsize(out)/1024:.1f} KB')