import pandas as pd
import hashlib
import base64
import json

def derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the password using SHA256."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64-encoded ciphertext with XOR."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


def xor_decrypt(data, key):
    """
    XOR decrypt data with a key
    """
    key_bytes = key.encode('utf-8')
    key_length = len(key_bytes)
    return bytes([data[i] ^ key_bytes[i % key_length] for i in range(len(data))])

if __name__ == '__main__':
    ## browsecomp_en
    df = pd.read_csv('./eval_data/browsecomp_en_raw.csv')
    for row in df.itertuples():
        problem = decrypt(row.problem, row.canary)
        answer = decrypt(row.answer, row.canary)
        df.at[row.Index, 'answer'] = answer
        df.at[row.Index, 'question'] = problem
        
    df = df[['question', 'answer']]
    
    records = df.to_dict(orient='records')
    with open('./eval_data/browsecomp_en.jsonl', 'w', encoding='utf-8') as f:
        for record in records:
            json_line = json.dumps(record, ensure_ascii=False)
            f.write(json_line + '\n')
    
    ## browsecomp_zh
    
    df = pd.read_csv('./eval_data/browsecomp_zh_raw.csv')
    for row in df.itertuples():
        question = decrypt(row.Question, row.canary)
        answer = decrypt(row.Answer, row.canary)
        df.at[row.Index, 'answer'] = answer
        df.at[row.Index, 'question'] = question
    
    df = df[['question', 'answer']]
    
    records = df.to_dict(orient='records')
    with open('./eval_data/browsecomp_zh.jsonl', 'w', encoding='utf-8') as f:
        for record in records:
            json_line = json.dumps(record, ensure_ascii=False)
            f.write(json_line + '\n')
    
    ## gaia
    
    from pandas import read_parquet
    df = read_parquet('./eval_data/gaia_raw.parquet')
    df = df.rename(columns={'Question': 'question', 'Final answer': 'answer'})
        
    df = df[['question', 'answer']]
    
    with open('./eval_data/gaia.jsonl', 'w', encoding='utf-8') as f:
        for record in records:
            json_line = json.dumps(record, ensure_ascii=False)
            f.write(json_line + '\n')
    
    ## DeepSearch
    
    df = pd.read_csv('./eval_data/xbench-deepsearch_raw.csv')
    for row in df.itertuples():
        prompt = xor_decrypt(base64.b64decode(row.prompt), row.canary).decode('utf-8')
        answer = xor_decrypt(base64.b64decode(row.answer), row.canary).decode('utf-8')
        df.at[row.Index, 'answer'] = answer
        df.at[row.Index, 'question'] = prompt
        
    df = df[['question', 'answer']]
    
    with open('./eval_data/xbench-deepsearch.jsonl', 'w', encoding='utf-8') as f:
        for record in records:
            json_line = json.dumps(record, ensure_ascii=False)
            f.write(json_line + '\n')