import os 

def get_cache_dfaust(sid, seq, i):
    return os.path.join(sid, seq, f"{sid}_{seq}_{i:05d}")