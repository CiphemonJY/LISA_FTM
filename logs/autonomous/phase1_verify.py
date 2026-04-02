
import os, gc, psutil, struct

print("=" * 60)
print("GGUF FILE VERIFICATION")
print("=" * 60)

process = psutil.Process()

# Valid GGUF files we found
files = {
    '14B Q8': '/tmp/qwen14b-q8.gguf',
    '32B Q8': '/tmp/qwen32b-q8.gguf',
    '70B Q4': '/tmp/Llama-3.3-70B-Instruct-Q4_K_M.gguf',
    '3B Q4': '/tmp/qwen32b_q4_parts/qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf',
}

print(f"\nInitial RAM: {process.memory_info().rss/1e9:.2f}GB")
print("\nTesting each file:\n")

results = {}

for name, path in files.items():
    if not os.path.exists(path):
        print(f"{name}: NOT FOUND")
        results[name] = 'not_found'
        continue
    
    size_gb = os.path.getsize(path) / 1e9
    print(f"{name} ({size_gb:.1f}GB):")
    
    # Check if it's a valid GGUF
    try:
        with open(path, 'rb') as f:
            magic = f.read(4)
            if magic == b'GGUF':
                version = struct.unpack('<I', f.read(4))[0]
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                print(f"  Valid GGUF v{version}, {tensor_count} tensors")
                
                # Try loading with llama.cpp
                from llama_cpp import Llama
                print(f"  Loading with llama.cpp...")
                
                llm = Llama(model_path=path, n_ctx=16, n_gpu_layers=0, 
                           n_batch=4, use_mmap=True, n_threads=4)
                
                print(f"  RAM after load: {process.memory_info().rss/1e9:.2f}GB")
                
                # Try generation
                result = llm("def", max_tokens=5, echo=False)
                text = result['choices'][0]['text'].strip()
                print(f"  Output: '{text}'")
                
                results[name] = 'success'
                del llm
            else:
                print(f"  Not GGUF: {magic}")
                results[name] = 'invalid'
    except Exception as e:
        print(f"  ERROR: {e}")
        results[name] = f'error: {e}'
    
    gc.collect()
    print(f"  RAM after cleanup: {process.memory_info().rss/1e9:.2f}GB\n")

print("=" * 60)
print("SUMMARY")
print("=" * 60)
for name, status in results.items():
    print(f"{name}: {status}")
