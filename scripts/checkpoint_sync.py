#!/usr/bin/env python3
"""
LISA Checkpoint Sync - Share model progress between Mac and Jetson
After each training round, sync the latest checkpoint with the other device
"""
import os, sys, time, subprocess, hashlib
from datetime import datetime

# Config - Use environment variables for sensitive info
JETSON_IP = os.environ.get("JETSON_IP", "YOUR_JETSON_IP")  # Set JETSON_IP env var
JETSON_USER = os.environ.get("JETSON_USER", "jetson")  # Set JETSON_USER env var
MAC_CHECKPOINT_DIR = "/tmp/lisa_standalone_checkpoints"
JETSON_CHECKPOINT_DIR = "/tmp/lisa_jetson_checkpoints"
LOG_FILE = "/tmp/lisa_sync.log"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()
    with open(LOG_FILE, "a") as f:
        f.write(f"[{ts}] {msg}\n")

def get_file_hash(path):
    """Quick hash to compare files."""
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return hashlib.md5(f.read(1024*1024)).hexdigest()  # First 1MB

def get_latest_checkpoint(dir_path):
    """Get latest checkpoint in directory."""
    if not os.path.exists(dir_path):
        return None, None
    checkpoints = sorted([f for f in os.listdir(dir_path) if f.startswith("model_round_") and f.endswith(".pt")])
    if checkpoints:
        latest = checkpoints[-1]
        path = os.path.join(dir_path, latest)
        round_num = int(latest.replace("model_round_", "").replace(".pt", ""))
        return path, round_num
    return None, None

def scp(src, dest, timeout=120):
    """Secure copy with timeout."""
    try:
        result = subprocess.run(
            ['scp', '-o', 'ConnectTimeout=10', src, dest],
            capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0
    except Exception as e:
        log(f"SCP error: {e}")
        return False

def sync_to_jetson():
    """Copy Mac's latest checkpoint to Jetson."""
    mac_ckpt, mac_round = get_latest_checkpoint(MAC_CHECKPOINT_DIR)
    if not mac_ckpt:
        return False
    
    jetson_ckpt, jetson_round = get_latest_checkpoint(JETSON_CHECKPOINT_DIR)
    
    # Skip if Jetson is already ahead or same
    if jetson_round and jetson_round >= mac_round:
        log(f"Jetson (r{jetson_round}) >= Mac (r{mac_round}), skipping Mac->Jetson")
        return False
    
    dest = f"{JETSON_USER}@{JETSON_IP}:{JETSON_CHECKPOINT_DIR}/model_round_{mac_round}.pt"
    log(f"SYNC: Mac r{mac_round} -> Jetson...")
    
    if scp(mac_ckpt, dest):
        log(f"✅ Synced Mac r{mac_round} to Jetson")
        return True
    else:
        log(f"❌ Failed to sync to Jetson")
        return False

def sync_from_jetson():
    """Copy Jetson's latest checkpoint to Mac."""
    jetson_ckpt, jetson_round = get_latest_checkpoint(JETSON_CHECKPOINT_DIR)
    if not jetson_ckpt:
        return False
    
    mac_ckpt, mac_round = get_latest_checkpoint(MAC_CHECKPOINT_DIR)
    
    # Skip if Mac is already ahead
    if mac_round and mac_round >= jetson_round:
        log(f"Mac (r{mac_round}) >= Jetson (r{jetson_round}), skipping Jetson->Mac")
        return False
    
    # Copy Jetson's checkpoint to Mac with Jetson's round number
    dest = f"{MAC_CHECKPOINT_DIR}/model_round_{jetson_round}.pt"
    
    # Check if already exists
    if os.path.exists(dest):
        log(f"Jetson r{jetson_round} already exists locally")
        return False
    
    log(f"SYNC: Jetson r{jetson_round} -> Mac...")
    
    # Download with reversed SCP (need to SSH to Jetson first)
    src = f"{JETSON_USER}@{JETSON_IP}:{jetson_ckpt}"
    if scp(src, dest):
        log(f"✅ Synced Jetson r{jetson_round} to Mac")
        return True
    else:
        log(f"❌ Failed to sync from Jetson")
        return False

def sync_checkpoints():
    """Main sync logic - bidirectionally sync checkpoints."""
    log("🔄 Checking for checkpoint sync...")
    
    # Get rounds
    _, mac_round = get_latest_checkpoint(MAC_CHECKPOINT_DIR)
    _, jetson_round = get_latest_checkpoint(JETSON_CHECKPOINT_DIR)
    
    log(f"Mac: r{mac_round}, Jetson: r{jetson_round}")
    
    # Ensure Jetson's checkpoint dir exists
    subprocess.run(['ssh', f'{JETSON_USER}@{JETSON_IP}', 
                   f'mkdir -p {JETSON_CHECKPOINT_DIR}'], capture_output=True)
    
    # Sync both directions
    synced = False
    
    if mac_round and (not jetson_round or mac_round > jetson_round):
        if sync_to_jetson():
            synced = True
    
    if jetson_round and (not mac_round or jetson_round > mac_round):
        if sync_from_jetson():
            synced = True
    
    return synced

def main():
    log("""
╔════════════════════════════════════════════════════════════╗
║  LISA Checkpoint Sync - Mac ↔ Jetson                    ║
╚════════════════════════════════════════════════════════════╝
""")
    
    last_sync = 0
    sync_interval = 120  # Sync every 2 minutes
    
    while True:
        try:
            now = time.time()
            
            # Sync periodically
            if now - last_sync > sync_interval:
                sync_checkpoints()
                last_sync = now
            
            time.sleep(30)
            
        except KeyboardInterrupt:
            log("Stopped")
            break
        except Exception as e:
            log(f"Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
