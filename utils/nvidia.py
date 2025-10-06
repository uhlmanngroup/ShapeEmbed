import shutil
import subprocess

def select_most_available_nvidia_device():
  if shutil.which("nvidia-smi") is None: return None
  out = subprocess.check_output([ "nvidia-smi"
                                , "--format=csv,noheader,nounits"
                                , "--query-gpu=memory.free" ])
  free_mem = [int(x) for x in out.split()]
  return free_mem.index(max(free_mem))
