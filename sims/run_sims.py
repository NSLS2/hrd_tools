import subprocess
from pathlib import Path

if __name__ == "__main__":
    configs = list(Path("configs").glob("config*toml"))
    subprocess.run(
        ["sbatch", "--array", f"0-{len(configs) - 1}", "batch.job"], check=True
    )
