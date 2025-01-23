# %% [markdown]
# proto-typing for building "database" of results

# %%
import pandas as pd
from dataclasses import asdict
from hello_world import reduce
from pathlib import Path

# %%

config = reduce.load_all_config(Path('/nsls2/data3/projects/next_iiia_hrd/xrt_output/'))

# %%
df = pd.DataFrame({k: {f'{outer}.{inner}': v for outer, cfg in asdict(v).items() for inner, v in cfg.items()} for k,v in config.items()}).T
df['job'] = [_.name[:-4] for _ in df.index]


# %%
for n, g in df.groupby('job'):
    print(n, len(g.index))
# %%

results = {k:reduce.reduce_file(k, mode='cython') for k in config}
# %%
