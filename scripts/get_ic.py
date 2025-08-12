import sys
import glob
import pandas as pd

fit_dir = sys.argv[1]
fps = glob.glob(f'{fit_dir}/out-*/ic.txt')
names = [fp.split('/')[-2].split('-')[-1] for fp in fps]
dfs = []
for fp, name in zip(fps, names):
    df_ = pd.read_csv(fp, delim_whitespace=True, names=['key', 'type', 'val'])
    df_ = df_.set_index('key')['val']
    dfs.append(df_.rename(name))
df = pd.concat(dfs, axis=1).transpose().sort_values(by='BIC')
print(df)
