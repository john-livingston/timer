import sys
import glob
import pandas as pd

fit_dir = sys.argv[1]
fps = glob.glob(f'{fit_dir}/out-*/ic.txt')
names = [fp.split('/')[-2].split('-')[-1] for fp in fps]
df = pd.concat([pd.read_csv(fp, names=[name], delim_whitespace=True) for fp,name in zip(fps,names)], axis=1).transpose().sort_values(by='BIC')
print(df)
