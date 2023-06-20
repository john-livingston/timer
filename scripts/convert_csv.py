import os
import sys
import pandas as pd

fp = sys.argv[1]
df = pd.read_csv(fp)
dn = os.path.dirname(fp)
fn = os.path.basename(fp)
if fn.endswith('.gz'):
    fn = os.path.splitext(fn)[0]
fp_out = os.path.join(dn, fn.replace('.csv', '.txt'))
df.to_csv(fp_out, index=False, header=False, sep=' ')
print('wrote file:', fp_out)
