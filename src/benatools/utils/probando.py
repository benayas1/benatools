from sklearn.datasets import load_boston
import pandas as pd
import numpy as np

data = load_boston()
df = pd.DataFrame(data=np.concatenate([data['data'], data['target'].reshape(-1,1)], axis=1), columns=list(data['feature_names'])+['TARGET'] )
print('Probado correctamente',len(df))