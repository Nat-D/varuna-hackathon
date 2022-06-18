import numpy as np

for i in range(200):
    x = np.load(f'data/train/img/{i}.npy')
    print(np.min(x))
    print(np.max(x))
