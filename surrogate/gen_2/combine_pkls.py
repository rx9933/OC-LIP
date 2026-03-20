import pickle
import glob
import os

all_data = []
for f in sorted(glob.glob('oed_training_data_job*.pkl')):
    print(f'Loading {f}')
    with open(f, 'rb') as fp:
        data = pickle.load(fp)
        all_data.extend(data)

print(f'Total samples: {len(all_data)}')
with open('oed_training_data_combined.pkl', 'wb') as f:
    pickle.dump(all_data, f)
print('Saved combined data to oed_training_data_combined.pkl')