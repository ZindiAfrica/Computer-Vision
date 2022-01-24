import numpy as np
from utils import read_train_data, read_test_data

#read training data
train_imgs, train_gts = read_train_data('train_data')

#remove dublicate training imgs
idx_to_rmv = []
for i in range(len(train_imgs)-1):
    for j in range(i+1, len(train_imgs)):
        if np.all(train_imgs[i] == train_imgs[j]):
            idx_to_rmv.append(i)
            if train_gts[i] != train_gts[j]:
                idx_to_rmv.append(j)

idx = [i for i in range(len(train_imgs)) if not(i in idx_to_rmv)]
print('unique train imgs:',len(idx))

#save unique training imgs
np.save('unique_train_imgs_rot_fixed', np.array(train_imgs)[idx])
np.save('unique_train_gts_rot_fixed', np.array(train_gts)[idx])

#read test data
test_imgs, test_gts, ids = read_test_data('test_data')

#save test data
np.save('test_imgs_rot_fixed', np.array(test_imgs))
np.save('test_gts', np.array(test_gts))
np.save('ids', np.array(ids))

