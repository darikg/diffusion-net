#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ga_regression import *
from matplotlib import pyplot as plt
from functools import partial


# In[2]:


# folder = r"D:\resynth\run_51_52\1k_faces\2025-07-02-11-02-11"  # okish
folder = r"D:\resynth\run_48_49\1k_faces\2025-07-03-07-07-51"  # ok
# folder = r"D:\resynth\run_09_10\1k_faces\2025-07-03-12-18-20"  # not great
# folder = r"D:\resynth\run_42_43\1k_faces\2025-06-26-14-39-11"  # ok ish
# folder = r"D:\resynth\run_38_39\1k_faces\2025-07-07-07-22-21"  # ok ish


# In[3]:


folder = Path(folder)
r = Reader(folder)
meta = r.metadata()
opts = meta['opts']
epoch, ltest = r.scalar('loss/test')
epoch, ltrain = r.scalar('loss/train')
plt.plot(epoch, ltrain, epoch, ltest)
plt.yscale('log')


# In[4]:


train_test_scenes = (meta['train_scenes'], meta['test_scenes'])
train_dataset, test_dataset = opts.load_datasets(precalc_ops=True, train_test_scenes=train_test_scenes)
expt = opts.experiment(train_dataset=train_dataset, test_dataset=test_dataset)
expt.model.load_state_dict(torch.load(opts.model_file))


# In[5]:


train_loader = DataLoader(expt.train_dataset, batch_size=None, shuffle=False)
# test_loader = DataLoader(expt.test_dataset, batch_size=None)
obs_train, preds_train = expt.predict(train_loader, agg_fn=np.stack)


# In[8]:


expt.model.outputs_at = 'vertices'
ch_idx = 0
i = np.argmax(preds_train[:, ch_idx])
labels, preds, weights = expt.load_item(expt.train_dataset[i])
preds.shape


# In[9]:


from pyvista import PolyData


# In[11]:


verts, faces, *_ = expt.train_dataset[i]
m = PolyData.from_regular_faces(verts.numpy(), faces.numpy())
for i, v in enumerate(preds.cpu().detach().numpy().T):
    m.point_data[f'x{i}'] = v


# In[14]:


from pvutils import iter_subplots


# In[15]:


for (i, j), p in iter_subplots(shape=(5, 3)):
    m.copy().plot(show_edges=True, scalars=f'x{i * j}')

