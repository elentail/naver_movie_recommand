import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('./embedding.pickle', 'rb') as fr:
    data = pickle.load(fr)
    
embedding = data['embedding']
index = data['index']

def onpick(event):
    
    ind = event.ind
    #print('onpick3 scatter:', ind)
    for k in ind:
        print(index[k])
    print("================")
    sys.stdout.flush()
    

# Fixing random state for reproducibility
np.random.seed(19680801)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(embedding[:,0],embedding[:,1],embedding[:,2], picker=True)
fig.canvas.mpl_connect('pick_event', onpick)

plt.show()