#!/usr/bin/env python
# coding: utf-8

# In[14]:



#K-means clustering is a type of unsupervised learning, which is used when you have unlabeled data.
#The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K.


# In[1]:


import numpy as np 
import matplotlib.pyplot as plt
import numpy as np 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 


# In[7]:



from sklearn.datasets import make_blobs


# In[10]:


x,y_true= make_blobs(n_samples=300 , centers=4 , cluster_std= 0.6 , random_state=0)
plt.scatter(x[:,0],x[:,1],s=50)


# In[12]:


from sklearn.cluster import KMeans
Kmeans=KMeans()
Kmeans.fit(x)
y_Kmeans=Kmeans.predict(x)


# In[13]:


y_Kmeans


# In[ ]:




