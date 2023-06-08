#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dahuffman import HuffmanCodec


# In[2]:


numb = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
codec = HuffmanCodec.from_data(numb)


# In[3]:


def cod(st_list):
    encoded = codec.encode(st_list)
    return encoded


# In[4]:


def decod(st):
    list_st = codec.decode(st)
    return list_st


# In[ ]:




