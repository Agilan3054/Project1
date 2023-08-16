#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dot, Dense
from keras.optimizers import Adam


# In[1]:


# Sample movie ratings data
data = {
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4],
    'movie_id': [101, 102, 101, 103, 102, 103, 104, 105],
    'rating': [5, 4, 3, 4, 2, 3, 5, 4]
}


# In[ ]:


df = pd.DataFrame(data)


# In[ ]:


# Map user and movie IDs to unique indices
user_ids = df['user_id'].unique()
movie_ids = df['movie_id'].unique()


# In[ ]:


user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
movie2idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}


# In[ ]:


df['user_idx'] = df['user_id'].map(user2idx)
df['movie_idx'] = df['movie_id'].map(movie2idx)


# In[ ]:


# Split data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


# In[ ]:


# Build the model using Keras
num_users = len(user_ids)
num_movies = len(movie_ids)
embedding_dim = 50


# In[ ]:


user_input = Input(shape=(1,))
movie_input = Input(shape=(1,))


# In[ ]:


user_embedding = Embedding(num_users, embedding_dim)(user_input)
movie_embedding = Embedding(num_movies, embedding_dim)(movie_input)


# In[ ]:


user_vec = Flatten()(user_embedding)
movie_vec = Flatten()(movie_embedding)


# In[ ]:


dot_product = Dot(axes=1)([user_vec, movie_vec])


# In[ ]:


model = Model(inputs=[user_input, movie_input], outputs=dot_product)
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))


# In[ ]:


# Train the model
model.fit([train_df['user_idx'], train_df['movie_idx']], train_df['rating'], epochs=10, verbose=1)


# In[ ]:


# Evaluate the model
loss = model.evaluate([test_df['user_idx'], test_df['movie_idx']], test_df['rating'])
print(f"Test loss: {loss:.4f}")


# In[ ]:


# Make recommendations
def recommend_movies(user_idx, num_recommendations=5):
    unrated_movies = np.setdiff1d(np.arange(num_movies), train_df[train_df['user_idx'] == user_idx]['movie_idx'])
    user_indices = np.full_like(unrated_movies, user_idx)
    predictions = model.predict([user_indices, unrated_movies])
    top_indices = predictions.argsort()[::-1][:num_recommendations]
    recommended_movies = [movie_ids[movie_idx] for movie_idx in unrated_movies[top_indices]]
    return recommended_movies


# In[ ]:


# Get recommendations for a specific user
user_idx = 0  # Change this to the desired user index
recommendations = recommend_movies(user_idx)


# In[ ]:


print(f"Recommended movies for User {user_idx}:")
for movie in recommendations:
    print(f"Movie {movie}")

