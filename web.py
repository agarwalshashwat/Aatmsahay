#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# In[2]:


web = Flask(__name__)
rdf_model = pickle.load(open('rdf_model.pkl', 'rb'))
nb_model = pickle.load(open('nb_model.pkl', 'rb'))
knn_model = pickle.load(open('knn_model.pkl', 'rb'))
le = pickle.load(open('le.pkl', 'rb'))
tfvec = pickle.load(open('tfvec.pkl', 'rb'))


# In[3]:


@web.route('/')
def home():
    return render_template('home.html')


# In[4]:


@web.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [str(x) for x in request.form.values()]
    int_features = [' '.join(int_features)]
    rdf = le.inverse_transform(rdf_model.predict(tfvec.transform(int_features).toarray()))
    nb = le.inverse_transform(nb_model.predict(tfvec.transform(int_features).toarray()))
    knn = le.inverse_transform(knn_model.predict(tfvec.transform(int_features).toarray()))

    output = rdf + nb + knn
    output = ' '.join(output)

    return render_template('home.html', prediction_text='The disease you might be having is according to the three models i.e; by naive bayes classifier, random forest classifier adn k-nearest neibhour is {}'.format(output))


# In[5]:


if __name__ == "__main__":
    web.run(debug=True)


# In[ ]:




