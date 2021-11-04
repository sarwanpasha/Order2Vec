import pandas as pd
import numpy as np
import time
from sklearn.svm import SVC
#import RandomBinningFeatures
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import RidgeClassifier
from sklearn import metrics
import scipy
import matplotlib.pyplot as plt 
#%matplotlib inline 
import csv

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import TruncatedSVD
import random
# import seaborn as sns
import os.path as path
import os
# import matplotlib
# import matplotlib.font_manager
# import matplotlib.pyplot as plt # graphs plotting
# import Bio
from Bio import SeqIO # some BioPython that will come in handy
#matplotlib inline

from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score


# from matplotlib import rc
# # for Arial typefont
# matplotlib.rcParams['font.family'] = 'Arial'

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import svm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, confusion_matrix 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from pandas import DataFrame

from sklearn.model_selection import KFold 
from sklearn.model_selection import RepeatedKFold

from sklearn.metrics import confusion_matrix

from numpy import mean
#import seaborn as sns

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# LSTM for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.sequence import pad_sequences

import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline



print("Packages Loaded!!!")


# # Read Frequency Vector
frequency_vector_read_final = np.load("/alina-data1/sarwan/PAKDD/Dataset/frequency_vectors_for_minimizer_3_mers.npy")

print("Frequency Vector Data Reading Done with length ==>>",len(frequency_vector_read_final))



read_path = "/alina-data1/sarwan/IEEE_BigData/Dataset/Complete Clustering Data/complete_other_attributes_only.csv"

variant_orig = []

with open(read_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        tmp = row
        variant_orig.append(tmp[1])
        
print("Attributed data Reading Done")

unique_varaints = list(np.unique(variant_orig))




int_variants = []
for ind_unique in range(len(variant_orig)):
    variant_tmp = variant_orig[ind_unique]
    ind_tmp = unique_varaints.index(variant_tmp)
    int_variants.append(ind_tmp)
    
print("Attribute data preprocessing Done")


freq_vec_reduced = []
int_variant_reduced = []
name_variant_reduced = []

for ind_reduced in range(len(frequency_vector_read_final)):
    if variant_orig[ind_reduced]=="B.1.1.7" or variant_orig[ind_reduced]=="B.1.617.2" or variant_orig[ind_reduced]=="AY.4" or variant_orig[ind_reduced]=="B.1.2" or variant_orig[ind_reduced]=="B.1" or variant_orig[ind_reduced]=="B.1.177"  or variant_orig[ind_reduced]=="P.1" or variant_orig[ind_reduced]=="B.1.1" or variant_orig[ind_reduced]=="B.1.429"  or variant_orig[ind_reduced]=="AY.12" or variant_orig[ind_reduced]=="B.1.160" or variant_orig[ind_reduced]=="B.1.526" or variant_orig[ind_reduced]=="B.1.1.519" or variant_orig[ind_reduced]=="B.1.351" or variant_orig[ind_reduced]=="B.1.1.214"  or variant_orig[ind_reduced]=="B.1.427" or variant_orig[ind_reduced]=="B.1.221" or variant_orig[ind_reduced]=="B.1.258" or variant_orig[ind_reduced]=="B.1.177.21" or variant_orig[ind_reduced]=="D.2" or variant_orig[ind_reduced]=="B.1.243"  or variant_orig[ind_reduced]=="R.1":
        freq_vec_reduced.append(frequency_vector_read_final[ind_reduced])
        int_variant_reduced.append(int_variants[ind_reduced])
        name_variant_reduced.append(variant_orig[ind_reduced])

print("Total Sequences after reducing data ==>>",len(freq_vec_reduced))


X = np.array(freq_vec_reduced)
y =  np.array(int_variant_reduced)
y_orig = np.array(name_variant_reduced)


from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit
sss = ShuffleSplit(n_splits=1, test_size=0.9)
sss.get_n_splits(X, y)
train_index, test_index = next(sss.split(X, y)) 

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]

y_train_orig, y_test_orig = y_orig[train_index], y_orig[test_index]

print("Train-Test Split Done")

print("X_train rows = ",len(X_train),"X_train columns = ",len(X_train[0]))
print("X_test rows = ",len(X_test),"X_test columns = ",len(X_test[0]))

# Classification Functions !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print("Random Fourier Features Starts here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

start_time = time.time()
rbf_feature = RBFSampler(gamma=1, n_components=500)
rbf_feature.fit(X_train)
X_features_train = rbf_feature.transform(X_train)
X_features_test_RFT = rbf_feature.transform(X)      


######################################################################################

start_time = time.time()

#for clustering, the input data is in variable X_features_test
from sklearn.cluster import KMeans


number_of_clusters = [22] #number of clusters

for clust_ind in range(len(number_of_clusters)):
    print("Number of Clusters = ",number_of_clusters[clust_ind])
    clust_num = number_of_clusters[clust_ind]
    
    kmeans = KMeans(n_clusters=clust_num, random_state=0).fit(X_features_test_RFT)
    kmean_clust_labels = kmeans.labels_
    
    np.save('/alina-data1/Zara/RCOMB/Results/K-means/Kmeans_Minimizer_RFT/Labels/New_Labels_kmeans_RFT.npy', kmean_clust_labels)


    end_time = time.time() - start_time
    print("Clustering Time in seconds =>",end_time)



    np.save('/alina-data1/Zara/RCOMB/Results/K-means/Kmeans_Minimizer_RFT/Labels/New_test_int_true_variants_22.npy', y)
    np.save('/alina-data1/Zara/RCOMB/Results/K-means/Kmeans_Minimizer_RFT/Labels/New_orig_true_variants_22.npy', y_orig)
    np.save('/alina-data1/Zara/RCOMB/Results/K-means/Kmeans_Minimizer_RFT/Labels/New_test_orig_true_variants_22.npy', y_test_orig)


print("All Processing Done!!!")

