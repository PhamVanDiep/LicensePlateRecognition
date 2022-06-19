import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from sklearn.utils import shuffle

# HEIGHT = 28
# WIDTH = 28

# def rotate(image):
#     image = image.reshape([HEIGHT, WIDTH])
#     image = np.fliplr(image)
#     image = np.rot90(image)
#     return image

# def load_dataset():
#     train = pd.read_csv("emnist-balanced-train.csv",delimiter = ',')
#     test = pd.read_csv("emnist-balanced-test.csv", delimiter = ',')
#     # mapp = pd.read_csv("emnist-balanced-mapping.txt", delimiter = ' ', index_col=0, header=None, squeeze=True)

#     # Split x and y
#     train_x = train.iloc[:,1:]
#     train_y = train.iloc[:,0]
#     del train

#     test_x = test.iloc[:,1:]
#     test_y = test.iloc[:,0]
#     del test

#     # Flip and rotate image
#     train_x = np.asarray(train_x)
#     train_x = np.apply_along_axis(rotate, 1, train_x)
#     # print ("train_x:",train_x.shape)

#     test_x = np.asarray(test_x)
#     test_x = np.apply_along_axis(rotate, 1, test_x)
#     # print ("test_x:",test_x.shape)

#     # Normalise
#     train_x = train_x.astype('float32')
#     train_x /= 255
#     test_x = test_x.astype('float32')
#     test_x /= 255

#     num_classes = train_y.nunique()

#     # One hot encoding
#     train_y = np_utils.to_categorical(train_y, num_classes)
#     test_y = np_utils.to_categorical(test_y, num_classes)

#     train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size= 0.10, random_state=7)

#     return ((train_x, train_y), (test_x, test_y), (val_x, val_y), num_classes)

def load_dataset():
    # load numbers dataset 60000 training images, 10000 test images
    (X_train_1, y_train_1), (X_test_1, y_test_1) = mnist.load_data()
    
    # concat numbers traning dataset and test dataset
    X_dataset_1 = np.concatenate((X_train_1,X_test_1), axis = 0)
    y_dataset_1 = np.concatenate((y_train_1,y_test_1), axis = 0)
    
    # relabel for numbers dataset after concat
    # A-Z : 0 : 25 , 0-9 : 26-35 
    for i in range(len(y_dataset_1)):
        y_dataset_1[i] = y_dataset_1[i] + 26 # alphabet -> number => +26 
    
    # reshape dataset to array
    X_dataset_1 = X_dataset_1.reshape((70000, 784))
    y_dataset_1 = y_dataset_1.reshape((70000, 1))
    mnist_dataset = np.concatenate((y_dataset_1, X_dataset_1), axis = 1)

    # load alphabet dataset
    dataset = pd.read_csv("A_Z Handwritten Data.csv").astype('float32')
    dataset.rename(columns={'0':'label'}, inplace=True)
    kaggle_dataset = dataset.to_numpy()

    # print(kaggle_dataset.shape)
    # print(mnist_dataset.shape)

    # create dataset by concat alphabet dataset and numbers dataset
    dataset = np.concatenate((kaggle_dataset, mnist_dataset), axis = 0)
    # print(dataset.shape)
    dataset_pd = pd.DataFrame(dataset, columns = [x for x in range(785)])
    dataset_pd = shuffle(dataset_pd)
    dataset_pd.rename(columns={0:'label'}, inplace=True)

    X = dataset_pd.drop('label',axis = 1)
    y = dataset_pd['label']

    data = X.to_numpy()
    label = y.to_numpy()

    # plt.imshow(data[9990].reshape(28,28))
    # print(label[9990])

    # vẽ biểu đồ xem số lượng mẫu mỗi loại
    # alphabets_mapper = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V'
    # ,22:'W',23:'X',24:'Y',25:'Z',26:'0',27:'1',28:'2', 29:'3', 30:'4',31:'5',32:'6', 33:'7', 34:'8', 35:'9'} 
    # dataset_alphabets = dataset_pd.copy()
    # dataset_pd['label'] = dataset_pd['label'].map(alphabets_mapper)
    # label_size = dataset_pd.groupby('label').size()
    # label_size.plot.barh(figsize=(10,10))

    # split dataset to 3 datasets as train, test, validate by rating 7:1.5:1.5 
    SIZE = data.shape[0]
    X_train,y_train = data[0:int(0.7*SIZE)], label[0:int(0.7*SIZE)]
    X_test, y_test = data[int(0.7*SIZE):int(0.85*SIZE)], label[int(0.7*SIZE):int(0.85*SIZE)]
    X_val, y_val = data[int(0.85*SIZE):SIZE], label[int(0.85*SIZE): SIZE]

    # print(X_train.shape)
    # print(X_test.shape)
    # print(X_val.shape)
    return ((X_train, y_train),(X_test, y_test),(X_val,y_val))