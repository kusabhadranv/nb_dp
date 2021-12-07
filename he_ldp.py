from __future__ import division
import pandas as pd
import numpy as np
import os
from math import ceil



class heldp:

    def __init__(self, dataFrame,enc): 

        data_X = data.iloc[:, :-1] 
        data_y = data.iloc[:, -1]  
        self.pre_process(data_X, data_y)
        self.he_encode(enc)


    def pre_process(self, data_X, data_y):
        self.ft_cnt = [0] * data_X.shape[1]
        self.num_X = np.zeros(data_X.shape)
        self.ft_cat = {}
        for i in range(len(data_X.columns)):
            col = data_X.iloc[:, i].astype('category')                             #Convert each column to category
            self.num_X[:, i] = np.array(col.cat.codes)                             #Convert category to number
            self.ft_cat[i] = col.cat.categories.tolist()                           #Save all possible catergories of a column  
            self.ft_cnt[i]=len(self.ft_cat[i])                                     ## of catergories in each column 

            
        data_y.is_copy = False                                                     #CHECK THIS!!!!
        y = data_y.astype('category')                                              #Convert label to category
        self.num_y = np.array(y.cat.codes)                                         #Convert label to number  
        self.cnt_y = y.cat.categories.tolist()                                     #All possible num_y

        
        self.enc_X = np.zeros((self.num_X.shape[0], sum(self.ft_cnt)))              #One hot encoding for X
        f_col = 0
        for i, fVal in enumerate(self.ft_cnt):
            self.enc_X[np.arange(self.num_X.shape[0]), f_col + self.num_X[:, i].astype(int)] = 1
            f_col += fVal

        
        self.enc_y = np.zeros((len(self.num_y), len(self.cnt_y)))                   #One hot encoding for y
        self.enc_y[np.arange(self.num_y.shape[0]), self.num_y.astype(int)] = 1
    
    
    def he_encode(self,enc):
        

        self.encoding = enc
        X = self.enc_X
        y = self.enc_y

        
        self.col_X = X.shape[1]                                                     #Columns for X
        self.col_y = y.shape[1]                                                     #Columns for y
        
        #Full Matrix [#samples , (#features * #classes) + 1(y)]
        self.full_mat = np.zeros((X.shape[0], (self.col_X * len(self.cnt_y) ) + self.col_y))  
        
        col_ind = 0
        for count in range(len(self.cnt_y)):
            for index, att_count in enumerate(self.ft_cnt):
                vec = np.random.randint(att_count, size=X.shape[0])                 #Adding random values
                self.full_mat[np.arange(X.shape[0]), col_ind + vec.astype(int)] = 1
                col_ind += att_count  
                
              
        
        for i in range(X.shape[0]):                                                 #Fill sample values in full matrix
            col_start = self.num_y[i] * self.col_X
            self.full_mat[i, col_start:(col_start+self.col_X)] = X[i, :]
        
        clInd = len(self.cnt_y) * self.col_X

        self.full_mat[:, clInd:(clInd+self.col_y)] = y                              #Fill y values in full matrix

    

    # Split into training and testing data
    def split(self, train_size, random_state):

        rows_X = int(train_size * self.num_X.shape[0])

        np.random.seed(random_state)                                                #Get random indices for splitting data
        indices = np.random.permutation(self.num_X.shape[0])
        train_ind, test_ind = indices[:rows_X], indices[rows_X:]                    #Split indices to train and test
        
        
        self.test_X = self.enc_X[test_ind, :]                                       #Get test_X and test_y                                  
        self.test_y = self.num_y[test_ind]

        tmpMat = self.full_mat[train_ind, :]
        reqMult = int(ceil(rows_X / tmpMat.shape[0]))

        self.train_X = np.repeat(tmpMat, reqMult, axis=0)
        self.train_X = self.train_X[:rows_X, :]

    
    def flip_coin(self, p, M, N, tol=0.1):                                          #Flip coin based on 
        condition = True
        while condition:                                                            #Loop until acceptable tolerance
            rndMat = np.random.random((M, N)) if N > 1 else np.random.random(M)
            rndMat = (rndMat < p).astype(int)
            prop = np.sum(rndMat) / (M * N)
            condition = np.abs(p - prop) > tol
        return rndMat

    
    def probability(self, d=1, th=1):

        self.p = 1 - (0.5 * np.exp( (self.eps * (th - 1) ) / 2 ))
        self.q = 0.5 * np.exp( (-0.5 * self.eps * th) / 2 )
    
    def add_noise(self, eps, threshold=0):
        
        np.random.seed()
        self.pert_mat = np.zeros(self.train_X.shape)        
        self.eps = eps
    
        L = np.random.laplace(scale=(2.0/eps), size=self.train_X.shape)              #Laplacian noise with scale (2/epsilon)
        self.pert_mat = self.train_X + L                                             #Adding laplacian noise

        if self.encoding == "THE":
            self.pert_mat = (self.pert_mat >= threshold).astype(int)
            self.probability(th=threshold)
        

    def calc_sum(self, value, n, p, q):                                             #Calculate the sum
        result = (value - (n * q)) / (p - q)
        if type(result) is np.ndarray:
            result[result <= 0] = 1 
        elif result <= 0:
            result = 1 
        return result

    def aggregate(self):
        att_sum = np.zeros((len(self.cnt_y), sum(self.ft_cnt)))                     #Attribute sum
        cls_sum = np.zeros(len(self.cnt_y))                                         #Class sum
        

        index = len(self.cnt_y) * self.col_X
        cls_sum = np.sum(self.pert_mat[:, index:], axis=0)
        for col_ind in range(len(self.cnt_y)):
            start_ind = col_ind * self.col_X
            att_sum[col_ind, :] = np.sum(self.pert_mat[:,start_ind:(start_ind+self.col_X)], axis=0)


        self.est_att_sum = np.zeros(att_sum.shape)                                  #Attribute sum estimate
        self.est_cls_sum = np.zeros(cls_sum.shape)                                  #Class sum estimate
  
        if self.encoding == "SHE":
            self.est_att_sum = att_sum
            self.est_cls_sum  = cls_sum


        if self.encoding == "THE":
            self.est_att_sum = self.calc_sum(att_sum, self.pert_mat.shape[0], self.p, self.q)
            self.est_cls_sum  = self.calc_sum(cls_sum, self.pert_mat.shape[0], self.p, self.q)

        self.est_att_sum = self.remove_random(self.est_cls_sum, self.est_att_sum)

    
    def remove_random(self, class_sum, att_sum):                                    #Subtract random values from attribute sum
        for clInd in range(len(self.cnt_y)):
            fColIdx = 0
            rndCount = self.train_X.shape[0] - class_sum[clInd]
            for att_count in self.ft_cnt:
                att_sum[clInd, fColIdx:(fColIdx+att_count)] -= round(rndCount / att_count)
                fColIdx += att_count
        if type(att_sum) is np.ndarray:
            att_sum[att_sum <= 0] = 1 
        elif att_sum <= 0:
            att_sum = 1 
        return att_sum
    
    def train(self):                                                                                                                                    
        self.att_prob = np.zeros(self.est_att_sum.shape)                            
        self.class_prob  = np.zeros(self.est_cls_sum.shape)
    
        self.class_prob = self.est_cls_sum / np.sum(self.est_cls_sum)               #Calculate class probability                
        
        for clInd in range(len(self.cnt_y)):                                        #Calculate attribute probability
            att_prob_ind = 0
            for att_count in self.ft_cnt:
                self.att_prob[clInd, att_prob_ind:(att_prob_ind+att_count)] = self.est_att_sum[clInd, att_prob_ind:(att_prob_ind+att_count)] / np.sum(self.est_att_sum[clInd, att_prob_ind:(att_prob_ind+att_count)])
                att_prob_ind += att_count
    

    def classify(self):
        test_result = [None] * self.test_X.shape[0] 
        for i in range(self.test_X.shape[0]):
            test_prob = np.zeros(len(self.class_prob))
            for ind in range(len(self.class_prob)):
                test_prob[ind] = self.class_prob[ind] * np.prod(self.att_prob[ind, self.test_X[i] == 1])
            test_result[i] = np.argmax(test_prob)                                    #Classify the sample
        result_y = np.array(test_result)
        return 100*(result_y==self.test_y).sum() / len(result_y)                     #Return the accuracy





#Main function
filename = ["car.data.txt","connect-4.data","agaricus-lepiota.data.csv","kr-vs-kp.data.txt","nursery.data"]
savename = ["car","connect","mushroom","chess","nursery"] 
epsilons = [0.5, 1, 2, 3, 4, 5]                                                     #Epsilon values
encoding = ["THE","SHE"]                                                            #0 - THE | 1 - SHE       
enc_id = 0                                                                             


for ind,file in enumerate(filename):
    location = os.path.join(os.path.dirname(__file__), 'datasets/'+file)            #Access dataset
    data = pd.read_csv(location)                                                    #Read dataset in dataframe
    
    nb = heldp(data, encoding[enc_id])                                              #Initialize and pre_process data
    nb.split(train_size=0.90, random_state=496)                                     #Split data into train & test
    acc_mat = np.zeros((len(epsilons), 2))                                          #Accuracy matrix of excel sheet

    print("Processing dataset:",file)
    print(encoding[enc_id]+" protocol")
    for index, eps in enumerate(epsilons):
        total_acc = 0                                                               #Total accuracy
        print ("*****************")
        for i in range(100):
            nb.add_noise(eps)
            nb.aggregate()
            nb.train()
            acc = nb.classify()
            total_acc += acc
        total_acc = total_acc / 100
        acc_mat[index,0] = eps                                                      #Set epsilon value in accuracy matrix
        acc_mat[index, 1] = total_acc                                               #Set accuracy in accuracy matrix
        
        print ("Epsilon:",eps,". Average accuracy after 100 runs:", total_acc, "%")
                                                                                    #Save result in a csv file

    result_file = os.path.join(os.path.dirname(__file__),'results/'+savename[ind]+'_'+encoding[enc_id]+'.csv')
    np.savetxt(result_file, acc_mat, delimiter=',', fmt='%.2f')



   

