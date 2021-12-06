from __future__ import division
import pandas as pd
import numpy as np
import os
from math import ceil



class deldp:

    def __init__(self, data): 

        data_X = data.iloc[:, :-1] 
        data_y = data.iloc[:, -1]  
        self.pre_process(data_X, data_y)
        self.de_enc()


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
        
    
    
    def de_enc(self):
        
        X = self.num_X
        y = self.num_y

        
        self.col_X = X.shape[1]                                                     #Columns for X
        self.col_y = 1                                                              #Columns for y
        
        #Full Matrix [#samples , (#features * #classes) + 1(y)]
        self.full_mat = np.zeros((X.shape[0], (self.col_X * len(self.cnt_y) ) + self.col_y))  
        for clInd in range(len(self.cnt_y)):                                        #Fill random values in full matrix
            for i, att_count in enumerate(self.ft_cnt):                         
                self.full_mat[:, (clInd*self.col_X)+i] = np.random.randint(att_count, size=X.shape[0])
              
        for i in range(X.shape[0]):                                                 #Fill sample values in full matrix
            col_start = self.num_y[i] * self.col_X
            self.full_mat[i, col_start:(col_start+self.col_X)] = X[i, :]
        
        clInd = len(self.cnt_y) * self.col_X
        self.full_mat[:, clInd] = self.num_y                                         #Fill y values in full matrix
        
        
    


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
        print(self.train_X.shape,"trfull")

    
    def flip_coin(self, p, M, N, tol=0.1):                                          #Flip coin based on 
        condition = True
        while condition:                                                            #Loop until acceptable tolerance
            rndMat = np.random.random((M, N)) if N > 1 else np.random.random(M)
            rndMat = (rndMat < p).astype(int)
            prop = np.sum(rndMat) / (M * N)
            condition = np.abs(p - prop) > tol
        return rndMat

    def probability(self, d):                                                       #Calculate probability based on epsilon
        p = np.exp(self.eps) / (np.exp(self.eps) + d - 1)
        q = 1.0 / (np.exp(self.eps) + d - 1)
        return p, q

    
    def add_noise(self, eps):

        np.random.seed()
        self.pert_mat = np.zeros(self.train_X.shape)        
        self.eps = eps

        pert_list = np.zeros(self.train_X.shape[1])                                 #Intialize list for perturbation
        pert_list[:(self.train_X.shape[1]-1)] = np.tile(self.ft_cnt, len(self.cnt_y))
        pert_list[self.train_X.shape[1]-1] = len(self.cnt_y)


        for ind, val in enumerate(pert_list):                      
            p, _ = self.probability(d=val)                                          #Calculate probability based on epsilon          
            K = self.flip_coin(p, self.train_X.shape[0], 1)                         #Random matrix based on probability          
            for i in range(self.train_X.shape[0]):
                if K[i] == 1:
                    self.pert_mat[i, ind] = self.train_X[i, ind]                    #Save true value
                else:            
                    sel = np.random.randint(val-1)                                  #Save noise 
                    if sel < self.train_X[i, ind]:
                        self.pert_mat[i, ind] = sel
                    else:
                        self.pert_mat[i, ind] = sel + 1 
            
        
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

        ind = len(self.cnt_y) * self.col_X
        idx, count = np.unique(self.pert_mat[:, ind], return_counts=True)           #Get occurence of each class
        cls_sum[idx.astype(int)] = count
        for clInd in range(len(self.cnt_y)):                                        #Compute attribute count
            dtCol = clInd * self.col_X
            att_ind = 0
            for i, att_count in enumerate(self.ft_cnt):
                idx, count = np.unique(self.pert_mat[:, dtCol+i], return_counts=True)
                att_sum[clInd, att_ind+idx.astype(int)] = count
                att_ind += att_count


        self.est_att_sum = np.zeros(att_sum.shape)                                  #Attribute sum estimate
        self.est_cls_sum = np.zeros(cls_sum.shape)                                  #Class sum estimate

        p, q = self.probability(d=len(self.cnt_y))                                  #Probability calculation based on eps
        self.est_cls_sum = self.calc_sum(cls_sum, self.pert_mat.shape[0], p, q)
        att_ind = 0
        for att_count in self.ft_cnt:
            p, q = self.probability(d=att_count)
            self.est_att_sum[:, att_ind:(att_ind+att_count)] = self.calc_sum(att_sum[:, att_ind:(att_ind+att_count)],self.pert_mat.shape[0], p, q)
            att_ind += att_count
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


for ind,file in enumerate(filename):
    location = os.path.join(os.path.dirname(__file__), 'datasets/'+file)            #Access dataset
    data = pd.read_csv(location)                                                    #Read dataset in dataframe
    
    nb = deldp(data)                                                                #Initialize and pre_process data
    nb.split(train_size=0.95, random_state=496)                                     #Split data into train & test
    acc_mat = np.zeros((len(epsilons), 2))                                          #Accuracy matrix of excel sheet

    print("Processing dataset:",file)
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

    result_file = os.path.join(os.path.dirname(__file__),'results/'+savename[ind]+'_DE.csv')
    np.savetxt(result_file, acc_mat, delimiter=',', fmt='%.2f')

    break


        
        

