import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class Score():
  def __init__(self, do_method):
    """
    """
    self.do_method = do_method

    if self.do_method=='multiclass':
      self.accuracy_l, self.loss_l = [], []

    elif self.do_method=='multilabel':
      self.accuracy_l, self.loss_l, self.TPrate_l, self.TNrate_l = [], [], [], []
  
  def add(self, loss, y_hat, y):
    """
    """
    if self.do_method=='multiclass':
      _, predicted = torch.max(y_hat.data, 1)
      accuracy = (100. * (predicted == y).sum() / y.numel()).detach().cpu().numpy()
    
    elif self.do_method=='multilabel':    
      predicted = (y_hat>0).float()
      #_, predicted = torch.max(y_hat, 1)
      accuracy = (100. * (predicted == y).sum() / y.numel()).detach().cpu().numpy()
      TPrate = (100. * ((predicted == y) & (y==1)).sum() / (y==1).sum()).detach().cpu().numpy()
      TNrate = (100. * ((predicted == y) & (y==0)).sum() / (y==0).sum()).detach().cpu().numpy()
      self.TPrate_l.append( float(TPrate) )
      self.TNrate_l.append( float(TNrate) )
    
    self.loss_l.append( float(loss) )
    self.accuracy_l.append( float(accuracy) )
      
  def print(self, text, num_epoch):
    """
    """
    if self.do_method=='multiclass':
      print('{} Epoch:{} \t loss:{:.3f} \t accuracy:{:.2f}'.format(text, num_epoch, np.mean(self.loss_l), np.mean(self.accuracy_l)))

    elif self.do_method=='multilabel':    
      print('{} Epoch:{} \t loss:{:.3f} \t accuracy:{:.2f} \t TPrate:{:.2f} \t TNrate:{:.2f}'.format(text, num_epoch, np.mean(self.loss_l), np.mean(self.accuracy_l), np.mean(self.TPrate_l), np.mean(self.TNrate_l)))

  def get_summary(self):
    """
    """
    if self.do_method=='multiclass':
      output = np.mean(self.loss_l), np.mean(self.accuracy_l)

    elif self.do_method=='multilabel':    
      output = np.mean(self.loss_l), np.mean(self.accuracy_l), np.mean(self.TPrate_l), np.mean(self.TNrate_l)
    return output
  

  def get_name(self):
    """
    """
    if self.do_method=='multiclass':
      output = ['loss', 'accuracy']

    elif self.do_method=='multilabel':    
      output = ['loss', 'accuracy', 'TPrate', 'TNrate']

    return output

  def plot_curve(self, score_train_l, score_test_l):
    """
    """
    score_name_l = self.get_name()

    plt.figure(figsize=(16,4))

    nb_score = len(score_train_l[0])
    for num_score in range(nb_score):
      # --- for each score, we goes over epoch and get back the corresponding score
      # --- we do this for the train and the test
      store_train_l, store_test_l = [], []
      for score_train in score_train_l:
        store_train_l.append(score_train[num_score])
      for score_test in score_test_l:  
        store_test_l.append(score_test[num_score])
      #print(score_name_l[num_score])
      #print(store_train_l)
      #print(store_test_l)
      plt.subplot(1, nb_score, num_score+1)
      plt.plot(store_train_l, 'g', store_test_l, 'r')
      plt.legend(['train', 'test'])
      plt.grid(True), plt.xlabel('# Epoch'); plt.ylabel(score_name_l[num_score])