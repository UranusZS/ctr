#!/usr/bin/python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import scipy.special as special


class BayesSmoothing(object):
    """
    refs:
        Click-Through Rate Estimation for Rare Events in Online Advertising
        http://www.cnblogs.com/bentuwuying/p/6498370.html
    """

    def __init__(self, alpha=1, beta=100):
        self.alpha = alpha
        self.beta = beta

    def fit(self, tries, success, iter_num=1000, epsilon=0.00000001):
        """
        fit 
        params:
            tries     impression_nums
            success   click_nums
        return:
            alpha, beta
        """
        assert(len(tries) == len(success))
        self._update_from_data_by_moment(tries, success)
        self._update_from_data_by_FPI(tries, success, iter_num, epsilon)
        return self.alpha, self.beta

    def predict(self, tries, success):
        """
        predict 
        params:
            tries     impression_nums
            success   click_nums
        return:
            ctrs
        """
        if isinstance(tries, int) and isinstance(success, int):
            return (float(success) + self.alpha) / (float(tries) + self.alpha +
                    self.beta)

        assert(len(tries) == len(success))
        res = []
        for i in range(len(tries)):
            imp = tries[i]
            click = success[i]
            res.append((float(click) + self.alpha) / (float(imp) + self.alpha +
                    self.beta))
        return res

    def _update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        '''
        estimate alpha, beta using fixed point iteration
        '''
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        '''
        fixed point iteration
        '''
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        for i in range(len(tries)):
            sumfenzialpha += (special.digamma(success[i]+alpha) - special.digamma(alpha))
            sumfenzibeta += (special.digamma(tries[i]-success[i]+beta) - special.digamma(beta))
            sumfenmu += (special.digamma(tries[i]+alpha+beta) - special.digamma(alpha+beta))

        return alpha*(sumfenzialpha/sumfenmu), beta*(sumfenzibeta/sumfenmu)

    def _update_from_data_by_moment(self, tries, success):
        '''
        estimate alpha, beta using moment estimation
        '''
        mean, var = self.__compute_moment(tries, success)
        self.alpha = (mean+0.000001) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)
        self.beta = (1.000001 - mean) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)

    def __compute_moment(self, tries, success):
        '''
        moment estimation
        '''
        ctr_list = []
        var = 0.0
        for i in range(len(tries)):
            ctr_list.append(float(success[i])/tries[i])
        mean = sum(ctr_list)/len(ctr_list)
        for ctr in ctr_list:
            var += pow(ctr-mean, 2)
        return mean, var/(len(ctr_list)-1)

if __name__ == '__main__':
    print("bayes_smoothing")
    hyper = BayesSmoothing(1, 100)
    I = [73, 2709, 67, 158, 118]
    C = [0, 30, 2, 3, 4]
    alpha, beta = hyper.fit(I, C)
    print("alpha: {0}, beta {1}".format(alpha, beta))
    print(hyper.predict(I, C))
