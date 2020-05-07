#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:59:12 2020

@author: jaoba
"""

import pandas as pd

main_data = pd.read_csv('ben-man.csv')

train_data = main_data.iloc[0:19805,:]
test_data = main_data.iloc[19806:20806,:]
eval_data = main_data.iloc[20807:,:]

eval_data.to_csv('eval.csv', index=False)
test_data.to_csv('test.csv', index=False)
train_data.to_csv('train.csv', index=False)