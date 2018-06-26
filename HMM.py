import numpy as np
import matplotlib.pyplot as plt
from pystruct.inference import inference_dispatch

n_samples=500

d='12,12,11,11,10,9,8,8,7,6,6,6,7,8,8,8,6,5,4,3,3,2,1,0,1,3,4,5,6,8,8,9,9,10,11,12,13,14,14,14,15,15,15,15'
d=np.array([float(c) for c in d.split()])

nClasses=100#離散クラスの数
p=20#隣接するノードのクラスが異なった場合のコスト、同じ場合は0

#データ項,ラベルとの差の絶対値
unaries=np.array([abs(i-j) for j in range(nClasses)] for i in d)

#ペアワイズ項について、同じラベルなら0、異なるラベルの場合p
pairwise=np.array((np.ma.ones((nClasses,nClasses))-np.eye(nClasses))*p)
