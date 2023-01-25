# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
#畫圖用
import matplotlib.pyplot as plt
#scatter參數(x_axis_data, y_axis_data, s=None, c=None, marker=None, 
# cmap=None, vmin=None, vmax=None, alpha=None, linewidths=None, edgecolors=None)
# s=>點的大小,c=>點的顏色,edgecolor=>點的邊框顏色,marker=>點的形狀
# linewidths=>邊框大小, alpha=>點的透明度 cmap=顏色主題的名稱
#vmin 刻度的最小值  vmax 刻度的最大值

# PDF輸出用
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

# sample data
sampleData1 = np.array([[166, 58.7],[176.0, 75.7],[171.0, 62.1],[173.0, 70.4],[169.0,60.1]])
print(sampleData1)

# 散佈圖格式
for p in sampleData1:
    plt.scatter(p[0], p[1], c='k', s=50)
plt.grid()
plt.xlabel('$x$') #x軸名稱
plt.ylabel('$y$') #x軸名稱
plt.show()

plt.figure(figsize=(10,5)) 
#用更大的圖包住原圖 用平均數171,65.4作為新原點
for p in sampleData1:
    plt.scatter(p[0], p[1], c='k', s=50)
plt.grid()
plt.plot([0,0],[-10,80],c='k',lw=1) #比171最多小10
plt.plot([171,171],[-10,80],c='k')
plt.plot([-10,180],[0,0],c='k',lw=1)
plt.plot([-10,180],[65.4,65.4],c='k')
plt.xlim(-10,180)#手動定義x軸的刻度 平均值＝171 max=176 min=166 zero=-10
plt.ylim(-10,80)#手動定義y軸的刻度 平均值＝65.4 max=75.7 min=58.7 zero=-17
plt.show()

# 計算樣本平均值
means = sampleData1.mean(axis=0)
print(means)

# 將平均值做為轉換座標用
sampleData2 = sampleData1 - means
print(sampleData2)

# 在新座標上畫散佈圖
for p in sampleData2:
    plt.scatter(p[0], p[1], c='k', s=50)
plt.grid()
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.plot([-6,6],[0,0], c='k')
plt.plot([0,0],[-7.5,11],c='k')
plt.xlim(-5.2,5.2)#x軸剛好4個值
plt.show()

# 定義預測函數
def L(W0, W1):
    #損失函數 W0=0 損失最少 
    return(5*W0**2 + 58*W1**2 - 211.2*W1 + 214.96)

# L(0, W1) 的圖形
plt.figure(figsize=(6,6))
W1 = np.linspace(0, 4, 501)
plt.ylim(1,3)
plt.plot(W1, L(0,W1))
plt.scatter(1.82,22.69,s=30)
plt.xlabel('$W_1$')
plt.ylabel('$L(0,W_1)$')
plt.grid()
plt.xlim(0,3.5)
plt.ylim(0,200)
plt.show()

def pred1(X):
    #新座標下的預測函數
    return 1.82*X

# 散佈圖與迴歸線(轉換新座標後)
for p in sampleData2:
    plt.scatter(p[0], p[1], c='k', s=50)
X=np.array([-6,6])
plt.plot(X, pred1(X), lw=1)
plt.plot([-6,6],[0,0], c='k')
plt.plot([0,0],[-11,11],c='k')
plt.xlim(-5.2,5.2)
plt.grid()
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.show()

def pred2(x):
    #轉回原座標的預測函數
    return 1.82*x - 245.9

# 畫出散佈圖與迴歸線
for p in sampleData1:
    plt.scatter(p[0], p[1], c='k', s=50)
x=np.array([166,176])
plt.plot(x, pred2(x), lw=1)
plt.grid()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()
