import matplotlib.pyplot as plt
import numpy as np

#網路上找的dataset 可以線性分割
dataset = np.array([
((-0.4, 0.3), -1),
((-0.3, -0.1), -1),
((-0.2, 0.4), -1),
((-0.1, 0.1), -1),
((0.9, -0.5), 1),
((0.7, -0.9), 1),
((0.8, 0.2), 1),
((0.4, -0.6), 1)])

numiter = 100

#判斷有沒有分類錯誤，並列印錯誤率
def check_error(w, b, dataset):
    result = None
    error = 0
    for x, s in dataset:
        x = np.array(x)
        if int(np.sign(w.T.dot(x))+b) != s:  #signum function
            result = x,s
            error += 1
    print(f"error=%s/%s" % (error, len(dataset)))
    return result

#PLA演算法實作
def pla(dataset,numiter):
    w = np.zeros(2)
    b = 0

    i = 0
    while check_error(w, b, dataset) is not None and numiter>0:
        print(f"-----iteration %s-----" % (i))
        x, s = check_error(w, b, dataset)
        w += s * x  #training parameter
        b += s
        numiter -= 1 #counting
        i += 1
    return w,b


#執行
w,b = pla(dataset,numiter)


#畫圖
ps = [v[0] for v in dataset]
fig = plt.figure()  #初始化
ax1 = fig.add_subplot(111) #建立畫板

#dataset前半後半已經分割好 直接畫就是
ax1.scatter([v[0] for v in ps[:4]], [v[1] for v in ps[:4]], s=10, c='b', marker="o", label='O') # [a:] 從index[a]開始到end, [:a] 從index[0]開始到index[a-1]
ax1.scatter([v[0] for v in ps[4:]], [v[1] for v in ps[4:]], s=10, c='r', marker="x", label='X')
l = np.linspace(-2,2)
slope,intercept = -w[0]/w[1], -b/w[1]
ax1.plot(l, slope*l + intercept, 'b-') #plt.plot(x,y,type)
plt.legend(loc='upper left');
plt.show()

