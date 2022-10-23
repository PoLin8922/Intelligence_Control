import matplotlib.pyplot as plt
import numpy as np

numiter = 100
error = 0
dataset = np.array([
((-0.4, 0.3), -1),
((-0.3, -0.1), -1),
((-0.2, 0.4), -1),
((-0.1, 0.1), -1),
((0.9, -0.5), 1),
((0.7, -0.9), 1),
((0.8, 0.2), 1),
((0.4, -0.6), 1)])

# haha it's fun !!!
def check_error(w, b, dataset):
    result = None
    global error

    error = 0
    for x, s in dataset:
        x = np.array(x)
        if int(np.sign(w.T.dot(x)+b)) != s:
            result = x,s
            error += 1
            
    return result

#PLA Algorithm
def pla(dataset,numiter):
    w = np.zeros(2) #weight
    b = 0  #bias

    i = 0
    while check_error(w, b, dataset) is not None and numiter>0:
        x, s = check_error(w, b, dataset)
        w += s * x  #training parameter
        b += s
        numiter -= 1 #counting
        print("-----iteration %s-----" % (i))
        print("error=%s/%s,  accuracy=%s" % (error, len(dataset), 1 - (float(error)/len(dataset))))
        i += 1

    print("-----iteration %s-----" % (i))
    print("error=%s/%s,  accuracy=%s" % (error, len(dataset), 1 - (float(error)/len(dataset))))
    print("-------success!-------")
    print("error=%s/%s,  accuracy=%s" % (error, len(dataset), 1 - (float(error)/len(dataset))))
    print("weight : %s" % (w))
    print("bias : %s" % (b))

    return w,b


if __name__ == "__main__":
    #excute
    w,b = pla(dataset,numiter)

    #drawing
    ps = [v[0] for v in dataset]
    fig = plt.figure()  #init
    ax1 = fig.add_subplot(111) 

    ax1.scatter([v[0] for v in ps[:4]], [v[1] for v in ps[:4]], s=10, c='b', marker="o", label='O') # [a:] from index[a] to end, [:a] from index[0] to index[a-1]
    ax1.scatter([v[0] for v in ps[4:]], [v[1] for v in ps[4:]], s=10, c='r', marker="x", label='X')
    l = np.linspace(-2,2)
    slope,intercept = -w[0]/w[1], -b/w[1]
    ax1.plot(l, slope*l + intercept, 'b-') #plt.plot(x,y,type)
    plt.legend(loc='upper left');
    plt.show()




