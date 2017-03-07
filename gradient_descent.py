import numpy as np

train_data = np.array([[1.1,1.5],[1.3,1.9],[1.5,2.3],[1.7,2.7],[1.9,3.1],[2.1,3.5],[2.3,3.9],[2.5,4.3],[2.7,4.7],[2.9,5.1]])
train_label = np.array([2.5,3.2,3.9,4.6,5.3,6,6.7,7.4,8.1,8.8])

def batchGradientDescent(x, y, theta, alpha, m, maxIterations):
    xTrains = x.transpose()
    for i in range(0, maxIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        gradient = np.dot(xTrains, loss) / m
        theta = theta - alpha * gradient

    return theta

m, n = np.shape(train_data)
theta = np.ones(n)
alpha = 0.1
maxIterations = 5000

theta = batchGradientDescent(train_data, train_label, theta, alpha, m, maxIterations)

x = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])

print np.dot(x, theta)
