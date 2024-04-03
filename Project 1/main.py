import numpy as np
import cvxpy as cp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
import time

def loaddata():
    cancer = np.loadtxt('breast-cancer-wisconsin.txt')[:,1:]
    diabetes = np.genfromtxt('diabetes.csv', delimiter=',')
    sonar = np.genfromtxt('sonar_csv.csv', delimiter=',')
    return cancer, diabetes, sonar

def MPM(data, data_name):
    start_MPM = time.time()
    accuracy_total = []
    error_total = []
    for i in np.arange(0,10):
        # 随机分割数据为训练集和测试集
        np.random.shuffle(data)
        split_index = int(data.shape[0] * 0.9)
        train_data = data[:split_index]
        test_data = data[split_index:]
        
        # 计算训练集中不同类别数据的均值和协方差
        data_0 = train_data[train_data[:,-1] == 0][:,:-1]
        data_1 = train_data[train_data[:,-1] == 1][:,:-1]
        mean_0 = np.mean(data_0, axis=0)
        mean_1 = np.mean(data_1, axis=0)
        cov_0 = np.cov(data_0, rowvar=False)
        cov_1 = np.cov(data_1, rowvar=False)
        cov0 = sqrtm(cov_0)
        cov1 = sqrtm(cov_1)

        # 用MPM方法计算w和b
        wx = cp.Variable(data.shape[1]-1)
        objective = cp.Minimize(cp.norm(cov0@wx) + cp.norm(cov1@wx))
        constraints = [wx.T@(mean_0-mean_1) == 1]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        w = wx.value
        b = w.T@mean_0 - np.sqrt(w.T@cov_0@w)/(np.sqrt(w.T@cov_0@w)+np.sqrt(w.T@cov_1@w))

        # 计算error
        kappa = 1/(np.sqrt(w.T@cov_0@w)+np.sqrt(w.T@cov_1@w))
        error = 1/(1+kappa**2)
        error_total.append(error)

        # 计算准确率
        amount = test_data.shape[0]
        T = 0
        for j in np.arange(0,amount):
            xj = test_data[j,:-1]
            yj = test_data[j,-1]
            gj = np.dot(w.T,xj) - b
            if gj >= 0 and yj == 0:
                T = T+1
            if gj <= 0 and yj == 1:
                T = T+1
        accuracy = T/amount
        accuracy_total.append(accuracy)
    end_MPM = time.time()
    print("The average MPM's guaranteed error on the " + data_name + " dataset is {:.2f}%.".format(np.mean(error)*100))
    print("The classification accuracy on the " + data_name + " dataset using MPM is {:.2f}%.".format(np.mean(accuracy_total)*100))
    print("MPM run time on the " + data_name + " dataset: ", end_MPM - start_MPM, "s")

def rbf_kernel(X, Y, sigma=1.0):
    return np.exp(-cdist(X, Y, 'sqeuclidean')/(2 * sigma ** 2))

def make_symmetric(matrix):
    return 0.5 * (matrix + matrix.T)

def kernelized_MPM(data, data_name, kernel_function):
    start_time = time.time()
    accuracy_total = []
    error_total = []
    for i in np.arange(0,10):
        np.random.shuffle(data)
        split_index = int(data.shape[0] * 0.9)
        train_data = data[:split_index]
        test_data = data[split_index:]
        
        data_0 = train_data[train_data[:,-1] == 0][:,:-1]
        data_1 = train_data[train_data[:,-1] == 1][:,:-1]
        
        mean_0 = np.mean(data_0, axis=0)
        mean_1 = np.mean(data_1, axis=0)
    
        # Compute the kernel matrix using all training data
        train_data_features = train_data[:,:-1]
        K = kernel_function(train_data_features, train_data_features)

        # Then, when computing K0_centered and K1_centered, use masking to only
        # use the examples from each class, but keep the full size of the kernel matrix
        mask_0 = train_data[:,-1] == 0
        mask_1 = train_data[:,-1] == 1

        K0_centered = K - np.outer(mask_0, mask_0.T @ K) / np.sum(mask_0)
        K1_centered = K - np.outer(mask_1, mask_1.T @ K) / np.sum(mask_1)
        
        # Compute the square roots of the centered kernel matrices
        eigval0, eigvec0 = np.linalg.eigh(K0_centered)
        eigval1, eigvec1 = np.linalg.eigh(K1_centered)

        eigval0[eigval0 < 0] = 0
        eigval1[eigval1 < 0] = 0

        K0_centered_sqrt = eigvec0 @ np.diag(np.sqrt(eigval0)) @ eigvec0.T
        K1_centered_sqrt = eigvec1 @ np.diag(np.sqrt(eigval1)) @ eigvec1.T

        K0_centered_sqrt = make_symmetric(K0_centered_sqrt)
        K1_centered_sqrt = make_symmetric(K1_centered_sqrt)

        # Compute the dual problem
        alpha = cp.Variable(K0_centered_sqrt.shape[0])
        objective = cp.Minimize(cp.quad_form(alpha, K0_centered_sqrt) + cp.quad_form(alpha, K1_centered_sqrt))
        
        # Calculating the mean in the kernel space
        mean_0_kernel = cp.sum(alpha[mask_0]) / cp.sum(mask_0)
        mean_1_kernel = cp.sum(alpha[mask_1]) / cp.sum(mask_1)

        constraints = [mean_0_kernel - mean_1_kernel == 1]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CVXOPT)
        alpha_val = alpha.value
        b = alpha_val.T @ mean_0 - np.sqrt(alpha_val.T @ K0_centered @ alpha_val) / (np.sqrt(alpha_val.T @ K0_centered @ alpha_val) + np.sqrt(alpha_val.T @ K1_centered @ alpha_val))
        w = alpha_val

        # 计算error
        kappa = 1/(np.sqrt(w.T@cov_0@w)+np.sqrt(w.T@cov_1@w))
        error = 1/(1+kappa**2)
        error_total.append(error)

        # 计算准确率
        amount = test_data.shape[0]
        T = 0
        for j in np.arange(0,amount):
            xj = test_data[j,:-1]
            yj = test_data[j,-1]
            gj = np.dot(w.T,xj) - b
            if gj >= 0 and yj == 0:
                T = T+1
            if gj <= 0 and yj == 1:
                T = T+1
        accuracy = T/amount
        accuracy_total.append(accuracy)
    end_time = time.time()
    print("The average MPM's (with Gaussian kernel) guaranteed error on the " + data_name + " dataset is {:.2f}%.".format(np.mean(error)*100))
    print("The classification accuracy on the " + data_name + " dataset using MPM (with Gaussian kernel) is {:.2f}%.".format(np.mean(accuracy_total)*100))
    print("MPM (with Gaussian kernel) run time on the " + data_name + " dataset: ", end_time - start_time, "s")

def evaluate_model(model, data, data_name):
    start_model = time.time()
    accuracy_total = []
    for _ in range(10):
        np.random.shuffle(data)
        split_index = int(data.shape[0] * 0.9)
        X_train, X_test = data[:split_index, :-1], data[split_index:, :-1]
        y_train, y_test = data[:split_index, -1], data[split_index:, -1]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracy_total.append(accuracy)
    end_model = time.time()
    print("The classification accuracy on the " + data_name + " dataset using " + model.__class__.__name__ + " is {:.2f}%.".format(np.mean(accuracy_total)*100))
    print(model.__class__.__name__ + " run time on the " + data_name + " dataset: ", end_model - start_model, "s")

def main():
    cancer, diabetes, sonar = loaddata()
    MPM(cancer, 'cancer')
    MPM(diabetes, 'diabetes')
    MPM(sonar, 'sonar')

    lda = LDA()
    logistic = LogisticRegression(max_iter=10000)
    svc = svm.SVC(max_iter=10000)

    for model in [lda, logistic, svc]:
        evaluate_model(model, cancer, 'cancer')
        evaluate_model(model, diabetes, 'diabetes')
        evaluate_model(model, sonar, 'sonar')

def kernel():
    kernelized_MPM(cancer, 'cancer', rbf_kernel)
    kernelized_MPM(diabetes, 'diabetes', rbf_kernel)
    kernelized_MPM(sonar, 'sonar', rbf_kernel)

main()

# kernel() # Failed to converge.



