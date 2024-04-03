import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import cvxpy as cp

# 定义矩阵
matrix_p = np.array([[4, -1], [-1, 1]])
matrix_n = np.array([[1, 1], [1, 2]])

# 计算矩阵的平方根
sqrt_matrix_p = sqrtm(matrix_p)
sqrt_matrix_n = sqrtm(matrix_n)

def plot_ellipse(center, matrix, kappa):
    # 创建单位圆上的点
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.vstack((np.cos(theta), np.sin(theta)))  # shape=(2,100)

    # 通过扩大 kappa，旋转和拉伸来转换圆
    ellipse = center[:,None] + matrix @ (kappa * circle)

    # 在椭圆上标注 kappa 值
    plt.text(ellipse[0,0], ellipse[1,0], f'{kappa}', fontsize=10, ha='right')

    # 绘制椭圆
    plt.plot(ellipse[0,:], ellipse[1,:])

# 画出不同kappa的椭圆
for kappa in [0.5, 1, 1.512]:
    plot_ellipse(np.array([0, 0]), sqrt_matrix_p, kappa)
    plot_ellipse(np.array([4, 2]), sqrt_matrix_n, kappa)

# 显示图形
plt.grid(True)
plt.axis('equal')
plt.show()

# 由MPM求解kappa
wx = cp.Variable(2)
objective = cp.Minimize(cp.norm(sqrt_matrix_p@wx)+cp.norm(sqrt_matrix_n@wx))
constraints = [wx.T@(np.array([0, 0])-np.array([4, 2])).T == 1]
prob = cp.Problem(objective, constraints)
s = prob.solve()
w = wx.value
kappa = 1/s
print(kappa)
