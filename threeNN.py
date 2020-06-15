import numpy
import scipy.special
import matplotlib.pyplot

class neuralNetwork :
    # 初始化函数——设定输入层节点、隐藏层结点和输出层节点的个数
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 输入、隐藏、输出
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 学习率
        self.Ir = learningrate

        # 链接权重，即输入层到隐藏层的权重、隐藏层到输出层的权重
        # 注！：随机生成链接初始权重, 3*3 的数值范围在（-0.5，0.5）随机数组
        #       numpy.random.rand(3, 3) - 0.5
        # self.wih = (numpy.random.rand(self.hnodes , self.innodes) - 0.5)
        # self.who = (numpy.random.rand(self.onodes , self.hnnodes) - 0.5)

        # 初始化权重（利用正态分布）
        # 正态分布中心值为0.0
        # 标准方差：pow()即node的-0.5次方
        # numpy数组的形状大小，行高列宽
        # 下面为ho、ih之间权重
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 定义调用激活函数的匿名函数，用于修改激活函数内部代码
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # 训练——学习给定训练集样本后，优化权重
    def train(self, inputs_list, targets_list):
        # 把输入和预结果的数组传进来
        # ndmin 指生成数组的最小维度
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 隐藏层的输入值 = 输入值 x 权重的乘积
        hidden_inputs = numpy.dot(self.wih, inputs)

        # 隐藏层的输出值 = 对输入值调用激活函数
        hidden_outputs = self.activation_function(hidden_inputs)

        # 输出层和隐藏层
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 计算输出值与预计值的误差
        output_errors = targets - final_outputs

        # 再用误差 点乘 ho间的权重的转置矩阵 得出隐藏值的优化
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 更新输出层和隐藏层之间的权重
        self.who += self.Ir * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        # 下面和上面类似，改变的是输入层和隐藏层的权重
        self.wih += self.Ir * numpy.dot((hidden_errors * hidden_outputs*(1.0-hidden_outputs)), numpy.transpose(inputs))
        pass

    # 查询——给定输入，从输出节点给出答案
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        # 将输入与权重相乘得出隐藏层值
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 隐藏层的输出值为上面这个隐藏层值调用激活函数后的结果
        hidden_outputs = self.activation_function(hidden_inputs)

        # 同上
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 同上
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# 输入层，隐藏层，输出层的个数
input_nodes = 784
hidden_node = 100
output_nodes = 10

# 学习率
learning_rate = 0.3

# 建立神经网络
n = neuralNetwork(input_nodes, hidden_node, output_nodes, learning_rate)



# 导入csv文件
# open 函数创建一个句柄和一个引用，句柄给data_file，后面的操作都用这个句柄完成
training_data_file = open("E:/专业/mnist_dataset/mnist_train_100.csv", 'r')

# training_data_list[0]表示第一行记录
training_data_list = training_data_file.readlines()
training_data_file.close()

for record in training_data_list:
    all_values = record.split(',')
    # 对像素范围[0,255]进行缩放，缩放到[0.01,1.0],即x*0.99+0.01
    # 缩放的目的是为了让特征点的差距欧氏距离减少，以防止太大的特征点权影响因子过大
    inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
    # 创建用零填充的数组,长度为output_nodes,第一个元素即正确的那个标签为0.99，其余为0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    pass




# 以上是对模型进行训练
# 接下来就是测试网络



# 导入数据
test_data_file = open("E:/专业/mnist_dataset/mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# 测试核心
all_values = test_data_list[0].split(',')
data = n.query(numpy.asfarray(all_values[1:])/255.0*0.99+0.01)

# 判断输出的数字是几
max_num = 0
for i in range(len(data)):
    if data[i] > max_num:
        max_num = data[i]
        num = i
print("图片上的数字是:", num)





# 输出图片部分
# numpy.asfarray()将文本字符串转换成实数，并创建这些数字的数组(string转int)
# [1：]，表示采用除了列表中的第一个元素以外的所有值
# reshape((28, 28))形成28行28列的矩阵形式
image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
# plt.imshow()函数负责对图像进行处理，并显示其格式
# cmap = 'Greys'灰度调色板
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')

# plt.show()则是将plt.imshow()处理后的函数显示出来
matplotlib.pyplot.show()