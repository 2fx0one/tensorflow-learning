import tensorflow as tf

# english https://jacobbuckman.com/post/tensorflow-the-confusing-parts-1/#understanding-tensorflow
# chinese https://mp.weixin.qq.com/s/wWlBKzXW5g-WCqIXUQPQjA
# Tensorflow 是用于表示某种类型的计算抽象（称为“计算图”）的框架

# ====第一个关键抽象：计算图====
# tf.constant 新建节点 包含常量
node = tf.constant(1)
node2 = tf.constant(2)

# + 操作在 Tensorflow 中重载，
# 所以同时添加两个张量会在图中增加一个节点，尽管它看起来不像是 Tensorflow 操作
sum_node = node + node2

print(node)
print(node2)
print(sum_node)


# ==== 第二个关键抽象：会话 ====
# 创建会话
# 会话包含一个指向全局图的指针，该指针通过指向所有节点的指针不断更新。这意味着在创建节点之前还是之后创建会话都无所谓。
sess = tf.Session()

# 可以使用 sess.run(node) 返回节点的值，并且 Tensorflow 将执行确定该值所需的所有计算。
print(sess.run(sum_node))

# 还可以传递一个列表，sess.run([node1，node2，...])，并让它返回多个输出
print(sess.run([node, node2, sum_node]))
# 一般来说，sess.run() 调用往往是最大的 TensorFlow 瓶颈之一，
# 所以调用它的次数越少越好。可以的话在一个 sess.run() 调用中返回多个项目，而不是进行多个调用。


#占位符 和 feed_dict
# 一个实用的应用可能涉及构建这样一个计算图：它接受输入，以某种（一致）方式处理它，并返回一个输出。
# 最直接的方法是使用占位符。占位符是一种用于接受外部输入的节点。
input_placeholder = tf.placeholder(tf.int32)

# 为了提供一个值，我们使用 sess.run() 的 feed_dict 属性。
# 注意传递给 feed_dict 的数值格式
print(sess.run(input_placeholder, feed_dict={input_placeholder: 2}))

# ==== 第三个关键抽象：计算路径 ====
node3 = tf.constant(3)

# 当我们在依赖于图中其他节点的节点上调用 sess.run() 时，我们也需要计算这些节点的值。
# 如果这些节点有依赖关系，那么我们需要计算这些值（依此类推......），直到达到计算图的“顶端”，也就是所有的节点都没有前置节点的情况。

# Tensorflow 仅通过必需的节点自动路由计算这一事实是它的巨大优势。如果计算图非常大并且有许多不必要的节点，
# 它就能节约大量运行时间。它允许我们构建大型的“多用途”图形，这些图形使用单个共享的核心节点集合根据采取的计算路径来做不同的任务。
# 对于几乎所有应用程序而言，根据所采用的计算路径考虑 sess.run() 的调用方法是很重要的。
print(sess.run(node3))
sum_node = input_placeholder + node3
print(sess.run(sum_node, feed_dict={input_placeholder: 2}))

# 变量和副作用
# 第三种节点 变量
# tf.constant（每次运行都一样）和 tf.placeholder（每次运行都不一样）
# 通常情况下具有相同的值，但也可以更新成新值。这个时候就要用到变量.

# 了解变量对于使用 Tensorflow 进行深度学习来说至关重要，因为模型的参数就是变量.
# 在训练期间，你希望通过梯度下降在每个步骤更新参数，但在计算过程中，你希望保持参数不变，并将大量不同的测试输入集传入到模型中。
# 模型所有的可训练参数很有可能都是变量。

# 创建变量，请使用 tf.get_variable()
# name 是一个唯一标识这个变量对象的字符串。它在全局图中必须是唯一的，所以要确保不会出现重复的名称。
# shape 是一个与张量形状相对应的整数数组，它的语法很直观——每个维度对应一个整数，并按照排列。
# 例如，一个 3×8 的矩阵可能具有形状 [3,8]。要创建标量，请使用空列表作为形状：[]。
count_variable = tf.get_variable("count", [])

# 一个变量节点在首次创建时，它的值基本上就是“null”，任何尝试对它进行计算的操作都会抛出这个异常。
# 我们只能先给一个变量赋值后才能用它做计算。有两种主要方法可以用于给变量赋值：初始化器和 tf.assign()。我们先看看 tf.assign()：
zero_node = tf.constant(0.)
assign_node = tf.assign(count_variable, zero_node)
sess.run(assign_node)
print(sess.run(count_variable))

# tf.assign(target,value) 有一些独特的属性：
# - 标识操作。tf.assign(target,value) 不做任何计算，它总是与 value 相等。
# - 副作用。当计算“流经”assign_node 时，就会给图中的其他节点带来副作用。
#   在这种情况下，副作用就是用保存在 zero_node 中的值替换 count_variable 的值。
# - 非依赖边。即使 count_variable 节点和 assign_node 在图中是相连的，两者都不依赖于其他节点。
#   这意味着在计算任一节点时，计算不会通过该边回流。不过，assign_node 依赖 zero_node，它需要知道要分配什么。

# “副作用”节点充斥在大部分 Tensorflow 深度学习工作流中，因此，请确保你对它们了解得一清二楚。
# 当我们调用 sess.run(assign_node) 时，计算路径将经过 assign_node 和 zero_node。

# 初始化器
const_init_node = tf.constant_initializer(0.)
count_variable = tf.get_variable("count2", [], initializer=const_init_node)
# print(sess.run(count_variable))
# 问题在于会话和图之间的分隔。我们已经将 get_variable 的 initializer 属性指向 const_init_node，
# 但它只是在图中的节点之间添加了一个新的连接。我们还没有做任何与导致异常有关的事情：
# 与变量节点（保存在会话中，而不是图中）相关联的内存仍然为“null”。我们需要通过会话让 const_init_node 更新变量。

# sess.run(tf.global_variables_initializer())
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(count_variable))

# ==== 优化器 ====
# 在深度学习中，典型的“内循环”训练如下：
#
#     获取输入和 true_output
#
#     根据输入和参数计算出一个“猜测”
#
#     根据猜测和 true_output 之间的差异计算出一个“损失”
#
#     根据损失的梯度更新参数

# 建立计算图
# 设置变量
a = tf.get_variable("a", [], initializer=tf.constant_initializer(0.))
b = tf.get_variable("b", [], initializer=tf.constant_initializer(0.))

a = tf.Variable(0., name="x")
b = tf.Variable(0., name="y")
input_placeholder = tf.placeholder(tf.float32)
output_placeholder = tf.placeholder(tf.float32)

x = input_placeholder
y = output_placeholder
# 建立模型
y_guess = a * x + b

# 损失函数
loss = tf.square(y - y_guess)
loss = tf.reduce_mean(loss)

# 设置优化器
optimizer = tf.train.GradientDescentOptimizer(1e-3)
train_op = optimizer.minimize(loss)

# 创建会话

# 初始话变量
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)


import random

true_a = random.random()
true_b = random.random()

for i in range(100):
    input_data = random.random()
    output_data = true_a * input_data + true_b

    _loss, _ = session.run([loss, train_op],
                           feed_dict={input_placeholder: input_data, output_placeholder: output_data})
    print(i, _loss)

print("True parameters:     a=%.4f, b=%.4f" % (true_a, true_b))
print("Learned parameters:  a=%.4f, b=%.4f" % tuple(session.run([a, b])))
