第一次作业

任务要求：

以谱聚类或者马尔科夫聚类对**鸢尾花数据集**进行处理，得到一图，并输出正确率。





任务思考及步骤：

步骤：
1.计算距离矩阵（例如欧氏距离）
2.利用KNN计算邻接矩阵 AA
3.由 AA 计算度矩阵 DD 和拉普拉斯矩阵 LL
4.标准化 L→D−1/2LD−1/2L→D−1/2LD−1/2
5.对矩阵 D−1/2LD−1/2D−1/2LD−1/2 进行特征值分解，得到特征向量
6.将样本送入 Kmeans 聚类
7.获得聚类结果 C=(C1,C2,⋯,Ck)



任务难点：

#利用KNN计算邻接矩阵 A

    def myKNN(S, k, sigma=3):
        N = len(S)
        A = np.zeros((N,N))
    
        for i in range(N):
            dist_with_index = zip(S[i], range(N))
            dist_with_index = sorted(dist_with_index, key=lambda x:x[0])
            neighbours_id = [dist_with_index[m][1] for m in range(k+1)] # xi's k nearest neighbours
    
            for j in neighbours_id: # x【j】 is x【i】's neighbour
                A[i][j] = np.exp(-S[i][j]/2/sigma/sigma)
                A[j][i] = A[i][j] # mutually
        return A
根据点求邻接矩阵；

如何得到邻接矩阵：

基本思想是，距离较远的两个点之间的边权重值较低，而距离较近的两个点之间的边权重值较高，不过这仅仅是定性，我们需要定量的权重值。一般来说，我们可以通过样本点距离度量的相似矩阵SS来获得邻接矩阵W





在这里构建邻接矩阵有三种方法：

ϵϵ-邻近法

K邻近法

全连接法



求拉普拉斯矩阵：

拉普拉斯矩阵：

L=D−W

D即为我们第二节讲的度矩阵，它是一个对角矩阵。而W即为我们第二节讲的邻接矩阵





谱聚类算法的主要缺点有：

1如果最终聚类的维度很高，则由于降维的幅度不够，谱聚类的运行速度和最后的聚类效果均不好。

2) 聚类效果依赖于相似矩阵，不同的相似矩阵得到的最终聚类效果可能很不同。









