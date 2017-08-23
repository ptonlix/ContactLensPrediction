# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:16:01 2017

@author: Lenovo
"""
from math import log
import numpy
import operator
import treePlotter
'''
函数功能：计算香侬熵
'''
def calcShannonEnt(dataSet):
    numEntries = len(dataSet) #计算数据的个数
    labelCounts = {} #初始化字典
    for featVec in dataSet:
        currentLabel = featVec[-1] #获取每一条数据的分类
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1 #这三句，统计每一个分类出现的次数
    shannonEnt = 0.0 #香侬熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries #计算每一种分类出现的概率
        shannonEnt -= prob * log(prob, 2) # 计算熵的公式
    return shannonEnt
'''
函数功能：划分数据集
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = [] #初始化列表
    for featVec in dataSet:
        if featVec[axis] == value: #比较第axis中的特征量与value相同与否
            reducedFeatVec = featVec[:axis] #复制featVec[0:axis]
            reducedFeatVec.extend(featVec[axis + 1:]) #将剩余的的特征也复制，用extend()方法
            retDataSet.append(reducedFeatVec) #将新列表添加进retDataSet中
    return retDataSet
'''
函数功能：选择最好的划分方式
'''
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1 #计算特征量的个数
    baseEntropy = calcShannonEnt(dataSet) #计算整个数据集的基本香侬熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]#用列表推导式计算整个数据集第i个特征量
        uniqueVals = set(featList)#用set生成集合数据，剔除相同值
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value) #根据每一个不同的特征值进行数据集的分割
            prob = len(subDataSet) / float(len(dataSet)) #计算每个子数据集的概率
            newEntropy += prob * calcShannonEnt(subDataSet) #累加每一个字数据集的熵
        infoGain = baseEntropy - newEntropy #信息增益是熵的减少或者是数据无序度的减少，减少的越多说明分类越合适
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
'''
函数功能：返回出现次数最多的分类名称，当数据集处理完所有属性，类标签还不是唯一的时候，
         采用多数投票的方法决定该叶子节点的分类
'''
def majorityCnt(classList): #返回出现次数最多的分类名称
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), \
                              key=operator.itemgetter(1), reverse=True)
    return classCount[0][0]

'''
函数功能:构造决策树
'''
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet] #获取数据集中出现的类别
    if classList.count(classList[0]) == len(classList): #类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1: #遍历完所有特征时返回出现次数最多的类别
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet) #选取做合适的划分特征量
    bestFeatLabel = labels[bestFeat] #保存特征量的偏移量
    myTree = {bestFeatLabel:{}} #用字典存储数的结构
    del(labels[bestFeat]) #
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:] #为了保证每次调用createTree()时不改变原始列表的内容，使用新变量subLabels代替原始列表
        myTree [bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),\
               subLabels)
    return myTree #返回构造的决策树
'''
函数功能：利用pickle来存储构造好的决策树
'''
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()
'''
函数功能：获取构造好的决策树
'''
def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

'''
函数功能：构造分类器
'''
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0] #获取决策树，第一个分类
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr) #获取输入的的数据的Label偏移量
    for key in secondDict.keys():
        if testVec[featIndex] == key: #读出输入数据的值与决策树对应分类的值进行比较
            if type(secondDict[key]).__name__ == 'dict' :#如果这个值对应的是一个字典，说明还没找到合适的分类，继续递归去找
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: #如果不是，就说明已经匹配出结果，返回匹配的类型结果
                classLabel = secondDict[key]
    return classLabel

'''
函数功能：获取隐形眼镜的数据
'''
def GetContactLensesData(filename):
    fr = open(filename)
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    return lenses, lensesLabels

'''
函数功能：交互方法 
'''
def ChooseContactLenses():
    lenses, lensesLabels = GetContactLensesData(r'D:\Learning\DataSet\lenses.txt')
    labels = lensesLabels.copy() #因为生成决策树时，修改了labels，我们需要拷贝一份
    lensesTree =  createTree(lenses, labels)
    treePlotter.createPlot(lensesTree)
    
    tearRate = input("How many tears？Options：reduced/normal\n")
    astigmatic = input("Does the eye have astigmatic? Options: yes/no\n")
    prescript = input("Are you myope or hyper? Options: myope/hyper\n")
    age = input("How many years you? Options：pre/presbyopic/young\n")
    personList = [age, prescript, astigmatic, tearRate]
    
    classLabel = classify(lensesTree, lensesLabels, personList)
    print("you should choose the contact lenses:%s" %classLabel)

ChooseContactLenses()
