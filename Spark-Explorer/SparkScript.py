#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:57:42 2016

@author: Sriharish
"""
#import random
import re
import pandas as pd

from pyspark import SparkContext, SparkConf

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


from pyspark.sql import SQLContext
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import regexp_replace, col

#from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
#from pyspark.mllib.regression import LabeledPoint
#from pyspark.mllib.util import MLUtils



conf = SparkConf().setAppName("FirstSpark").setMaster("local")
conf.set("spark.driver.allowMultipleContexts", "true")

sc = SparkContext(conf=conf)
sql_sc = SQLContext(sc)

def readAndLoadData():
    dataset_pd = pd.read_csv("Training.csv",encoding="latin2")
#    print(list(dataset_pd.columns.values))
    return dataset_pd

def pandasClean(dataset_pd):    
    dataset_pd = dataset_pd.drop(["User_ID","Link","Post_Type","Hints from annotator"],axis=1)
    dataset_pd = dataset_pd.dropna(axis=0, how="any")
    dataset_pd['Drinking_Label'] = dataset_pd.Drinking_Label.str.lower().str.strip()
    return dataset_pd
    
def pandasToSpark(dataset_pd):
    dataset_df = sql_sc.createDataFrame(dataset_pd)
    return dataset_df
    
def LDAModelApply(dataset):
    lda = LDA(k=10, maxIter=10)
    model = lda.fit(dataset)
    print(model.isDistributed())
    print("Vocab Size", model.vocabSize())
    ll = model.logLikelihood(dataset)
    lp = model.logPerplexity(dataset)
    print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
    print("The upper bound bound on perplexity: " + str(lp))
    print(model.topicsMatrix())
    # Describe topics.
    topics = model.describeTopics(10)
    
    print(topics.printSchema())
    print(model.explainParams())
#    topics.rdd.saveAsTextFile("LDATopics.txt")
    print("The topics described by their top-weighted terms:")
    topics.show(truncate=False)
#    model.save("MyLDAModel")
    topics.describe()    
    # Shows the result
    print("Showing Result")
    transformed = model.transform(dataset)
    transformed.show(truncate=False)    
    
    return model
        
def removePunctuation(text):
    temp = " ".join(str(x).lower() for x in text)
    return re.sub('[^a-z| |0-9]', '', temp)

def createVector(data):
    tokenizer = Tokenizer(inputCol="Message", outputCol="words")
    wordsData = tokenizer.transform(data)
    print(wordsData.describe())
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
    featurizedData = hashingTF.transform(wordsData)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)
    for features_label in rescaledData.select("Message","features","rawFeatures", "label").take(3):
        print(features_label)
    return rescaledData

def stringIndexerApply(data):
    indexer = StringIndexer(inputCol="Drinking_Label", outputCol="Drinking_Label_Index")
    indexed = indexer.fit(data).transform(data)
    return indexed
    
def dfToRDD(data):
    assembler = VectorAssembler(inputCols=["features"],outputCol="features")
    output = assembler.transform(data)


def randomForestApply(data):
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
    labelIndexer.transform(data).select("indexedLabel", "label").rdd.map(lambda x: ",".join(map(str, x))).coalesce(1).saveAsTextFile("rfModel_col")
    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
    
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])
    testData.select("label").rdd.map(lambda x: ",".join(map(str, x))).coalesce(1).saveAsTextFile("rfModel_")

    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=20)
    
    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf])
    
    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)
    
    # Make predictions.
    predictions = model.transform(testData)
    print(predictions.columns)
    predictions.select("probability","prediction", "indexedLabel").rdd.map(lambda x: ",".join(map(str, x))).coalesce(1).saveAsTextFile("rfModel")

    # Select example rows to display.
    predictions.select("prediction", "indexedLabel", "features").show(20)
    
    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Accuracy = %g" % (accuracy*100))
    print("Test Error = %g" % (1.0 - accuracy))
    
    rfModel = model.stages[2]
    print(rfModel)  # summary only
        
def main():
    dataset_pd = readAndLoadData()
    dataset_pd = pandasClean(dataset_pd)            
    print(dataset_pd.columns)
    dataset_df = sql_sc.createDataFrame(dataset_pd)
    dataset_df = stringIndexerApply(dataset_df)
    message = dataset_df.select(regexp_replace(col("Message"), "\p{Punct}", "").alias("Message"), ("Drinking_Label"))
    
    
#    Add a id column before
    message = message.withColumn("label", monotonically_increasing_id())
#    Crearing vector from document
    rescaledData = createVector(message)
    print(rescaledData.columns)
#    rescaled_indexed = stringIndexerApply(rescaledData)
    print(rescaledData.show(5))
    forLR = rescaledData.select(col("Drinking_Label").alias("label"),"features")
    randomForestApply(forLR)
#    (rfOutput))
#.map(lambda row: LabeledPoint(row.label,row.features))
#    forRdd = rescaled_indexed.select(col("Drinking_Label_Index").alias("label"),"features").rdd.map(lambda row: LabeledPoint(row.getAs[Double]("label"),row.getAs[org.apache.spark.mllib.linalg.Vector]("features")))
    

#    model = LDAModelApply(rescaledData.select("label","features"))
    
#   Data Preparation for Logistic Regression
  
if __name__ == "__main__":
    main()