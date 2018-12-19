# Databricks notebook source
new_df=spark.read.parquet('mnt/mnist/FE.parquet')

# COMMAND ----------

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline


# Load training data
#df2=df.select(col('image'),col('label_col').cast('double'))
# Split the data into train and test
# assembler = VectorAssembler(inputCols="image",outputCol="features")
# df = assembler.transform(df)
splits = new_df.randomSplit([0.8, 0.2], 1234)
train = splits[0]
test = splits[1]




# specify layers for the neural network:
# input layer of size 28*28 (features), two intermediate of size 5 and 4
# and output of size 3 (classes)
layers = [28*28, 50, 50, 10]

# create the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(featuresCol="image",labelCol="label", maxIter=100, layers=layers, blockSize=128, seed=1234)
pipeline = Pipeline(stages=[trainer])

# train the model
model = pipeline.fit(train)

# compute accuracy on the test set
result = model.transform(test)
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

# COMMAND ----------

##NOTE: service deployment always gets the model from the current working dir.
import os

model_name = "mnist.mml"
model_dbfs = os.path.join("/dbfs", model_name)

model.write().overwrite().save(model_name)
print("saved model to {}".format(model_dbfs))

# COMMAND ----------

test.write.mode('overwrite').parquet('/mnt/mnist/testmnist')