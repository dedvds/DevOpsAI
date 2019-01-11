# Databricks notebook source
# MAGIC %md # Machine Learning with Spark MLlib
# MAGIC Spark MLlib, sometimes known as Spark ML, is a library for building machine learning solutions on Spark.
# MAGIC 
# MAGIC ## Data Preparation and Exploration
# MAGIC Machine learning begins with data preparation and exploration. We'll start by loading a dataframe of data about flights between airports in the US.

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *

flightSchema = StructType([
  StructField("DayofMonth", IntegerType(), False),
  StructField("DayOfWeek", IntegerType(), False),
  StructField("Carrier", StringType(), False),
  StructField("OriginAirportID", StringType(), False),
  StructField("DestAirportID", StringType(), False),
  StructField("DepDelay", IntegerType(), False),
  StructField("ArrDelay", IntegerType(), False),
])
spark.conf.set(
  "fs.azure.account.key.<blobname>.blob.core.windows.net",
  "<Key Name>")
flights = spark.read.csv('wasbs://<containername>@<blobname>.blob.core.windows.net/raw-flight-data.csv', schema=flightSchema, header=True)
flights.show()

# COMMAND ----------

# MAGIC %md The data includes a record of each flight, including how late it departed and arrived. Let's see how many rows are in the data set:

# COMMAND ----------

flights.count()

# COMMAND ----------

# MAGIC %md ### Data Cleansing
# MAGIC Generally, before you can use data to train a machine learning model, you need to do some pre-processing to clean the data so it's ready for use. For example, does our data include some duplicate rows?

# COMMAND ----------

flights.count() - flights.dropDuplicates().count()

newdf=flight
newdf2=newdf

# COMMAND ----------

# MAGIC %md Yes it does.
# MAGIC 
# MAGIC Does it have any missing values in the **ArrDelay** and **DepDelay** columns?

# COMMAND ----------

flights.count() - flights.dropDuplicates().dropna(how="any", subset=["ArrDelay", "DepDelay"]).count()

# COMMAND ----------

# MAGIC %md Yes.
# MAGIC 
# MAGIC So let's clean the data by removing the duplicates and replacing the missing values with 0.

# COMMAND ----------

flights=flights.dropDuplicates().fillna(value=0, subset=["ArrDelay", "DepDelay"])
flights.count()

# COMMAND ----------

# MAGIC %md ### Exploring the Data
# MAGIC The data includes details of departure and arrival delays. However, we want to simply classify flights as *late* or *not late* based on a rule that defines a flight as *late* if it arrives more than 25 minutes after its scheduled arrival time. We'll select the columns we need, and create a new one that indicates whether a flight was late or not with a **1** or a **0**.

# COMMAND ----------

flights = flights.select("DayofMonth", "DayOfWeek", "Carrier", "OriginAirportID","DestAirportID",
                         "DepDelay", "ArrDelay", ((col("ArrDelay") > 25).cast("Int").alias("Late")))
flights.show()

# COMMAND ----------

# MAGIC %md OK, let's examine this data in more detail. The machine learning algorithms we are going to use are based on statistics; so let's look at some fundamental statistics for our flight data.

# COMMAND ----------

flights.describe().show()

# COMMAND ----------

# MAGIC %md The *DayofMonth* must be a value between 1 and 31, and the mean is around halfway between these values; which seems about right. The same is true for the *DayofWeek* which is a value between 1 and 7. *Carrier* is a string, so there are no numeric statistics; and we can ignore the statistics for the airport IDs - they're just unique identifiers for the airports, not actually numeric values. The departure and arrival delays range between 63 or 94 minutes ahead of schedule, and over 1,800 minutes behind schedule. The means are much closer to zero than this, and the standard deviation is quite large; so there's quite a bit of variance in the delays. The *Late* indicator is a 1 or a 0, but the mean is very close to 0; which implies that there significantly fewer late flights and non-late flights.
# MAGIC 
# MAGIC Let's verify that assumption by creating a table and using a SQL statement to count the number of late and non-late flights:

# COMMAND ----------

flights.createOrReplaceTempView("flightData")
spark.sql("SELECT Late, COUNT(*) AS Count FROM flightData GROUP BY Late").show()

# COMMAND ----------

# MAGIC %md Yes, it looks like there are more non-late flights than late ones - we can see this more clearly with a visualization. To use the notebooks's native visualization tools, we'll need to use an embedded SQL query to retreve a sample of the data:

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM flightData

# COMMAND ----------

# MAGIC %md The results of the query are shown in a table above, but you can also view the data returned as a **Bar** chart, showing the count of the ***&lt;id&gt;*** value by the ***Late*** key. This should confirm that there are significantly more on-time flights than late ones in the sample of 1000 records returned by the query.
# MAGIC 
# MAGIC While we're at it, we can also view histograms and box plots of the delays. Change the plot options to show a **Histogram** of **DepDelay** and confirm that most of the delays are within 100 minutes or so (either way) of 0, but there are a few extremely high delays. These are outliers. You can see these even more clearly if you change the plot type to a **Box Plot** in which the median value is shown as a line inside a box that represents the second and third quartiles of the delay values. The extreme outliers are shown as markers beyond the *whiskers* that indicate the first and fourth quartiles.
# MAGIC 
# MAGIC So we have two problems: our data is *unbalanced* with more negative classes than positive ones, and the outlier values make the distribution of the data extremely *skewed*. Both of these issues are likely to affect any machine learning model we create from it as the most common class and extreme delay values might dominate the training of the model. We'll address this by removing the outliers and *undersampling* the dominant class - in this case non-late flights.

# COMMAND ----------

# Remove flights with outlier delays
# flights = flights.filter("DepDelay < 150 AND ArrDelay < 150")

# # Undersample the most commonly occurring Late class
# pos = flights.filter("Late = 1")
# neg = flights.filter("Late = 0")
# posCount = pos.count()
# negCount = neg.count()
# print(float(posCount))
# print(float(negCount))
# if posCount > negCount:
#   pos = pos.sample(True, float(negCount/(negCount + posCount)),seed=123)
# else:
#   neg = neg.sample(True, float(posCount/(negCount + posCount)),seed=123)
# flights = neg.union(pos).orderBy(rand()) # randomize order of unioned data so we can visualize a mixed sample in the notebook
# flights.createOrReplaceTempView("flightData")
# flights.describe().show()

# COMMAND ----------

# MAGIC %md Our statistics look a little better now, and we still have a lot of data. Let's take a look at that visually.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM flightData

# COMMAND ----------

# MAGIC %md View histograms and box plots of the delays, and a bar chart of the *Late* classes as you did previously to see a more even distribution (though the delays are still skewed and far from *normal*).
# MAGIC 
# MAGIC You can also start to explore relationships in the data. For example, group the box plots of arrival delay by day or carrier to see if lateness varies by these factors. A box plot of **DepDelay** grouped by the **Late** indicator should show that on-time flights have a very low median departure delay and small variance compared to late flights.
# MAGIC 
# MAGIC Finally, to get a clearer picture of the relationship between **DepDelay** and **ArrDelay**, plot both of these fields as a scatter plot - you should see a linear relationship between these two - the later a flight departs, the later it tends to arrive!
# MAGIC 
# MAGIC We can use statistics to quantify this correlation:

# COMMAND ----------

flights.corr("DepDelay", "ArrDelay")

# COMMAND ----------

# MAGIC %md A correlation is a value between -1 and 1. A value close to 1 indicates a *positive* correlation - in other words, increases in one value tend to correlate with increases in the other.

# COMMAND ----------

# MAGIC %md ## Training a Machine Learning Model
# MAGIC OK, now we're ready to build a machine learning model.
# MAGIC First, we'll split the data randomly into two sets for training and testing the model:

# COMMAND ----------

# Split the data for training and testing
splits = flights.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("Late", "trueLabel")
print("Training:", train.count(), ". Test:", test.count())


# COMMAND ----------

display(flights)

# COMMAND ----------

# MAGIC %md ### Define the Pipeline and Train the Model
# MAGIC Now we'll define a pipeline of steps that prepares the *features* in our data, and then trains a model to predict our **Late** *label* from the features.
# MAGIC 
# MAGIC A pipeline encapsulates the transformations we need to make to the data to prepare features for modeling, and then fits the features to a machine learning algorithm to create a model. In this case, the pipeline:
# MAGIC - Creates indexes for all of the categorical columns in our data. These are columns that represent categories, not numeric values.
# MAGIC - Normalizes numeric columsn so they're on a similar scale - this prevents large numeric values from dominating the training. In this case, we only have one numeric value (**DepDelay**), so this step isn't strictly necessary - but it's included to show how its done.
# MAGIC - Assembles all of the categorical indexes and the vector of normalized numeric values into a single vector of features.
# MAGIC - Fits the features to a logistic regression algorithm to create a model.
# MAGIC 
# MAGIC Using a pipeline makes it easier to use the trained model with new data by encapsulating all of the feature preparation steps and ensuring numeric features used to generate predictions from the model are scaled using the same distribution statistics as the training data.

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline

# Create indexes for the categorical features
monthdayIndexer = StringIndexer(inputCol="DayofMonth", outputCol="DayofMonthIdx")
weekdayIndexer = StringIndexer(inputCol="DayOfWeek", outputCol="DayOfWeekIdx")
carrierIndexer = StringIndexer(inputCol="Carrier", outputCol="CarrierIdx")
originIndexer = StringIndexer(inputCol="OriginAirportID", outputCol="OriginAirportIdx")
destIndexer = StringIndexer(inputCol="DestAirportID", outputCol="DestAirportIdx")

# Normalize numeric features
numVect = VectorAssembler(inputCols = ["DepDelay"], outputCol="numFeatures")
minMax = MinMaxScaler(inputCol = numVect.getOutputCol(), outputCol="normFeatures")

# Assemble a vector of features (exclude ArrDelay as we won't have this when predicting new flights)
assembler = VectorAssembler(inputCols = ["DayofMonthIdx", "DayOfWeekIdx", "CarrierIdx",
                                         "OriginAirportIdx","DestAirportIdx","normFeatures"],
                            outputCol="features")

# Train a logistic regression classification model using the pipeline
lr = LogisticRegression(labelCol="Late",featuresCol="features",maxIter=10,regParam=0.3)

pipeline = Pipeline(stages=[monthdayIndexer, weekdayIndexer, carrierIndexer,originIndexer,destIndexer, numVect, minMax, assembler, lr])
model = pipeline.fit(train)
print(model)

# COMMAND ----------

# MAGIC %md ### Test the Model
# MAGIC Now we're ready to apply the model to the test data.

# COMMAND ----------

prediction = model.transform(test)
predicted = prediction.select("features", "rawPrediction", "probability", col("prediction").cast("Int"), "trueLabel")
predicted.show(100, truncate=False)

# COMMAND ----------

# MAGIC %md ### Compute Confusion Matrix Metrics
# MAGIC Classifiers are typically evaluated by creating a *confusion matrix*, which indicates the number of:
# MAGIC - True Positives
# MAGIC - True Negatives
# MAGIC - False Positives
# MAGIC - False Negatives
# MAGIC 
# MAGIC From these core measures, other evaluation metrics such as *accuracy*, *precision* and *recall* can be calculated.

# COMMAND ----------

tp = float(predicted.filter("prediction == 1 AND truelabel == 1").count())
fp = float(predicted.filter("prediction == 1 AND truelabel == 0").count())
tn = float(predicted.filter("prediction == 0 AND truelabel == 0").count())
fn = float(predicted.filter("prediction == 0 AND truelabel == 1").count())
metrics = spark.createDataFrame([
 ("TP", tp),
 ("FP", fp),
 ("TN", tn),
 ("FN", fn),
 ("Accuracy", (tp + tn)/(tp + fp + tn + fn)),
 ("Precision", tp / (tp + fp)),
 ("Recall", tp / (tp + fn))],["metric", "value"])
metrics.show()

# COMMAND ----------

# MAGIC %md ### Review the Area Under ROC
# MAGIC Another way to assess the performance of a classification model is to measure the area under a *received operator characteristic (ROC) curve* for the model. The **spark.ml** library includes a **BinaryClassificationEvaluator** class that you can use to compute this. A ROC curve plots the True Positive and False Positive rates for varying *threshold* values (the probability value over which a class label is predicted). The area under this curve gives an overall indication of the models accuracy as a value between 0 and 1. A value under 0.5 means that a binary classification model (which predicts one of two possible labels) is no better at predicting the right class than a random 50/50 guess.

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol="trueLabel",
                                          rawPredictionCol="rawPrediction",
                                          metricName="areaUnderROC")
auc = evaluator.evaluate(prediction)
print ("AUC = ", auc)

# COMMAND ----------

##NOTE: by default the model is saved to and loaded from /dbfs/ instead of cwd!
import os
model_name = "delayedflight.mml"
model_dbfs = os.path.join("/dbfs", model_name)

model.write().overwrite().save(model_name)
print("saved model to {}".format(model_dbfs))

# COMMAND ----------

test.write.mode('overwrite').parquet('testsetdelay')

# COMMAND ----------

# MAGIC %fs ls

# COMMAND ----------

