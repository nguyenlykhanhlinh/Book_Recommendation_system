from flask import Flask, render_template, request
from pyspark.ml.evaluation import RegressionEvaluator
import pyspark as ps
from pyspark.sql import SQLContext
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col

app = Flask(__name__)

def recommend(user_id):
    # Create SparkSession
    spark = ps.sql.SparkSession.builder \
        .appName("Book_recommendation") \
        .getOrCreate()

    # Read ratings and books data
    ratings_df = spark.read.csv('C:/Users/MyPC/Documents/PYTHON/Big_data/Big_data/goodbooks-10k/ratings.csv', header=True, inferSchema=True)
    books_df = spark.read.csv('C:/Users/MyPC/Documents/PYTHON/Big_data/Big_data/goodbooks-10k/books.csv', header=True, inferSchema=True)

    # Split data into training and validation sets
    training_df, validation_df = ratings_df.randomSplit([0.8, 0.2])

    # Set model parameters
    iterations = 10
    regularization_parameter = 0.1
    rank = 4

    # Train the ALS model
    als = ALS(
        maxIter=iterations,
        regParam=regularization_parameter,
        rank=rank,
        userCol="user_id",
        itemCol="book_id",
        ratingCol="rating"
    )
    model = als.fit(training_df)

    # Make predictions for the specified user
    user_predictions = model.transform(validation_df.filter(validation_df.user_id == user_id))
    user_predictions = user_predictions.filter(col('prediction').isNotNull())

    # Get recommended books for the user
    recommended_books = user_predictions.join(books_df, user_predictions.book_id == books_df.book_id, "inner") \
        .select(books_df.title, books_df.author, books_df.image_url)

    return recommended_books.limit(10).collect()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_id = request.form['user_id']
        predictions = recommend(user_id)
        return render_template('recommend.html', recommendations=predictions)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


