# Perceptron-Iris-Classifier

This repository contains an implementation of a `perceptron classifier` for the Iris dataset from scratch. The perceptron algorithm is used to classify the Iris flowers into two classes: setosa and versicolor, based on their sepal length and sepal width.

Here are the three steps of the perceptron algorithm summarized briefly:

1. Initialize weights and biases randomly.
2. Iterate through the training data, adjusting weights for misclassifications.
3. Use the trained perceptron to make predictions on new data by calculating the activation and applying the activation function.

## Usage

```python
import numpy as np
import pandas as pd
from perceptron import Perceptron


df = pd.read_csv("iris.data")
first_row = df.columns
df.rename(
    columns={
        first_row[0]: "sepal_length",
        first_row[1]: "sepal_width",
        first_row[2]: "petal_length",
        first_row[3]: "petal_width",
        first_row[4]: "label",
    },
    inplace=True,
)

first_row_tmp = []
for i in range(len(first_row[:-1])):
    first_row_tmp.append(float(first_row[i]))
first_row_tmp.append(first_row[-1])

df.loc[-1] = first_row_tmp
df.index = df.index + 1
df.sort_index(inplace=True)
df = df[(df["label"] == "Iris-setosa") | (df["label"] == "Iris-versicolor")]

p = Perceptron(df[["sepal_length", "sepal_width"]], df["label"])
p.fit(lr=1, iteration=1000)


# Generating a Plot Line Using Plotly
import plotly.express as px

fig = px.scatter(
    df, x="sepal_length", y="sepal_width", color="label", width=750, height=500
)
x = np.linspace(df["sepal_length"].min(), df["sepal_length"].max(), 100)
y = -p.w[1] * x / p.w[2] - p.w[0] / p.w[2]
fig.add_scatter(x=x, y=y)
fig.show()
