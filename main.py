from argparse import ArgumentParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from process_data import process_strings
from sklearn.svm import SVC, LinearSVC

# read data sets
train = pd.read_csv("train.csv", dtype={"score": np.int32, "text": str})
test = pd.read_csv("test.csv", dtype={"score": np.int32, "text": str})
evaluation = pd.read_csv("evaluation.csv", dtype={"score": np.int32, "text": str})

# choose type of data processing
parser = ArgumentParser("Demonstration of SVM in training, predicting, and evaluating data sets.")
parser.add_argument("-v", "--verbose", action="store_true", help="Print additional logs.")
parser.add_argument("-c", "--crossvalidate", action="store_true", help="Perform cross validation scoring and show plot.")
parser.add_argument("-r", "--regsweep", action="store_true", help="Perform 4-fold CV and regularization parameter sweep")
parser.add_argument("-p", "--preprocessor", default="sklearn",
                    help="Type of preprocessor to be used. Options are sklearn' or 'own'.")
args = parser.parse_args()
algorithm = args.preprocessor

# combine training and test data set
# to have more data for cross validations
train = train.append(test)

# preprocess data
if algorithm == "own":
    print("Preprocessing data with own algorithm...")
    train["text"] = process_strings(train)
    evaluation["text"] = process_strings(evaluation)
    vec = TfidfVectorizer()
else:
    print("Preprocessing data with SKlearn...")
    train["text"] = train["text"].str.strip().str.lower()
    evaluation["text"] = evaluation["text"].str.strip().str.lower()
    vec = TfidfVectorizer(stop_words='english')

# transform data
print("Transforming data...")
x_train = vec.fit_transform(train["text"]).toarray()
x_eval = vec.transform(evaluation["text"]).toarray()


# k-Fold cross validation
if args.crossvalidate:
    print("Cross validating data...")
    model = LinearSVC(C=1, verbose=args.verbose)
    results = list()
    repeats = range(2, 6)
    for r in repeats:
        results.append(cross_val_score(model, x_train, train["score"].values, cv=r))
    plt.boxplot(results, labels=[str(r) for r in repeats], showmeans=True)
    plt.xlabel("Number of bins")
    plt.ylabel("Model accuracy")
    plt.title("Box plot of accuracy with different amount of bins")
    plt.show()
    exit(0)

if args.regsweep:
    print("Performing regularization param sweep...")
    results = list()
    repeats = [0.1,0.5,1,1.5,2]
    for r in repeats:
        model = LinearSVC(C=r, verbose=args.verbose)
        results.append(cross_val_score(model, x_train, train["score"].values, cv=4))
    plt.boxplot(results, labels=[str(r) for r in repeats], showmeans=True)
    plt.xlabel("C")
    plt.ylabel("Model accuracy")
    plt.title("Box plot of accuracy with different amounts of regularization")
    plt.show()
    exit(0)

#define best model
model = LinearSVC(C=0.5, verbose=args.verbose)

# train model on selected data bins
n_bins = 4
scores = cross_val_score(model, x_train, train["score"].values, cv=n_bins)
idx = np.argmax(scores)
size = int(len(x_train) / n_bins)
mask = np.ones(len(x_train), bool)
mask[idx*size:(idx+1)*size] = 0
print("Training model...")
model.fit(x_train[mask], train["score"].values[mask])

# measure accuracies of models
print("Scoring model...")
acc_train = model.score(x_train, train["score"].values)
acc_eval = model.score(x_eval, evaluation["score"].values)

print(f"Accuracy on training data = {round(acc_train, 3)} %")
print(f"Accuracy on evaluation data = {round(acc_eval, 3)} %")

# calculate how many predictions were off
test_data = evaluation
test_vec = x_eval
count0 = np.size(np.where(test_data["score"] == 0))
count1 = np.size(np.where(test_data["score"] == 1))
correct0 = np.size(
    np.where(np.logical_and(model.predict(test_vec) == test_data["score"].values, test_data["score"].values == 0)))
correct1 = np.size(
    np.where(np.logical_and(model.predict(test_vec) == test_data["score"].values, test_data["score"].values == 1)))

# confusion matrix for sentiment
fig, ax = plt.subplots()
plot_data = [[count1 - correct1, correct1], [correct0, count0 - correct0]]
plot = ax.imshow(plot_data, cmap="PuBu")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["0", "1"])
ax.set_yticklabels(["1", "0"])
plt.colorbar(plot)
plt.xlabel("true sentiment")
plt.ylabel("predicted sentiment")
plt.title("Confusion matrix of true to predicted sentiment")
plt.show()
