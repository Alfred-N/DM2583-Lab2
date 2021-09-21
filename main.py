import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from process_data import process_strings
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

# read data sets
train = pd.read_csv("train.csv", dtype={"score": np.int32, "text": str})
test = pd.read_csv("test.csv", dtype={"score": np.int32, "text": str})
evaluation = pd.read_csv("evaluation.csv", dtype={"score": np.int32, "text": str})



# choose type of data processing
algorithm = input("Please choose the type of data processing. Choices are 'sklearn' or 'own'.\n").lower()
if algorithm == "own":
    train["text"] = process_strings(train)
    test["text"] = process_strings(test)
    evaluation["text"] = process_strings(evaluation)
    vec = TfidfVectorizer()
elif algorithm == "sklearn":
    train["text"] = train["text"].str.strip().str.lower()
    test["text"] = test["text"].str.strip().str.lower()
    evaluation["text"] = evaluation["text"].str.strip().str.lower()
    vec = TfidfVectorizer(stop_words='english')
else:
    print("Invalid choice")
    exit(1)



# transform data
print("Transforming data...")
x_train = vec.fit_transform(train["text"]).toarray()
x_test = vec.transform(test["text"]).toarray()
x_eval = vec.transform(evaluation["text"]).toarray()


# train model
print("Training model...")
# param C is the regularization parameter, I guess we could implement cross-validation using train/test to tune it
# and then finally evaluate the model on the (unseen) evaluation data

# you could try SVC too (high-dimensional) but I found it is extremely slow to train
# model = SVC(kernel="rbf", C=1, verbose=True)
model = LinearSVC(C=1, verbose=True)
model.fit(x_train, train["score"].values)

# measure accuracies of models
print("Scoring model...")
acc_train = model.score(x_train, train["score"].values)
acc_test = model.score(x_test, test["score"].values)
acc_eval = model.score(x_eval, evaluation["score"].values)

print(f"Accuracy on training data = {round(acc_train, 3)} %")
print(f"Accuracy on test data = {round(acc_test, 3)} %")
print(f"Accuracy on evaluation data = {round(acc_eval, 3)} %")

# calculate how many predictions were off
test_data = evaluation
test_vec = x_eval
count0 = np.size(np.where(test_data["score"] == 0))
count1 = np.size(np.where(test_data["score"] == 1))
correct0 = np.size(np.where(np.logical_and(model.predict(test_vec) == test_data["score"].values, test_data["score"].values == 0)))
correct1 = np.size(np.where(np.logical_and(model.predict(test_vec) == test_data["score"].values, test_data["score"].values == 1)))

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
