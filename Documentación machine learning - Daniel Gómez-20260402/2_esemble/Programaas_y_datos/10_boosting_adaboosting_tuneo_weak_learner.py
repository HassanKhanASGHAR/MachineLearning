results=[]
names=[]

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# define dataset
svc=SVC(probability=True, kernel='linear')
lr=LogisticRegression()
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=6)
models=[svc,lr]
# define the model
for i in models:
  model = AdaBoostClassifier(base_estimator=i)
  # evaluate the model
  cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=1)
  n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
  # report performance
  results.append(n_scores)
  names.append(i)
  print('Accuracy for {} : %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

