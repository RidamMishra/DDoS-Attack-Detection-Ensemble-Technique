import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted label',
           ylabel='True label')
    
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    return ax

def show_confusion_matrix(y_true, y_pred):
    class_names = ['Normal', 'Attack']
    
    plot_confusion_matrix(y_true, y_pred, classes=class_names, normalize=True, cmap=plt.cm.Blues)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.show()

    print('\n')
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    print('\n')

# Load dataset
path = 'sdn_traffic.csv'
df = pd.read_csv(path)
print(df.head())

# Preprocessing
to_drop = ['dt']
df = df.drop(to_drop, axis='columns')
df = df[df['pktrate'] != 0]

df['src'] = [int(i.split('.')[3]) for i in df['src']]
df['dst'] = [int(i.split('.')[3]) for i in df['dst']]
df['switch'] = df['switch'].astype(str)
df['src'] = df['src'].astype(str)
df['dst'] = df['dst'].astype(str)
df['port_no'] = df['port_no'].astype(str)
df['Protocol'] = df['Protocol'].astype(str)

df.fillna(df.mean(), inplace=True)

X = df.drop(['label'], axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Instantiate individual classifiers
gnb = GaussianNB()
gbc = GradientBoostingClassifier()
svc = SVC(gamma='auto', probability=True
lr = LogisticRegression(solver='liblinear', multi_class='ovr')

# Create an ensemble model using VotingClassifier
ensemble = VotingClassifier(estimators=[
    ('gnb', gnb),
    ('gbc', gbc),
    ('svc', svc),
    ('lr', lr)
], voting='soft')

ensemble.fit(X_train, y_train)

# Predict using the ensemble model
y_train_pred = ensemble.predict(X_train)
y_test_pred = ensemble.predict(X_test)

train_score = ensemble.score(X_train, y_train)
test_score = ensemble.score(X_test, y_test)
print(f"Training Score: {train_score * 100:.2f}%")
print(f"Testing Score: {test_score * 100:.2f}%")
show_confusion_matrix(y_test, y_test_pred)
