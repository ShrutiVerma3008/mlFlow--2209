import mlflow
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#loading the data
wine = load_wine()
x= wine.data
y=wine.target

# train test split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=.10,random_state=42)

# defing teh parameters for RF model
max_depth = 10
n_estimators = 15

# mention experiment name
# mlflow.set_experiment('test123')

with mlflow.start_run(experiment_id=169396345193464966):
    rf = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42)
    rf.fit(x_train,y_train)

    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)

    mlflow.log_metric('accurecy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimator',n_estimators)

    #creating the confusion matrics

    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot = True,fmt = 'd',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('predicted')
    plt.title("Confusion matrix")

    #saving th plot
    plt.savefig("test.png")

    #log artifacts using mlflow
    mlflow.log_artifact('test.png')
    mlflow.log_artifact(__file__)

    print(accuracy)