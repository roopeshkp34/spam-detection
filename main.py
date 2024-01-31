from typing import Any, Dict, List
import pandas as pd

from sklearn.model_selection import train_test_split
#CounterVectorizer Convert the text into matrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score

def spam_or_ham(mail: List[str]) -> Dict[str, Any]:
    data=pd.read_csv('spam.csv')
    data['Spam']=data['Category'].apply(lambda x:1 if x=='spam' else 0)
    
    # Train test data
    X_train,X_test,y_train,y_test=train_test_split(data.Message,data.Spam,test_size=0.25)

    # model
    clf=Pipeline([
        ('vectorizer',CountVectorizer()),
        ('nb',MultinomialNB())
    ])
    
    # Train the model
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)

    prediction = clf.predict_proba(mail)

    predicted_label = prediction.argmax()
    if predicted_label ==0:
        spam = False
    else:
       spam = True
    return {"spam": spam, "accuracy": prediction[0][predicted_label], "precision": precision_score(y_test, y_pred)}

emails=[
    "Subject: neon retreat ho ho ho , we ' re around to that most wonderful time of the year - - - neon"
]  
print(spam_or_ham(emails))