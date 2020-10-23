    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    dataset = pd.read_csv('Dataset.tsv', delimiter = '\t',quoting = 3)
    
    import re
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    corpus = []
    for i in range(0, dataset.shape[0]):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = None)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:,1:].values
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)

    
    y_pred = classifier.predict(X_test)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)