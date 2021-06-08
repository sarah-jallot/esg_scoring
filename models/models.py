class Model():
    def __init__(self, X, y, custom_clusters, params, labels="mkmean_labels"):
        self.X = X
        self.y = y
        self.labels = labels
        self.custom_clusters = [0, 1, 2]
        self.params = params

    def train_test_split(self, test_size=0.4):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                                random_state=0)

    def train_model(self):
        self.models = []
        self.X_train = self.X_train.reset_index().drop(columns="index")
        self.y_train = self.y_train.reset_index().drop(columns="index")
        model = RandomForestClassifier(**self.params)
        X_train1 = self.X_train[self.X_train[labels].isin(custom_clusters) == False].copy()
        y_train1 = self.y_train.iloc[X_train1.index, :].copy()
        model.fit(X_train1, y_train1)
        self.models.append(model)
        for cluster in custom_clusters:
            model = RandomForestClassifier(**params)
            mask = (self.X_train[labels] == cluster)
            X_train2 = self.X_train[mask].copy()
            y_train2 = self.y_train.iloc[self.X_train[mask].index, :].copy()
            model.fit(X_train2, y_train2)
            self.models.append(model)

        classes = list(self.models[0].classes_)
        for model in self.models:
            classes.extend(list(model.classes_))
            self.classes_ = sorted(list(set(classes)))

    def predict(self):
        self.X_test = self.X_test.reset_index().drop(columns="index")
        self.y_test = self.y_test.reset_index().drop(columns="index")
        y_preds = pd.DataFrame(index=self.X_test.index, columns=["predictions"])
        X_test1 = self.X_test[self.X_test[labels].isin(custom_clusters) == False].copy()
        y_pred1 = self.models[0].predict(X_test1)[:, np.newaxis]
        y_preds.iloc[X_test1.index, :] = y_pred1.copy()

        for cluster in self.custom_clusters:
            index = self.custom_clusters.index(cluster)
            mask = (self.X_test[labels] == cluster)
            X_test2 = self.X_test[mask].copy()
            y_pred2 = self.models[index].predict(X_test2)[:, np.newaxis]
            y_preds.iloc[X_test2.index, :] = y_pred2

        self.y_preds = y_preds["predictions"]
        self.y_test = self.y_test['ESG Score Grade']
        self.accuracy = (self.y_test == self.y_preds).sum().sum() / len(self.y_test)