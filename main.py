import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class App:
    def __init__(self):
        self.dataset_name=None
        self.classifier_name=None
        self.params=dict()
        self.clf=None
        self.X, self.y=None,None
        self.Init_Streamlit_Page()


    def run(self):
        self.get_dataset()
        self.add_parameter_ui()
        self.generate()
    def Init_Streamlit_Page(self):
        st.title("ML methods with Streamlit")
        st.write("""
        # Explore different classifier and datasets
        Which one is the best?
        """)
        self.dataset_name=st.sidebar.selectbox('Select Dataset',('Iris','Breast Cancer','Wine'))
        st.write(f'## {self.dataset_name} Dataset')
        self.classifier_name=st.sidebar.selectbox('Select Classifier',('KNN','SVM','Random Forest'))

    def get_dataset(self):
        if self.dataset_name=='Iris':
            data=datasets.load_iris()
        elif self.dataset_name=='Breast Cancer':
            data=datasets.load_breast_cancer()
        else:
            data=datasets.load_wine()
        self.X=data.data
        self.y=data.target
        st.write('Shape of dataset:',self.X.shape)
        st.write('number of classes:',len(np.unique(self.y)))


    def add_parameter_ui(self):
        if self.classifier_name == 'SVM':
            C = st.sidebar.slider('C', 0.01, 15.0)
            self.params['C'] = C
        elif self.classifier_name == 'KNN':
            K = st.sidebar.slider('K', 1, 15)
            self.params['K'] = K
        else:
            max_depth = st.sidebar.slider('max_depth', 2, 15)
            self.params['max_depth'] = max_depth
            n_estimators = st.sidebar.slider('n_estimators', 1, 100)
            self.params['n_estimators'] = n_estimators

    def get_classifier(self):
        if self.classifier_name == 'SVM':
            self.clf = SVC(C=self.params['C'])
        elif self.classifier_name == 'KNN':
            self.clf = KNeighborsClassifier(n_neighbors=self.params['K'])
        else:
            self.clf = RandomForestClassifier(n_estimators=self.params['n_estimators'])

    def create_sinusoid(self):
        fig = plt.figure()
        Fs = 8000
        f = 5
        sample = 8000
        x = np.arange(sample)
        y = np.sin(2 * np.pi * f * x / Fs)
        plt.plot(x, y)
        plt.xlabel('sample(n)')
        plt.ylabel('voltage(V)')
        return fig

    def generate(self):
        self.get_classifier()
        #### CLASSIFICATION ####
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=1234)

        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        st.write(f'Classifier = {self.classifier_name}')
        st.write(f'Accuracy =', acc)

        #### PLOT DATASET ####
        # Project the data onto the 2 primary principal components
        pca = PCA(2)
        X_projected = pca.fit_transform(self.X)

        x1 = X_projected[:, 0]
        x2 = X_projected[:, 1]
        fig, ax = plt.subplots()
        scatter = ax.scatter(x1, x2, c=self.y, alpha=0.8, cmap='viridis', edgecolor='k',
                             s=100)
        legend = ax.legend(*scatter.legend_elements(), title="Classes")

        ax.add_artist(legend)
        ax.set_title('PCA Projection of Dataset')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        scatter.set_rasterized(True)
        scatter.set_facecolor('white')

        st.pyplot(fig)

        fig = self.create_sinusoid()
        plt.show()
        st.pyplot(fig)

def main():
    app = App()
    app.run()
if __name__ == "__main__":
    main()





