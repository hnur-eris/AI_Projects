import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import joblib

class HiringModel:
    def __init__(self):
        self.__df = self.get_data()
        self.x_train, self.x_test, self.y_train, self.y_test = self.train_test_values()    
        self.standardize_feature()
        self.choose_kernel()
        self.fit_model(self.selected_kernel)
        self.evaluate_model()
        self.visualize_decision()
        self.save_model()

    def get_data(self, samples=200):
        data = []
        for _ in range(samples):
            experience_years = round(random.uniform(0, 10), 2)
            technical_score = round(random.uniform(0, 100), 2)
            
            if experience_years < 2 and technical_score < 60:
                label = 1
            else:
                label = 0
            data.append([experience_years, technical_score, label])
        return pd.DataFrame(data, columns=['experienced_years', 'technical_score', 'label'])

    def train_test_values(self):
        x = self.__df[['experienced_years', 'technical_score']]
        y = self.__df['label']
        
        return train_test_split(x, y, test_size=0.2, random_state=42)

    def standardize_feature(self):
        self.scaler = StandardScaler()
        self.x_train_scaled = self.scaler.fit_transform(self.x_train)
        self.x_test_scaled = self.scaler.transform(self.x_test)
    
    
    def choose_kernel(self):
        valid_kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        
        print("Available kernels: ", ", ".join(valid_kernels))
        self.selected_kernel = input("Please select a kernel type: ").strip().lower()

        if self.selected_kernel not in valid_kernels:
            print("Invalid kernel. Defaulting to 'linear'.")
            self.selected_kernel = "linear"

    def fit_model(self, kernel='linear'):
        self.svm_model = SVC(kernel=kernel, C=1.0, random_state=42)
        self.svm_model.fit(self.x_train_scaled, self.y_train)

        
    def visualize_decision(self):
        h = 0.01
        x_min, x_max = self.x_train_scaled[:, 0].min() - 1, self.x_train_scaled[:, 0].max() + 1
        y_min, y_max = self.x_train_scaled[:, 1].min() - 1, self.x_train_scaled[:, 1].max() + 1
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))

        z = self.svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        plt.contourf(xx, yy, z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(self.x_train_scaled[:, 0], 
                    self.x_train_scaled[:, 1], 
                    c=self.y_train, 
                    cmap=plt.cm.coolwarm, 
                    edgecolors='k'
                )
        plt.xlabel('Experience')
        plt.ylabel('Technical Score')
        plt.title('SVM Decision Boundary')
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        
    def predict_user_input(self):
        experience = float(input("Enter years of experience (0-10):  "))
        technical_score = float(input("Enter your technical score (0-100):  "))
        try :
            if not (0 <= experience <= 10):
                raise ValueError("Experience must be between 0 and 10")
            if not (0 <= technical_score <= 100):
                raise ValueError("Techncal score must be between 0 and 100")
            
            input_data = pd.DataFrame([[experience, technical_score]], columns=['experienced_years', 'technical_score'])
            input_scaled = self.scaler.transform(input_data)
            prediction = self.svm_model.predict(input_scaled)
            
            print("\nPrediction:", "Hired" if prediction[0] == 0 else "Not Hired")
        except Exception as e:
            print(f"Error : {e}")
    
    def evaluate_model(self):
        y_pred = self.svm_model.predict(self.x_test_scaled)
        print("Accuracy:", accuracy_score(self.y_test, y_pred), "Success" if accuracy_score(self.y_test, y_pred) >= 0.5 else "Failed")
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
        print("Classification Report:\n", classification_report(self.y_test, y_pred))


    def grid_search(self):
        print("\nLooking the best values to our model...")
        
        param_grid = {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': [0.1, 1, 10],
            'gamma': [0.01, 0.1, 1]}

        grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.x_train_scaled, self.y_train)

        print(f"\nBest Parameters: {grid_search.best_params_}")
        print(f"Best Score: {grid_search.best_score_:.4f}\n\n")
        self.svm_model = grid_search.best_estimator_

    def save_model(self):
        joblib.dump(self.svm_model, './pkl_files/model.pkl')
        joblib.dump(self.scaler, './pkl_files/scaler.pkl')