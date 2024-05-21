import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from scipy.stats import ks_2samp
import scipy.stats as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

class Model:
    
    def __init__(self, xgb_params=None, rf_params=None, class_weights=None,
                 train_data_path=None,  features=None, cv=None):
        self.xgb_params = xgb_params 
        self.rf_params = rf_params 
        self.class_weights = class_weights 
        self.train_data_path = train_data_path
        self.features = features
        self.cv = cv

    def fit(self):
        # Load and preprocess training data
        df_train = pd.read_csv(self.train_data_path)
        X_train = df_train.drop(["ID", "86"], axis=1)[self.features]
        y_train = df_train["86"]

        # Create classifiers
        rf_classifier = RandomForestClassifier(**self.rf_params)
        xgb_classifier = xgb.XGBClassifier(scale_pos_weight=self.class_weights[1], **self.xgb_params)
        gaussian_nb = GaussianNB()
        svc_classifier = SVC(probability=True, class_weight=self.class_weights)
        final_estimator = LogisticRegression(class_weight=self.class_weights, random_state=42)
        
        estimators = [('random_forest', rf_classifier),
                      ('xgboost', xgb_classifier),
                      ('GaussianNB', GaussianNB()),
                      ('SVC', svc_classifier)
                     ]
        
        
        
        # Create a StackingClassifier
        classifier = StackingClassifier(estimators=estimators, final_estimator=final_estimator)

        # Perform cross-validation and metrics 
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        y_pred_prob_cv = cross_val_predict(classifier, X_train, y_train, cv=skf, method='predict_proba')[:, 1]
        self.metrics(y_train, y_pred_prob_cv)
        
        # finalize the model with parameters on the entire training data
        classifier.fit(X_train, y_train)
        
        # Save the trained model
        self.classifier = classifier

    def metrics(self, y_test, y_pred_prob):

        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # Calculate Precision-Recall curve and AUC
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        auc_pr = auc(recall, precision)

        # Calculate KS statistic
        ks_statistic, p_value = ks_2samp(y_pred_prob[y_test == 1], y_pred_prob[y_test == 0])

        # Plot metrics
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        axs[0, 0].hist(y_pred_prob[y_test == 1], bins=100, color="blue", edgecolor="black", label="class 1")
        axs[0, 0].hist(y_pred_prob[y_test == 0], bins=100, color="green", alpha=0.5, edgecolor="black", label="class 0")
        axs[0, 0].set_xlabel("Probability", color="white")
        axs[0, 0].set_ylabel("Frequency", color="white")
        axs[0, 0].set_title("Histogram of Predicted Probabilities", color="white")
        axs[0, 0].tick_params(axis="both", colors="white")
        axs[0, 0].legend()

        axs[0, 1].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
        axs[0, 1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        axs[0, 1].set_xlim([0.0, 1.0])
        axs[0, 1].set_ylim([0.0, 1.05])
        axs[0, 1].set_xlabel("False Positive Rate", color="white")
        axs[0, 1].set_ylabel("True Positive Rate", color="white")
        axs[0, 1].set_title("Receiver Operating Characteristic", color="white")
        axs[0, 1].legend(loc="lower right")
        axs[0, 1].tick_params(axis="both", colors="white")

        axs[1, 0].plot(recall, precision, color="darkorange", lw=2, label=f"AUC = {auc_pr:.2f}")
        axs[1, 0].set_xlim([0.0, 1.0])
        axs[1, 0].set_ylim([0.0, 1.05])
        axs[1, 0].set_xlabel("Recall", color="white")
        axs[1, 0].set_ylabel("Precision", color="white")
        axs[1, 0].set_title("Precision-Recall Curve", color="white")
        axs[1, 0].legend(loc="lower left")
        axs[1, 0].tick_params(axis="both", colors="white")

        y_0 = np.sort(y_pred_prob[y_test == 0])
        y_1 = np.sort(y_pred_prob[y_test == 1])
        cdf_y_0 = [st.norm.cdf(x) for x in y_0]
        cdf_y_1 = [st.norm.cdf(x) for x in y_1]

        axs[1, 1].plot(np.arange(0, len(cdf_y_0)) / len(cdf_y_0), cdf_y_0, label="class 0")
        axs[1, 1].plot(np.arange(0, len(cdf_y_1)) / len(cdf_y_1), cdf_y_1, label="class 1")
        axs[1, 1].set_xlabel("% of sample", color="white")
        axs[1, 1].set_ylabel("F(x)", color="white")
        axs[1, 1].set_title(f"KS CDF curve normalize: {round(ks_statistic, 2)} \n p-value = {p_value}", color="white")
        axs[1, 1].legend()
        axs[1, 1].tick_params(axis="both", colors="white")

        plt.tight_layout()
        plt.savefig("metrics.png", dpi=400, bbox_inches="tight", transparent=False)
        plt.show()

    def predict(self, test_data_path, top_selection=1000):
        df_test = pd.read_csv(test_data_path)
        X_test = df_test[self.features]
        y_pred_proba = self.classifier.predict_proba(X_test)[:, 1]
        df_test["probability_prediction"] = y_pred_proba
        df_test = df_test.sort_values(by="probability_prediction", ascending=False)
        selected_id = df_test.iloc[:top_selection].ID.unique()
        df_test["promising"] = 0
        df_test.loc[df_test.ID.isin(selected_id), "promising"] = 1
        return df_test
