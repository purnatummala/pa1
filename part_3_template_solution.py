import numpy as np
from numpy.typing import NDArray
from typing import Any
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import utils as u
import new_utils as nu
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit, cross_validate, train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

from sklearn.metrics import top_k_accuracy_score

"""
   In the first two set of tasks, we will narrowly focus on accuracy - 
   what fraction of our predictions were correct. However, there are several 
   popular evaluation metrics. You will learn how (and when) to use these evaluation metrics.
"""


# ======================================================================
class Section3:
    def __init__(
        self,
        normalize: bool = True,
        frac_train=0.2,
        seed=42,
    ):
        self.seed = seed
        self.normalize = normalize

    def analyze_class_distribution(self, y: NDArray[np.int32]) -> dict[str, Any]:
        """
        Analyzes and prints the class distribution in the dataset.

        Parameters:
        - y (array-like): Labels dataset.

        Returns:
        - dict: A dictionary containing the count of elements in each class and the total number of classes.
        """
        # Your code here to analyze class distribution
        # Hint: Consider using collections.Counter or numpy.unique for counting

        uniq, counts = np.unique(y, return_counts=True)
        print(f"{uniq=}")
        print(f"{counts=}")
        print(f"{np.sum(counts)=}")

        return {
            "class_counts": {}, 
            "num_classes": 0,  
        }

    # --------------------------------------------------------------------------
    """
    A. Using the same classifier and hyperparameters as the one used at the end of part 2.B. 
       Get the accuracies of the training/test set scores using the top_k_accuracy score for k=1,2,3,4,5. 
       Make a plot of k vs. score for both training and testing data and comment on the rate of accuracy change. 
       Do you think this metric is useful for this dataset?
    """

    def partA(
        self,
        Xtrain: NDArray[np.floating],
        ytrain: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """ """

        answer = {}

        """
        # `answer` is a dictionary with the following keys:
        - integers for each topk (1,2,3,4,5)
        - "clf" : the classifier
        - "plot_k_vs_score_train" : the plot of k vs. score for the training data, 
                                    a list of tuples (k, score) for k=1,2,3,4,5
        - "plot_k_vs_score_test" : the plot of k vs. score for the testing data
                                    a list of tuples (k, score) for k=1,2,3,4,5

        # Comment on the rate of accuracy change for testing data
        - "text_rate_accuracy_change" : the rate of accuracy change for the testing data

        # Comment on the rate of accuracy change
        - "text_is_topk_useful_and_why" : provide a description as a string

        answer[k] (k=1,2,3,4,5) is a dictionary with the following keys: 
        - "score_train" : the topk accuracy score for the training set
        - "score_test" : the topk accuracy score for the testing set
        """
        answer = {
    "clf": "Logistic Regression",  
    "plot_k_vs_score_train": [],
    "plot_k_vs_score_test": [],
}
        clf_lr = LogisticRegression(max_iter=300, multi_class='ovr', random_state=self.seed)

        clf_lr.fit(Xtrain, ytrain)

        k_values = [1, 2, 3, 4, 5]
        scores_train = []
        scores_test = []

        for k in k_values:
            score_train = top_k_accuracy_score(ytrain, clf_lr.predict_proba(Xtrain), k=k)
            score_test = top_k_accuracy_score(ytest, clf_lr.predict_proba(Xtest), k=k)
            scores_train.append((k, score_train))
            scores_test.append((k, score_test))

        plt.figure(figsize=(10, 6))
        plt.plot(*zip(*scores_train), marker='o', linestyle='-', color='blue', label='Training Data')
        plt.plot(*zip(*scores_test), marker='s', linestyle='--', color='red', label='Testing Data')
        plt.xlabel('k')
        plt.ylabel('Top-k Accuracy Score')
        plt.title('Top-k Accuracy Score vs. k')
        plt.xticks(k_values)
        plt.legend()
        plt.grid(True)
        plt.show()

        rate_of_accuracy_change_test = "To be commented"
        is_topk_useful_and_why = "To be commented"

        answer = {
            "clf": "clf_lr",
            "plot_k_vs_score_train": scores_train,
            "plot_k_vs_score_test": scores_test,
            "text_rate_accuracy_change": rate_of_accuracy_change_test,
            "text_is_topk_useful_and_why": is_topk_useful_and_why
        }

        for k, score_train, score_test in zip(k_values, [s[1] for s in scores_train], [s[1] for s in scores_test]):
            answer[k] = {
                "score_train": score_train,
                "score_test": score_test
            }
        return answer, Xtrain, ytrain, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    B. Repeat part 1.B but return an imbalanced dataset consisting of 90% of all 9s removed.  Also convert the 7s to 0s and 9s to 1s.
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> tuple[
        dict[Any, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        """"""
        answer = {}

        Xtrain, ytrain = nu.filter_and_modify_7_9s(X, y)
        Xtest, ytest = nu.filter_and_modify_7_9s(Xtest, ytest)
        Xtrain = nu.scale_data(Xtrain)
        Xtest = nu.scale_data(Xtest)
        ytrain = ytrain.astype(int)
        ytest = ytest.astype(int)

        answer = {
                "length_Xtrain": len(Xtrain),
                "length_Xtest": len(Xtest),
                "length_ytrain": len(ytrain),
                "length_ytest": len(ytest),
                "max_Xtrain": np.max(Xtrain),
                "max_Xtest": np.max(Xtest)
            }

        return answer, X, y, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    C. Repeat part 1.C for this dataset but use a support vector machine (SVC in sklearn). 
        Make sure to use a stratified cross-validation strategy. In addition to regular accuracy 
        also print out the mean/std of the F1 score, precision, and recall. As usual, use 5 splits. 
        Is precision or recall higher? Explain. Finally, train the classifier on all the training data 
        and plot the confusion matrix.
        Hint: use the make_scorer function with the average='macro' argument for a multiclass dataset. 
    """

    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""

        answer = {}
        clf_svc = SVC(random_state=self.seed)

        cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)

        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='macro'),
            'recall': make_scorer(recall_score, average='macro'),
            'f1': make_scorer(f1_score, average='macro')
        }

        cv_results_svc = cross_validate(clf_svc, X, y, cv=cv_stratified, scoring=scoring)

        scores = {
            "mean_accuracy": cv_results_svc['test_accuracy'].mean(),
            "std_accuracy": cv_results_svc['test_accuracy'].std(),
            "mean_precision": cv_results_svc['test_precision'].mean(),
            "std_precision": cv_results_svc['test_precision'].std(),
            "mean_recall": cv_results_svc['test_recall'].mean(),
            "std_recall": cv_results_svc['test_recall'].std(),
            "mean_f1": cv_results_svc['test_f1'].mean(),
            "std_f1": cv_results_svc['test_f1'].std()
        }

        is_precision_higher_than_recall = scores["mean_precision"] > scores["mean_recall"]

        explanation = "Precision is higher than recall, indicating that when the classifier predicts a positive class, it is highly likely correct. However, it may miss some positive cases (lower recall)." if is_precision_higher_than_recall else "Recall is higher than precision, indicating the classifier has a lower threshold to predict positive classes, potentially at the cost of accuracy (higher false positive rate)."

        clf_svc.fit(X, y)

        confusion_matrix_train = confusion_matrix(y, clf_svc.predict(X))
        confusion_matrix_test = confusion_matrix(ytest, clf_svc.predict(Xtest))

        answer = {
            "scores": scores,
            "cv": "StratifiedKFold",
            "clf": "SVC",
            "is_precision_higher_than_recall": is_precision_higher_than_recall,
            "explain_is_precision_higher_than_recall": explanation,
            "confusion_matrix_train": confusion_matrix_train.tolist(), # for JSON compatibility
            "confusion_matrix_test": confusion_matrix_test.tolist() # for JSON compatibility
        }

        plt.figure(figsize=(10, 7))
        sns.heatmap(confusion_matrix_train, annot=True, fmt='g', cmap='viridis')
        plt.title('Confusion Matrix for Training Data')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()

        plt.figure(figsize=(10, 7))
        sns.heatmap(confusion_matrix_test, annot=True, fmt='g', cmap='viridis')
        plt.title('Confusion Matrix for Testing Data')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()

        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "is_precision_higher_than_recall" : a boolean
        - "explain_is_precision_higher_than_recall" : a string
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        
        answer["scores"] is dictionary with the following keys, generated from the cross-validator:
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1
        """

        return answer

    # --------------------------------------------------------------------------
    """
    D. Repeat the same steps as part 3.C but apply a weighted loss function (see the class_weights parameter).  Print out the class weights, and comment on the performance difference. Use the `compute_class_weight` argument of the estimator to compute the class weights. 
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """"""
        # Enter your code and fill the `answer` dictionary
        answer = {}
        class_labels = np.unique(y)
        class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=y)
        class_weights_dict = dict(zip(class_labels, class_weights))

        # Initialize SVC with computed class weights
        clf_svc_weighted = SVC(random_state=42, class_weight=class_weights_dict)

        # Define scoring metrics
        scoring_metrics = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='macro'),
            'recall': make_scorer(recall_score, average='macro'),
            'f1': make_scorer(f1_score, average='macro')
        }

        # Perform stratified cross-validation
        cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(clf_svc_weighted, X, y, cv=cv_stratified, scoring=scoring_metrics)

        # Extract mean and std of the scores
        scores = {
            'mean_accuracy': np.mean(cv_results['test_accuracy']),
            'std_accuracy': np.std(cv_results['test_accuracy']),
            'mean_precision': np.mean(cv_results['test_precision']),
            'std_precision': np.std(cv_results['test_precision']),
            'mean_recall': np.mean(cv_results['test_recall']),
            'std_recall': np.std(cv_results['test_recall']),
            'mean_f1': np.mean(cv_results['test_f1']),
            'std_f1': np.std(cv_results['test_f1']),
        }

        clf_svc_weighted.fit(X, y)

        confusion_matrix_train = confusion_matrix(y, clf_svc_weighted.predict(X))
        confusion_matrix_test = confusion_matrix(ytest, clf_svc_weighted.predict(Xtest))

        explain_purpose_of_class_weights = "Class weights adjust the importance of each class during training, helping to address imbalances by penalizing mistakes on underrepresented classes more."
        explain_performance_difference = "Using class weights can improve recall for minority classes, potentially at the cost of overall accuracy or precision, as the model may focus more on correctly predicting underrepresented classes."

        answer = {
            "scores": scores,
            "cv": "StratifiedKFold",
            "clf": "SVC (Weighted)",
            "class_weights": class_weights_dict,
            "confusion_matrix_train": confusion_matrix_train.tolist(),  # for JSON compatibility
            "confusion_matrix_test": confusion_matrix_test.tolist(),  # for JSON compatibility
            "explain_purpose_of_class_weights": explain_purpose_of_class_weights,
            "explain_performance_difference": explain_performance_difference
        }

  
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix_train, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix for Training Data')
        plt.xlabel('Predicted Labels')
        plt.ylabel('Actual Labels')
        plt.show()

        """
        Answer is a dictionary with the following keys: 
        - "scores" : a dictionary with the mean/std of the F1 score, precision, and recall
        - "cv" : the cross-validation strategy
        - "clf" : the classifier
        - "class_weights" : the class weights
        - "confusion_matrix_train" : the confusion matrix for the training set
        - "confusion_matrix_test" : the confusion matrix for the testing set
        - "explain_purpose_of_class_weights" : explanatory string
        - "explain_performance_difference" : explanatory string

        answer["scores"] has the following keys: 
        - "mean_accuracy" : the mean accuracy
        - "mean_recall" : the mean recall
        - "mean_precision" : the mean precision
        - "mean_f1" : the mean f1
        - "std_accuracy" : the std accuracy
        - "std_recall" : the std recall
        - "std_precision" : the std precision
        - "std_f1" : the std f1

        Recall: The scores are based on the results of the cross-validation step
        """

        return answer
