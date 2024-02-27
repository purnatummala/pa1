# Inspired by GPT4

# Information on type hints
# https://peps.python.org/pep-0585/

# GPT on testing functions, mock functions, testing number of calls, and argument values
# https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from typing import Any
from numpy.typing import NDArray

import numpy as np
import utils as u

# Initially empty. Use for reusable functions across
# Sections 1-3 of the homework
import new_utils as nu


# ======================================================================
class Section1:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None

        Notes: notice the argument `seed`. Make sure that any sklearn function that accepts
        `random_state` as an argument is initialized with this seed to allow reproducibility.
        You change the seed ONLY in the section of run_part_1.py, run_part2.py, run_part3.py
        below `if __name__ == "__main__"`
        """
        self.normalize = normalize
        self.frac_train = frac_train
        self.seed = seed

    # ----------------------------------------------------------------------
    """
    A. We will start by ensuring that your python environment is configured correctly and 
       that you have all the required packages installed. For information about setting up 
       Python please consult the following link: https://www.anaconda.com/products/individual. 
       To test that your environment is set up correctly, simply execute `starter_code` in 
       the `utils` module. This is done for you. 
    """

    def partA(self):
        # Return 0 (ran ok) or -1 (did not run ok)
        answer = u.starter_code()
        return answer

    # ----------------------------------------------------------------------
    """
    B. Load and prepare the mnist dataset, i.e., call the prepare_data and filter_out_7_9s 
       functions in utils.py, to obtain a data matrix X consisting of only the digits 7 and 9. Make sure that 
       every element in the data matrix is a floating point number and scaled between 0 and 1 (write
       a function `def scale() in new_utils.py` that returns a bool to achieve this. Checking is not sufficient.) 
       Also check that the labels are integers. Print out the length of the filtered 𝑋 and 𝑦, 
       and the maximum value of 𝑋 for both training and test sets. Use the routines provided in utils.
       When testing your code, I will be using matrices different than the ones you are using to make sure 
       the instructions are followed. 
    """

    def partB(
        self,
    ):
        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        Xtrain = nu.scale_data(Xtrain)
        Xtest = nu.scale_data(Xtest)
        ytrain = ytrain.astype(int)
        ytest = ytest.astype(int)

        assert issubclass(ytrain.dtype.type, np.integer), "Training labels are not integers."
        assert issubclass(ytest.dtype.type, np.integer), "Test labels are not integers."
        answer = {}

        # Enter your code and fill the `answer` dictionary
        
        answer = {
        "length_Xtrain": len(Xtrain),
        "length_Xtest": len(Xtest),
        "length_ytrain": len(ytrain),
        "length_ytest": len(ytest),
        "max_Xtrain": np.max(Xtrain),
        "max_Xtest": np.max(Xtest)
        }
        return answer, Xtrain, ytrain, Xtest, ytest

    """
    C. Train your first classifier using k-fold cross validation (see train_simple_classifier_with_cv 
       function). Use 5 splits and a Decision tree classifier. Print the mean and standard deviation 
       for the accuracy scores in each validation set in cross validation. (with k splits, cross_validate
       generates k accuracy scores.)  
       Remember to set the random_state in the classifier and cross-validator.
    """

    # ----------------------------------------------------------------------
    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        # Enter your code and fill the `answer` dictionary
        clf = DecisionTreeClassifier(random_state=self.seed)
        cv = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        cv_results = u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf, cv=cv)
        print(cv_results)
        answer = {}
        answer['clf'] = clf
        answer['cv'] = cv

        answer = {}
        scores = {}



        mean_accuracy = cv_results['test_score'].mean()
        std_accuracy = cv_results['test_score'].std()

        mean_fit_time = cv_results['fit_time'].mean()
        std_fit_time = cv_results['fit_time'].std()

        scores['mean_fit_time'] = mean_fit_time
        scores['std_fit_time'] = std_fit_time
        scores['mean_accuracy'] = mean_accuracy
        scores['std_accuracy'] = std_accuracy

        answer['scores'] = scores
        answer = {
        "clf": clf,
        "cv": cv,
        "scores": {
            'mean_fit_time': mean_fit_time,
            'std_fit_time': std_fit_time,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy
                    }
        }
        
        return answer

    # ---------------------------------------------------------
    """
    D. Repeat Part C with a random permutation (Shuffle-Split) 𝑘-fold cross-validator.
    Explain the pros and cons of using Shuffle-Split versus 𝑘-fold cross-validation.
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        # Enter your code and fill the `answer` dictionary


        answer = {}
        clf = DecisionTreeClassifier(random_state=self.seed)
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=self.seed)
        cv_results = u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf, cv=cv)



        answer['clf'] = clf
        answer['cv'] = cv

        scores = {}



        mean_accuracy = cv_results['test_score'].mean()
        std_accuracy = cv_results['test_score'].std()

        mean_fit_time = cv_results['fit_time'].mean()
        std_fit_time = cv_results['fit_time'].std()

        scores['mean_fit_time'] = mean_fit_time
        scores['std_fit_time'] = std_fit_time
        scores['mean_accuracy'] = mean_accuracy
        scores['std_accuracy'] = std_accuracy

        answer['scores'] = scores
        answer['explain_kfold_vs_shuffle_split'] = """
                    K-Fold Cross-Validation splits the dataset into k consecutive folds, each fold used once as a test set while the k-1 remaining folds form the training set. It ensures that every observation from the original dataset has the chance of appearing in training and test set. It's well-suited for smaller datasets where maximizing the amount of training data is critical.

                    Shuffle-Split Cross-Validation generates a user-defined number of independent train/test dataset splits. Samples are first shuffled and then split into a pair of train and test sets. It's more flexible than k-fold, allowing for control over the number of iterations and the size of the test sets. It can be more suitable for large datasets or when requiring more randomness in the selection of train/test samples.

                    Pros of Shuffle-Split:
                    - More control over the size of the test set and the number of resampling iterations.
                    - Better for large datasets due to its randomness and efficiency.

                    Cons of Shuffle-Split:
                    - Less systematic coverage of the data compared to k-fold.
                    - Potential for higher variance in test performance across iterations due to randomness.

                    Empirical advantages:
                    - Takes less time
                    - More accuracy
                """
        return answer

    # ----------------------------------------------------------------------
    """
    E. Repeat part D for 𝑘=2,5,8,16, but do not print the training time. 
       Note that this may take a long time (2–5 mins) to run. Do you notice 
       anything about the mean and/or standard deviation of the scores for each k?
    """

    def partE(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        k_values = [2, 5, 8, 16]
        results = {}
        answer = {}
        for k in k_values:
                print(f"{k=}")
                clf = DecisionTreeClassifier(random_state=self.seed)
                cv = KFold(n_splits=k, shuffle=True, random_state=self.seed)

                cv_results = u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf, cv=cv)


                scores = {}



                mean_accuracy = cv_results['test_score'].mean()
                std_accuracy = cv_results['test_score'].std()

                mean_fit_time = cv_results['fit_time'].mean()
                std_fit_time = cv_results['fit_time'].std()

                scores['mean_fit_time'] = mean_fit_time
                scores['std_fit_time'] = std_fit_time
                scores['mean_accuracy'] = mean_accuracy
                scores['std_accuracy'] = std_accuracy
                answer[k] = {
                    'clf': clf,
                    'cv': cv,
                    'scores': scores
                }

        # Enter your code, construct the `answer` dictionary, and return it.

        return answer

    # ----------------------------------------------------------------------
    """
    F. Repeat part D with a Random-Forest classifier with default parameters. 
       Make sure the train test splits are the same for both models when performing 
       cross-validation. (Hint: use the same cross-validator instance for both models.)
       Which model has the highest accuracy on average? 
       Which model has the lowest variance on average? Which model is faster 
       to train? (compare results of part D and part F)

       Make sure your answers are calculated and not copy/pasted. Otherwise, the automatic grading 
       will generate the wrong answers. 
       
       Use a Random Forest classifier (an ensemble of DecisionTrees). 
    """

    def partF(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ) -> dict[str, Any]:
        """ """
        Xtrain, ytrain, Xtest, ytest = u.prepare_data()

        clf_rf = RandomForestClassifier(random_state=self.seed)
        clf_dt = DecisionTreeClassifier(random_state=self.seed)

        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=self.seed)

        cv_results_rf = u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf_rf, cv=cv)


        cv_results_dt = u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf_dt, cv=cv)

        scores_RF = {}
        scores_DT = {}

        answer = {}

        mean_accuracy = cv_results_rf['test_score'].mean()
        std_accuracy = cv_results_rf['test_score'].std()

        mean_fit_time = cv_results_rf['fit_time'].mean()
        std_fit_time = cv_results_rf['fit_time'].std()

        scores_RF['mean_fit_time'] = mean_fit_time
        scores_RF['std_fit_time'] = std_fit_time
        scores_RF['mean_accuracy'] = mean_accuracy
        scores_RF['std_accuracy'] = std_accuracy



        mean_accuracy = cv_results_dt['test_score'].mean()
        std_accuracy = cv_results_dt['test_score'].std()

        mean_fit_time = cv_results_dt['fit_time'].mean()
        std_fit_time = cv_results_dt['fit_time'].std()

        scores_DT['mean_fit_time'] = mean_fit_time
        scores_DT['std_fit_time'] = std_fit_time
        scores_DT['mean_accuracy'] = mean_accuracy
        scores_DT['std_accuracy'] = std_accuracy


        answer['clf_RF']  = clf_rf
        answer['clf_DT'] = clf_dt

        answer['scores_RF'] = scores_RF
        answer['scores_DT'] = scores_DT

        if scores_DT['mean_accuracy'] < scores_RF['mean_accuracy']:
            answer['model_highest_accuracy'] = 'Random Forest'
        else:
            answer['model_highest_accuracy'] = 'Decision Tree'

        if scores_DT['std_accuracy'] < scores_RF['std_accuracy']:
            answer['model_lowest_variance'] = 'Decision Tree'
        else:
            answer['model_lowest_variance'] = 'Random Forest'

        if scores_DT['mean_fit_time'] < scores_RF['mean_fit_time']:
            answer['model_fastest'] = 'Decision Tree'
        else:
            answer['model_fastest'] = 'Random Forest'


        clf_rf.fit(Xtrain, ytrain)
        y_pred_test_orig = clf_rf.predict(Xtest)

        return answer

    # ----------------------------------------------------------------------
    """
    G. For the Random Forest classifier trained in part F, manually (or systematically, 
       i.e., using grid search), modify hyperparameters, and see if you can get 
       a higher mean accuracy.  Finally train the classifier on all the training 
       data and get an accuracy score on the test set.  Print out the training 
       and testing accuracy and comment on how it relates to the mean accuracy 
       when performing cross validation. Is it higher, lower or about the same?

       Choose among the following hyperparameters: 
         1) criterion, 
         2) max_depth, 
         3) min_samples_split, 
         4) min_samples_leaf, 
         5) max_features 
    """

    def partG(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """
        Perform classification using the given classifier and cross validator.

        Parameters:
        - clf: The classifier instance to use for classification.
        - cv: The cross validator instance to use for cross validation.
        - X: The test data.
        - y: The test labels.
        - n_splits: The number of splits for cross validation. Default is 5.

        Returns:
        - y_pred: The predicted labels for the test data.

        Note:
        This function is not fully implemented yet.
        """


        """
        List of parameters you are allowed to vary. Choose among them.
         1) criterion,
         2) max_depth,
         3) min_samples_split, 
         4) min_samples_leaf,
         5) max_features 
         5) n_estimators
        """

        answer = {}
        clf = RandomForestClassifier(random_state=42)
        cv = 5
        param_grid = {
            'max_depth': [10, 20, 30, None],
            'n_estimators': [10, 50, 100, 200]
        }

        grid_search = GridSearchCV(clf, param_grid, cv=cv, scoring='accuracy', refit=True)
        grid_search.fit(X, y)

        best_estimator = grid_search.best_estimator_
        y_pred_train_orig = clf.fit(X, y).predict(X)
        y_pred_train_best = best_estimator.predict(X)
        y_pred_test_orig = clf.predict(Xtest)
        y_pred_test_best = best_estimator.predict(Xtest)

        answer = {
            "clf": clf,
            "default_parameters": clf.get_params(),
            "best_estimator": best_estimator,
            "grid_search": grid_search,
            "mean_accuracy_cv": grid_search.best_score_,
            "confusion_matrix_train_orig": confusion_matrix(y, y_pred_train_orig),
            "confusion_matrix_train_best": confusion_matrix(y, y_pred_train_best),
            "confusion_matrix_test_orig": confusion_matrix(ytest, y_pred_test_orig),
            "confusion_matrix_test_best": confusion_matrix(ytest, y_pred_test_best),
            "accuracy_orig_full_training": accuracy_score(y, y_pred_train_orig),
            "accuracy_best_full_training": accuracy_score(y, y_pred_train_best),
            "accuracy_orig_full_testing": accuracy_score(ytest, y_pred_test_orig),
            "accuracy_best_full_testing": accuracy_score(ytest, y_pred_test_best),
        }






        """
           `answer`` is a dictionary with the following keys: 
            
            "clf", base estimator (classifier model) class instance
            "default_parameters",  dictionary with default parameters 
                                   of the base estimator
            "best_estimator",  classifier class instance with the best
                               parameters (read documentation)
            "grid_search",  class instance of GridSearchCV, 
                            used for hyperparameter search
            "mean_accuracy_cv",  mean accuracy score from cross 
                                 validation (which is used by GridSearchCV)
            "confusion_matrix_train_orig", confusion matrix of training 
                                           data with initial estimator 
                                (rows: true values, cols: predicted values)
            "confusion_matrix_train_best", confusion matrix of training data 
                                           with best estimator
            "confusion_matrix_test_orig", confusion matrix of test data
                                          with initial estimator
            "confusion_matrix_test_best", confusion matrix of test data
                                            with best estimator
            "accuracy_orig_full_training", accuracy computed from `confusion_matrix_train_orig'
            "accuracy_best_full_training"
            "accuracy_orig_full_testing"
            "accuracy_best_full_testing"
               
        """

        return answer
