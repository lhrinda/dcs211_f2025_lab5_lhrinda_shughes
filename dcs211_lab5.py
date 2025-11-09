import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)
import seaborn as sns   # yay for Seaborn plots!
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter, defaultdict
import random

###########################################################################
def drawDigitHeatmap(pixels: np.ndarray, showNumbers: bool = True) -> None:
    ''' Draws a heat map of a given digit based on its 8x8 set of pixel values.
    Parameters:
        pixels: a 2D numpy.ndarray (8x8) of integers of the pixel values for
                the digit
        showNumbers: if True, shows the pixel value inside each square
    Returns:
        None -- just plots into a window
    '''
    (fig, axes) = plt.subplots(figsize = (4.5, 3))  # aspect ratio

    rgb = (0, 0, 0.5)  # each in (0,1), so darkest will be dark blue
    colormap = sns.light_palette(rgb, as_cmap=True)    
    # all seaborn palettes: 
    # https://medium.com/@morganjonesartist/color-guide-to-seaborn-palettes-da849406d44f

    # plot the heatmap;  see: https://seaborn.pydata.org/generated/seaborn.heatmap.html
    # (fmt = "d" indicates to show annotation with integer format)
    sns.heatmap(pixels, annot = showNumbers, fmt = "d", linewidths = 0.5, \
                ax = axes, cmap = colormap)
    plt.show(block = True)

###########################################################################
def fetchDigit(df: pd.core.frame.DataFrame, which_row: int) -> tuple[int, np.ndarray]:
    ''' For digits.csv data represented as a dataframe, this fetches the digit from
        the corresponding row, reshapes, and returns a tuple of the digit and a
        numpy array of its pixel values.
    Parameters:
        df: pandas data frame expected to be obtained via pd.read_csv() on digits.csv
        which_row: an integer in 0 to len(df)
    Returns:
        a tuple containing the reprsented digit and a numpy array of the pixel
        values
    '''
    digit  = int(round(df.iloc[which_row, 64]))
    pixels = df.iloc[which_row, 0:64]   # don't want the rightmost rows
    pixels = pixels.values              # converts to numpy array
    pixels = pixels.astype(int)         # convert to integers for plotting
    pixels = np.reshape(pixels, (8,8))  # makes 8x8
    return (digit, pixels)              # return a tuple

###########################################################################
def cleanTheData(df: pd.core.frame.DataFrame) -> np.ndarray:
    ''' Clean the dataframe by dropping columns that are entirely 
        NaN, dropping rows containing any NaN, and converting the result to 
        a float64 NumPy array (features first, label in the last column).
    Parameters:
        df: pandas data frame obtained by reading csv
    Returns:
        A: a NumPy array (dtype float64) of the cleaned data
    '''
    # 1) find fully-empty columns and drop them (class-style)
    cols_all_nan = []
    for col in df.columns:
        if df[col].isna().all():
            cols_all_nan.append(col)
    df_clean = df.drop(columns=cols_all_nan)

    # 2) drop any rows with NaN
    df_clean = df_clean.dropna()

    # 3) convert to numpy and cast to float64
    A = df_clean.to_numpy().astype('float64')
    return A

###################
def predictiveModel(train_set: np.ndarray, features: np.ndarray) -> int:
    '''Uses the 1-NN predictive model on a given training set with given features
    Parameters:
        train_set: a training data set with one row cooresponding to an image of a digit
        features: the pixel values for a given digit
    Returns:
        an int coorespondng to the predicted digit for the given test arrays
    '''
    # 1) separates array into data and labels 
    pixels = train_set[:, :-1]
    labels = train_set[:, -1]

    # 2) Find smallest distance from set
    distances = np.linalg.norm(pixels - features, axis=1)
    nearest_index = np.argmin(distances)

    # Return predicted int
    return int(labels[nearest_index])

###################
def splitData(A: np.ndarray, test_fraction: float = 0.2, swap: bool = False) -> list:
    ''' Splits an array into test/train
    Parameters:
        A: numpy array where last column is the label
        test_fraction: fraction of rows to use for the test set (default 0.2)
        swap: if False -> LAST test_fraction of rows, if True -> FIRST test_fraction of rows
    Returns:
        [X_test, y_test, X_train, y_train]
    '''
    n = len(A)
    cut = int((1 - test_fraction) * n)

    if not swap:
        # test = last 20%, train = first 80%
        train = A[:cut]
        test  = A[cut:]
    else:
        # test = first 20%, train = last 80%
        test  = A[:n - cut]
        train = A[n - cut:]

    X_train = train[:, :-1]
    y_train = train[:, -1].astype(int)
    X_test  = test[:, :-1]
    y_test  = test[:, -1].astype(int)

    return [X_test, y_test, X_train, y_train]

###################
def compareLabels(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    '''Prints totals, accuracy, and a few mismatches
    Parameters:
        y_true: 1D numpy array of true integer labels
        y_pred: 1D numpy array of predicted integer labels
    Returns:
        None
    '''
    total = len(y_true)
    correct = int(np.sum(y_true == y_pred))
    accuracy = correct / total if total > 0 else 0.0
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.3f}")

    mismatches = np.where(y_true != y_pred)[0]
    if len(mismatches) > 0:
        print("First 10 mismatches (index: true → pred):")
        for idx in mismatches[:10]:
            print(f"  {idx}: {int(y_true[idx])} → {int(y_pred[idx])}")
    else:
        print("No mismatches!")

def run_knn_sklearn(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, k: int) -> np.ndarray:
    '''Fit scikit-learn KNN with chosen k and return predictions for X_test
    Parameters:
        X_train: 2D numpy array of training features (rows = samples, cols = features)
        y_train: 1D numpy array of training labels (integers 0–9)
        X_test: 2D numpy array of test features
        k: integer number of neighbors to use (k ≥ 1)
    Returns:
        y_pred: 1D numpy array of predicted integer labels for X_test
    '''
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn.predict(X_test)

###################
def findBestK(X_train: np.ndarray, y_train: np.ndarray, seeds: list = [8675309, 5551212, 42], k_values: list = [1, 3, 5, 7, 9], val_fraction: float = 0.2) -> dict:
    ''' Finds the best k using a shuffle split for multiple seeds
    Parameters:
        X_train: 2D numpy array of training features
        y_train: 1D numpy array of training labels (ints 0–9)
        seeds: list of integer seeds to control shuffling/splitting
        k_values: list of k values (neighbors) to evaluate
        val_fraction: fraction of rows to hold out for validation for each seed
    Returns:
        results: dict with:
            'per_seed': list of dicts:
                { 'seed': int, 'best_k': int, 'best_acc': float, 'k_to_acc': {k: acc} }
            'final_k': int
    '''
    per_seed = []

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)

        # shuffle indices
        n = len(X_train)
        idx = list(range(n))
        random.shuffle(idx)

        # split into sub-train / validation
        cut = int((1 - val_fraction) * n)
        idx_tr = idx[:cut]
        idx_va = idx[cut:]

        X_tr = X_train[idx_tr]
        y_tr = y_train[idx_tr]
        X_va = X_train[idx_va]
        y_va = y_train[idx_va]

        # evaluate each k
        k_to_acc = {}
        for k in k_values:
            y_pred = run_knn_sklearn(X_tr, y_tr, X_va, k)
            acc = float(np.mean(y_pred == y_va))
            k_to_acc[k] = acc

        # pick best k for this seed
        best_k = sorted(k_to_acc.keys(), key=lambda kk: (-k_to_acc[kk], kk))[0]
        best_acc = k_to_acc[best_k]
        per_seed.append({
            'seed': seed,
            'best_k': best_k,
            'best_acc': best_acc,
            'k_to_acc': k_to_acc
        })

    # decide a final k across seeds: majority vote -> avg acc -> smaller k
    vote = Counter([r['best_k'] for r in per_seed])
    max_votes = max(vote.values())
    candidates = [k for k, v in vote.items() if v == max_votes]

    if len(candidates) == 1:
        final_k = candidates[0]
    else:
        avg_acc = defaultdict(list)
        for r in per_seed:
            for k, acc in r['k_to_acc'].items():
                avg_acc[k].append(acc)
        best_by_avg = sorted(candidates, key=lambda k: (-np.mean(avg_acc[k]), k))
        final_k = best_by_avg[0]

    return {'per_seed': per_seed, 'final_k': final_k}

###################
def trainAndTest(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, best_k: int) -> np.ndarray:
    ''' Train k-NN with best_k and predict labels for X_test
    Parameters:
        X_train: 2D numpy array of training features (rows = samples, cols = features)
        y_train: 1D numpy array of training labels (integers 0–9)
        X_test: 2D numpy array of test features
        best_k: integer number of neighbors to use (k ≥ 1)
    Returns:
        y_pred: 1D numpy array of predicted integer labels for X_test
    '''
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return y_pred

###################
def main() -> None:
    # load csv
    filename = 'digits.csv'
    df = pd.read_csv(filename, header=0)
    print(f"{filename} : file read into a pandas dataframe...")

    A = cleanTheData(df)

    SHOW_HEATMAP = False
    if SHOW_HEATMAP:
        r = random.randint(0, len(df) - 1)
        digit, pixels = fetchDigit(df, r)
        print(f"Preview digit at row {r}: {digit}")
        drawDigitHeatmap(pixels)

    # split 1: test = last 20%
    print("\n=== First split (train=first 80%, test=last 20%) ===")
    X_test, y_test, X_train, y_train = splitData(A, test_fraction=0.2, swap=False)

    # choose best k (three seeds)
    fbk = findBestK(X_train, y_train, seeds=[8675309, 5551212, 123123], k_values=[1, 3, 5, 7, 9], val_fraction=0.2)
    best_k = fbk['final_k']
    print(f"[findBestK] final chosen k = {best_k}")
    for r in fbk['per_seed']:
        print(f"  seed {r['seed']}: best_k={r['best_k']}, acc={r['best_acc']:.3f}")

    # train & test with best_k
    y_pred_best = trainAndTest(X_train, y_train, X_test, best_k)
    print(f"\n[trainAndTest] First split, k={best_k}")
    compareLabels(y_test, y_pred_best)

    # also report a guessed k (k=3) using scikit-learn
    guessed_k = 3
    y_pred_k3 = run_knn_sklearn(X_train, y_train, X_test, guessed_k)
    print(f"\n[scikit-learn] First split (k={guessed_k})")
    compareLabels(y_test, y_pred_k3)

    # baseline 1-NN + gather first 5 wrong for heatmaps
    train_set = np.column_stack([X_train, y_train])
    test_set  = np.column_stack([X_test,  y_test])

    correct = 0
    wrong = []
    for i in range(len(test_set)):
        test_features = test_set[i, :-1]
        true_label = int(test_set[i, -1])
        pred = predictiveModel(train_set, test_features)
        if pred == true_label:
            correct += 1
        else:
            wrong.append((test_features, true_label, pred))
    print(f"\n[1-NN manual] First split accuracy: {correct/len(test_set):.3f}")
    
    if wrong:
        print("First five misclassified (first split):")
        for j in range(min(5, len(wrong))):
            pixels_flat, true_label, pred = wrong[j]
            print(f"  #{j+1}: True={true_label}, Pred={pred}")
            drawDigitHeatmap(pixels_flat.reshape(8, 8).astype(int), showNumbers=True)

    # split 2: test = first 20%
    print("\n=== Swapped split (train=last 80%, test=first 20%) ===")
    X_test2, y_test2, X_train2, y_train2 = splitData(A, test_fraction=0.2, swap=True)

    # choose best k on swapped training set
    fbk2 = findBestK(X_train2, y_train2, seeds=[8675309, 5551212, 123123], k_values=[1, 3, 5, 7, 9], val_fraction=0.2)
    best_k2 = fbk2['final_k']
    print(f"[findBestK - swapped] final chosen k = {best_k2}")
    for r in fbk2['per_seed']:
        print(f"  seed {r['seed']}: best_k={r['best_k']}, acc={r['best_acc']:.3f}")

    # train & test with best_k on swapped split
    y_pred_best_2 = trainAndTest(X_train2, y_train2, X_test2, best_k2)
    print(f"\n[trainAndTest] Swapped split (k={best_k2})")
    compareLabels(y_test2, y_pred_best_2)
        
    y_pred_k3_2 = run_knn_sklearn(X_train2, y_train2, X_test2, guessed_k)
    print(f"\n[scikit-learn] Swapped split (k={guessed_k})")
    compareLabels(y_test2, y_pred_k3_2)

    # baseline 1-NN on swapped split
    train_set2 = np.column_stack([X_train2, y_train2])
    test_set2  = np.column_stack([X_test2,  y_test2])

    correct2 = 0
    wrong2 = []
    total2 = len(test_set2)
    for i in range(total2):
        test_features = test_set2[i, :-1]
        true_label = int(test_set2[i, -1])
        pred = predictiveModel(train_set2, test_features)
        if pred == true_label:
            correct2 += 1
        else:
            wrong2.append((test_features, true_label, pred))
    print(f"\n[1-NN manual] Swapped split accuracy: {correct2/total2:.3f}")

    if wrong2:
        print("First five misclassified (swapped split):")
        for idx in range(min(5, len(wrong2))):
            pixels_flat, true_label, pred = wrong2[idx]
            print(f"  #{idx+1}: True={true_label}, Pred={pred}")
            drawDigitHeatmap(pixels_flat.reshape(8, 8).astype(int), showNumbers=True)

###############################################################################
# wrap the call to main inside this if so that _this_ file can be imported
# and used as a library, if necessary, without executing its main
if __name__ == "__main__":
    main()