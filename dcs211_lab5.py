import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)
import seaborn as sns   # yay for Seaborn plots!
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import random
from tqdm import tqdm

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
    A = df_clean.to_numpy()
    A = A.astype('float64')

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
def main() -> None:
    # for read_csv, use header=0 when row 0 is a header row
    filename = 'digits.csv'
    df = pd.read_csv(filename, header = 0)
    print(df.head())
    print(f"{filename} : file read into a pandas dataframe...")

    A = cleanTheData(df)

    num_to_draw = 1
    for i in range(num_to_draw):
        # let's grab one row of the df at random, extract/shape the digit to be
        # 8x8, and then draw a heatmap of that digit
        random_row = random.randint(0, len(df) - 1)
        (digit, pixels) = fetchDigit(df, random_row)
        print(f"The digit is {digit}")
        print(f"The pixels are\n{pixels}")  
        drawDigitHeatmap(pixels)
        plt.show()

    # split 1: test = last 20%
    X_test, y_test, X_train, y_train = splitData(A, test_fraction=0.2, swap=False)
    train_set = np.column_stack([X_train, y_train])
    test_set  = np.column_stack([X_test,  y_test])

    guessed_k = 3  # small k smooths 1-NN noise but stays local
    y_pred_sklearn = run_knn_sklearn(X_train, y_train, X_test, guessed_k)
    print(f"[scikit-learn] First split, k={guessed_k}")
    compareLabels(y_test, y_pred_sklearn)

    correct = 0
    total = len(test_set)
    wrong = []
    
    for i in tqdm(range(total), desc="Predicting test digits"):
        test_features = test_set[i, :-1]
        true_label = int(test_set[i, -1])
        predicted_label = predictiveModel(train_set, test_features)
        if predicted_label == true_label:
            correct += 1
        else:
            wrong.append((test_features, true_label, predicted_label))
    
    accuracy = correct / total
    
    print(f"\nAccuracy: {accuracy:.3f}")

    # split 2: test = first 20%

    X_test2, y_test2, X_train2, y_train2 = splitData(A, test_fraction=0.2, swap=True)
    train_set2 = np.column_stack([X_train2, y_train2])
    test_set2  = np.column_stack([X_test2,  y_test2])

    y_pred_sklearn_2 = run_knn_sklearn(X_train2, y_train2, X_test2, guessed_k)
    print(f"\n[scikit-learn] Swapped split, k={guessed_k}")
    compareLabels(y_test2, y_pred_sklearn_2)
        
    correct2 = 0
    total2 = len(test_set)
    wrong2 = []

    
    for i in tqdm(range(total2), desc="Predicting test digits again"):
        test_features = test_set2[i, :-1]
        true_label = int(test_set2[i, -1])
        predicted_label = predictiveModel(train_set2, test_features)
        if predicted_label == true_label:
            correct2 += 1
        else:
            wrong2.append((test_features, true_label, predicted_label))
    
    accuracy2 = correct2 / total2
    print(f"\nAccuracy #2: {accuracy2:.3f}")

    print("Reviewing incorrect answers from first test:")
    for j in range(min(5, len(wrong))):
        pixels_flat, true_label, predicted_label = wrong[j]
        pixels_8x8 = pixels_flat.reshape((8, 8)).astype(int)
        print(f"#{j+1}: True={true_label}, Predicted={predicted_label}")
        drawDigitHeatmap(pixels_8x8, showNumbers=True)

    print("Reviewing incorrect answers from second test:")
    for k in range(min(5, len(wrong2))):
        pixels_flat, true_label, predicted_label = wrong2[k]
        pixels_8x8 = pixels_flat.reshape((8, 8)).astype(int)
        print(f"#{k+1}: True={true_label}, Predicted={predicted_label}")
        drawDigitHeatmap(pixels_8x8, showNumbers=True)

###############################################################################
# wrap the call to main inside this if so that _this_ file can be imported
# and used as a library, if necessary, without executing its main
if __name__ == "__main__":
    main()
