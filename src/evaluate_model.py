import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt
import shap

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and prints various metrics."""
    preds = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    auc = roc_auc_score(y_test, preds)
    f1 = f1_score(y_test, (preds > 0.5).astype(int))
    mcc = matthews_corrcoef(y_test, (preds > 0.5).astype(int))

    print(f"AUC: {auc}")
    print(f"F1-Score: {f1}")
    print(f"Matthews Correlation Coefficient: {mcc}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, preds)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()

def explain_model(model, X_test):
    """Generates and plots SHAP values for model explanation."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Plot summary plot
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    shap.summary_plot(shap_values, X_test)


if __name__ == '__main__':
    # Load the test data
    test_df = pd.read_csv('data/processed/test.csv')
    features = [col for col in test_df.columns if col not in ['SK_ID_CURR', 'TARGET']]
    X_test = test_df[features]
    y_test = test_df['TARGET']

    # Load the model
    model = joblib.load('models/lightgbm_model.pkl')

    # Evaluate and explain the model
    evaluate_model(model, X_test, y_test)
    explain_model(model, X_test)
