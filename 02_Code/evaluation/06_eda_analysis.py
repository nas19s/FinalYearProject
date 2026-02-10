import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "01_Data")
INPUT_FILE = os.path.join(DATA_DIR, "final_feature_dataset.csv")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "04_Results", "figures")

# Create figures directory
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set visual style for academic report
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

def run_eda():
    print("Starting Exploratory Data Analysis...")
    
    if not os.path.exists(INPUT_FILE):
        print("Error: Dataset not found.")
        return
    
    df = pd.read_csv(INPUT_FILE)
    
    # Filter for the Medium Term target (Label_Month) as decided in Baseline
    # Remove Neutrals (-1) to match our modeling approach
    target_col = 'Label_Month'
    df_clean = df[df[target_col] != -1].copy()
    
    # Map numeric labels to text for better plotting
    df_clean['Direction'] = df_clean[target_col].map({0: 'Down', 1: 'Up'})
    
    print(f"Analyzing {len(df_clean)} samples (Neutrals removed).")

    # --- PLOT 1: Class Imbalance ---
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(x='Direction', data=df_clean, palette='viridis', order=['Down', 'Up'])
    plt.title('Distribution of Target Labels (Medium Term)')
    plt.xlabel('Stock Direction')
    plt.ylabel('Count')
    
    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + 0.35, p.get_height() + 5))
    
    save_path_1 = os.path.join(FIGURES_DIR, "class_distribution.png")
    plt.savefig(save_path_1, dpi=300)
    print(f"Saved Plot 1: {save_path_1}")

    # --- PLOT 2: Feature vs Target (Violin Plot) ---
    # Does 'Gunning Fog' (Readability) actually differ between UP and DOWN stocks?
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Direction', y='Gunning_Fog', data=df_clean, palette='muted', order=['Down', 'Up'])
    plt.title('Readability (Gunning Fog) by Stock Direction')
    plt.xlabel('Stock Direction')
    plt.ylabel('Gunning Fog Index (Higher = More Complex)')
    
    save_path_2 = os.path.join(FIGURES_DIR, "readability_vs_direction.png")
    plt.savefig(save_path_2, dpi=300)
    print(f"Saved Plot 2: {save_path_2}")

    # --- PLOT 3: Sentiment vs Target ---
    # Checking if Sentiment separates the classes better than Readability
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Direction', y='Sentiment', data=df_clean, palette='coolwarm', order=['Down', 'Up'])
    plt.title('VADER Sentiment Score by Stock Direction')
    plt.xlabel('Stock Direction')
    plt.ylabel('Compound Sentiment Score')
    
    save_path_3 = os.path.join(FIGURES_DIR, "sentiment_vs_direction.png")
    plt.savefig(save_path_3, dpi=300)
    print(f"Saved Plot 3: {save_path_3}")

    # --- PLOT 4: Correlation Matrix ---
    # Check if our features are redundant (highly correlated)
    cols_to_corr = [
        'Gunning_Fog', 'Flesch_Ease', 'Sentiment', 
        'RSI', 'MACD', 'Volume_Change', 'Return_Month'
    ]
    plt.figure(figsize=(10, 8))
    corr = df_clean[cols_to_corr].corr()
    sns.heatmap(corr, annot=True, cmap='RdBu', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    
    save_path_4 = os.path.join(FIGURES_DIR, "correlation_matrix.png")
    plt.savefig(save_path_4, dpi=300)
    print(f"Saved Plot 4: {save_path_4}")

    print("EDA Complete. Check the '04_Results/figures' folder.")

if __name__ == "__main__":
    run_eda()