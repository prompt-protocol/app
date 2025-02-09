import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
import io
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from google.colab import drive

# --- Constantes e Configurações ---
DRIVE_MOUNT_PATH = "/content/drive"
GRAPHPATH = os.path.join(DRIVE_MOUNT_PATH, "MyDrive", "graficos")
SUMMARYPATH = os.path.join(DRIVE_MOUNT_PATH, "MyDrive", "sumarios")
DEFAULT_PALETTE = "viridis"
FIGSIZE = (10, 6)
DATASET_URL = "https://docs.google.com/spreadsheets/d/1_jKH6GX7N7Vutxn4dJ_fKeLDJraLaz9jXLUKTRIiu4c/export?format=csv"

# Configuração de estilo
sns.set(style="darkgrid")
plt.rcParams.update({
    'figure.facecolor': '#282c34',
    'axes.facecolor': '#282c34',
    'axes.edgecolor': 'white',
    'axes.labelcolor': 'cyan',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'text.color': 'white',
    'grid.color': 'gray',
    'grid.linestyle': '--',
    'legend.facecolor': '#282c34',
    'legend.edgecolor': 'white',
    'figure.titlesize': 16,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
})

# --- Funções Auxiliares ---

def save_fig(filename):
    filepath = os.path.join(GRAPHPATH, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

def create_figure():
    return plt.figure(figsize=FIGSIZE)

def load_data_from_url(url):
    """Carrega dados de uma URL CSV."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Lança exceção para erros HTTP
        csv_content = response.content.decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_content))
        return df
    except Exception as e:
        print(f"Erro ao carregar dados da URL: {e}")
        return None

# --- Funções de Visualização ---

def generate_boxplot(df, x, y, title="", filename="", hue=None):
    create_figure()
    if hue is not None:
        sns.boxplot(x=x, y=y, data=df, palette=DEFAULT_PALETTE, hue=hue)
    else:
        sns.boxplot(x=x, y=y, data=df, palette=DEFAULT_PALETTE)  # Removido hue=x
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    save_fig(filename)

def generate_histogram(df, column, title="", filename="", bins=20, kde=True):
    create_figure()
    sns.histplot(data=df, x=column, kde=kde, bins=bins, color='skyblue')
    plt.title(title)
    save_fig(filename)

def generate_scatterplot(df, x, y, title="", filename="", hue=None, style=None):
    create_figure()
    sns.scatterplot(x=x, y=y, data=df, alpha=0.7, hue=hue, style=style, palette=DEFAULT_PALETTE)
    plt.title(title)
    save_fig(filename)

def generate_heatmap(df, title="", filename="", annot=True):
    create_figure()
    # Select only numeric columns for the correlation matrix
    df_numeric = df.select_dtypes(include=np.number)
    corr = df_numeric.corr()
    sns.heatmap(corr, annot=annot, cmap=DEFAULT_PALETTE, fmt=".2f", linewidths=.5)
    plt.title(title)
    save_fig(filename)

def generate_violinplot(df, x, y, title="", filename="", hue=None, split=False):
    create_figure()
    sns.violinplot(x=x, y=y, data=df, palette=DEFAULT_PALETTE, hue=hue, split=split)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    save_fig(filename)

# --- Função Principal (main) ---
if __name__ == "__main__":
    drive.mount(DRIVE_MOUNT_PATH, force_remount=True)
    os.makedirs(GRAPHPATH, exist_ok=True)
    os.makedirs(SUMMARYPATH, exist_ok=True)

    # --- Carregar Dados da URL ---
    df = load_data_from_url(DATASET_URL)
    if df is None:
        print("Não foi possível carregar os dados. Encerrando.")
        exit()

    # --- Pré-processamento (Label Encoding) ---
    label_encoder = LabelEncoder()
    if 'CategoriaPrompt' in df.columns:
        df['CategoriaPrompt_encoded'] = label_encoder.fit_transform(df['CategoriaPrompt'])
    else:
        print("Coluna 'CategoriaPrompt' não encontrada para Label Encoding.")


    # --- Análises e Visualizações ---

    # 1. Distribuição das Avaliações Humanas (Histogramas)
    for col in ['Coerencia (Humana)', 'Relevancia (Humana)', 'Profundidade (Humana)', 'Consciencia (Humana)']:
        if col in df.columns:
            generate_histogram(df, col, title=f"Distribuição de {col}", filename=f"1_hist_{col}.png")
        else:
            print(f"A coluna '{col}' não foi encontrada no DataFrame.")

    # 2. Boxplot Comparando Categorias de Prompt (Avaliação Humana)
    for col in ['Coerencia (Humana)', 'Relevancia (Humana)', 'Profundidade (Humana)', 'Consciencia (Humana)']:
        if 'CategoriaPrompt' in df.columns and col in df.columns:
            generate_boxplot(df, 'CategoriaPrompt', col, title=f"{col} por Categoria de Prompt", filename=f"2_boxplot_{col}.png")
        else:
            print(f"Colunas necessárias ('CategoriaPrompt' ou '{col}') não encontradas.")

    # 3. Scatterplot BLEU vs. ROUGE
    if 'BLEU' in df.columns and 'ROUGE' in df.columns:
        generate_scatterplot(df, 'BLEU', 'ROUGE', title="BLEU vs. ROUGE", filename="3_scatter_bleu_rouge.png", hue='CategoriaPrompt_encoded' if 'CategoriaPrompt_encoded' in df.columns else None)
    else:
        print("Colunas 'BLEU' ou 'ROUGE' não encontradas.")

    # 4. Histograma da Perplexidade
    if 'Perplexidade' in df.columns:
        generate_histogram(df, 'Perplexidade', title="Distribuição da Perplexidade", filename="4_hist_perplexidade.png")
    else:
        print("Coluna 'Perplexidade' não encontrada.")

    # 5. Violin Plot da Perplexidade por Categoria
    if 'Perplexidade' in df.columns and 'CategoriaPrompt' in df.columns:
        generate_violinplot(df, 'CategoriaPrompt', 'Perplexidade', title="Perplexidade por Categoria de Prompt", filename="5_violin_perplexidade.png")
    else:
        print("Colunas necessárias ('CategoriaPrompt' ou 'Perplexidade') não encontradas.")

    # 6. Scatterplot Perplexidade vs. Coerência
    if 'Perplexidade' in df.columns and 'Coerencia (Humana)' in df.columns:
        generate_scatterplot(df, 'Perplexidade', 'Coerencia (Humana)', title="Perplexidade vs. Coerência", filename="6_scatter_perplexidade_coerencia.png", hue='CategoriaPrompt_encoded' if 'CategoriaPrompt_encoded' in df.columns else None)
    else:
        print("Colunas 'Perplexidade' ou 'Coerencia (Humana)' não encontradas.")

    # 7. Heatmap de Correlação
    generate_heatmap(df, title="Heatmap de Correlação", filename="7_heatmap.png")


    # 8. Scatterplot Diversidade vs. Originalidade
    if 'Diversidade' in df.columns and 'Originalidade' in df.columns:
        generate_scatterplot(df, 'Diversidade', 'Originalidade', title="Diversidade vs. Originalidade", filename="8_scatter_diversidade_originalidade.png", hue= 'CategoriaPrompt_encoded' if 'CategoriaPrompt_encoded' in df.columns else None)
    else:
        print("Colunas 'Diversidade' ou 'Originalidade' não encontradas.")

    # 9. Boxplot Diversidade por Categoria
    if 'Diversidade' in df.columns and 'CategoriaPrompt' in df.columns:
        generate_boxplot(df, 'CategoriaPrompt', 'Diversidade', title="Diversidade por Categoria de Prompt", filename="9_boxplot_diversidade.png")
    else:
        print("Colunas necessárias ('CategoriaPrompt' ou 'Diversidade') não encontradas.")

    # 10. Boxplot Originalidade por Categoria
    if 'Originalidade' in df.columns and 'CategoriaPrompt' in df.columns:
        generate_boxplot(df, 'CategoriaPrompt', 'Originalidade', title="Originalidade por Categoria de Prompt", filename="10_boxplot_originalidade.png")
    else:
        print("Colunas necessárias ('CategoriaPrompt' ou 'Originalidade') não encontradas.")

    # 11. Scatterplot Profundidade vs. Consciência
    if 'Profundidade (Humana)' in df.columns and 'Consciencia (Humana)' in df.columns:
        generate_scatterplot(df, 'Profundidade (Humana)', 'Consciencia (Humana)', title="Profundidade vs. Consciência", filename="11_scatter_profundidade_consciencia.png", hue='CategoriaPrompt_encoded' if 'CategoriaPrompt_encoded' in df.columns else None)
    else:
        print("Colunas 'Profundidade (Humana)' ou 'Consciencia (Humana)' não encontradas.")

    print("Análise e gráficos gerados com sucesso!")
