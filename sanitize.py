import pandas as pd
import numpy as np
import re

# Configurações de arquivos
INPUT_FILE = "files/itens.xlsx"
OUTPUT_CLEAN = "compras_primeiros_1000_limpo.xlsx"
OUTPUT_FINAL = "tabelaInicial.xlsx"

# Índices de colunas a remover na segunda etapa (0-based)
COLS_TO_DROP = [5, 6, 7]


def remove_case_insensitive_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicates based on case-insensitive comparison of the 'Produto/Serviço' column.
    Keeps the first occurrence of each product.
    """
    # Create a temporary lowercase version of 'Produto/Serviço' for comparison
    df["produto_lower"] = df["Produto/Serviço"].str.lower()

    # Drop duplicates based on the lowercase version
    df_no_dupes = df.drop_duplicates(subset=["produto_lower"])

    # Remove the temporary column
    df_no_dupes = df_no_dupes.drop("produto_lower", axis=1)

    return df_no_dupes


def clean_table(df: pd.DataFrame, max_rows: int = 100) -> tuple[pd.DataFrame, list]:
    print(f"Número inicial de linhas: {len(df)}")

    # 1) Remove colunas totalmente vazias e as colunas "Unnamed"
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    # 2) Remove linhas totalmente vazias
    rows_before = len(df)
    df = df.dropna(how="all")
    print(f"Linhas removidas (vazias): {rows_before - len(df)}")

    # 3) Remove linhas com texto indesejado na primeira coluna
    rows_before = len(df)
    col0 = df.columns[0]
    pattern = r"Contabilis\s*-\s*Desenvolvido\s*por\s*3Tecnos\s*Tecnologia"
    df = df[~df[col0].fillna("").str.contains(pattern, regex=True)]
    print(f"Linhas removidas (padrão Contabilis): {rows_before - len(df)}")

    # 4) Remove case-insensitive duplicates
    rows_before = len(df)
    df = remove_case_insensitive_duplicates(df)
    print(f"Linhas removidas (duplicatas case-insensitive): {rows_before - len(df)}")

    # 5) Identifica colunas "zeradas"
    zero_cols = []
    for c in df.columns:
        try:
            vals = df[c].str.replace(",", ".", regex=False).astype(float)
        except Exception:
            continue
        if (vals == 0).all():
            zero_cols.append(c)

    # 6) Esvazia colunas zeradas
    for c in zero_cols:
        df[c] = ""

    print(f"Número final de linhas antes do subset: {len(df)}")

    # 7) Subconjunto dos primeiros registros
    df_subset = df.head(max_rows)
    print(f"Número de linhas após subset: {len(df_subset)}")

    return df_subset, zero_cols


def drop_unnecessary(df: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    # Remove colunas pelo índice
    df = df.drop(df.columns[cols_to_drop], axis=1)
    return df


def main():
    # Leitura inicial
    df = pd.read_excel("files/itens.xlsx", header=8, dtype=str)  # ler tudo como string

    # Limpeza inicial
    df_clean, zero_cols = clean_table(df, max_rows=70000)  # Aumentado para 70000 linhas
    print(f"Gerado: {OUTPUT_CLEAN}")
    print(f"- Colunas zeradas (esvaziadas): {zero_cols}")
    print(f"- Linhas totais após limpeza: {len(df_clean)}")

    # Remoção de colunas desnecessárias
    df_final = drop_unnecessary(df_clean, COLS_TO_DROP)
    df_final.to_excel(OUTPUT_FINAL, index=False)
    print(f"Gerado: {OUTPUT_FINAL}")
    print(f"- Colunas removidas (índices): {COLS_TO_DROP}")


if __name__ == "__main__":
    main()
