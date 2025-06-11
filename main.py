# script.py
import pandas as pd
import numpy as np


def main():
    # 1) Leia o Excel definindo a 9ª linha como header
    df = pd.read_excel(
        "files/itens.xlsx",
        header=8,
        dtype=str,  # ler tudo como string para preservar formatação
    )

    # 2) Remove colunas totalmente vazias e as "Unnamed"
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    # 3) (Opcional) Remove linhas totalmente vazias
    df = df.dropna(how="all")

    # 4) Remove linhas cujo valor na primeira coluna contenha o texto indesejado
    col0 = df.columns[0]
    pattern = r"Contabilis\s*-\s*Desenvolvido\s*por\s*3Tecnos\s*Tecnologia"
    df = df[~df[col0].fillna("").str.contains(pattern, regex=True)]

    # 5) Identifica colunas “zeradas” (todas as células iguais a "0", "0.0", "0,000" etc.)
    zero_cols = []
    for c in df.columns:
        # tenta converter para float; se falhar, ignora a coluna
        try:
            vals = (
                df[c]
                .str.replace(",", ".", regex=False)  # vírgula → ponto
                .astype(float)
            )
        except Exception:
            continue

        if (vals == 0).all():
            zero_cols.append(c)

    # 6) Deixa essas colunas vazias (substitui todos os valores por string vazia)
    for c in zero_cols:
        df[c] = ""

    # 7) (Se quiser só os 1 000 primeiros registros)
    df_subset = df.head(1000)

    # 8) Salva o resultado
    df_subset.to_excel("compras_primeiros_1000_limpo.xlsx", index=False)
    print("Gerado compras_primeiros_1000_limpo.xlsx")
    print(f"- Colunas zeradas (esvaziadas): {zero_cols}")
    print(f"- Linhas totais após limpeza: {len(df_subset)}")


if __name__ == "__main__":
    main()
