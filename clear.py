# remove_cols.py
import pandas as pd


def main():
    # 1) Abra o arquivo limpo que você já gerou
    df = pd.read_excel("compras_primeiros_1000_limpo.xlsx", dtype=str)

    # 2) Verifique o nome das colunas (opcional, só para conferir)
    print("Colunas antes:", df.columns.tolist())

    # 3) Remove colunas desnecessárias
    df = df.drop(df.columns[[5, 6, 7]], axis=1)

    # 4) Confirme
    print("Colunas depois:", df.columns.tolist())

    # 5) Salve num novo arquivo
    df.to_excel("tabelaInicial.xlsx", index=False)
    print("Arquivo 'compras_sem_GH.xlsx' gerado.")


if __name__ == "__main__":
    main()
