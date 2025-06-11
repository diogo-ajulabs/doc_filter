import os
import pandas as pd
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json

# Carrega variáveis de ambiente (incluindo OPENAI_API_KEY)
load_dotenv()

# Template do prompt para análise de itens
ITEM_ANALYSIS_TEMPLATE = """Analise o seguinte item e forneça uma normalização seguindo ESTRITAMENTE as regras e exemplos abaixo:

REGRAS DE NORMALIZAÇÃO:

1. REGRA PRINCIPAL - Unificação de Medidas:
   - SEMPRE ignore TODAS as unidades de medida (ml, g, mg, kg, etc)
   - SEMPRE ignore TODAS as quantidades (10 comp, 20 comp, 3 ampolas, etc)
   - SEMPRE unifique itens com mesmo nome base, independente das medidas
   - Mantenha apenas especificações que mudam a função do produto (ex: FPS)

2. Para MARCAS:
   - Em cosméticos e materiais: converta para termo genérico
   - Em medicamentos: mantenha o nome específico

EXEMPLOS DE CONVERSÃO DE MARCAS:
- "Nescau" → "Achocolatado em Pó"
- "Nestogeno" → "Fórmula Infantil"
- "Neutrogena Sun Fresh" → "Protetor Solar"
- "Neosoro" → "Soro Fisiológico"
- "Novalgina" → "Dipirona"
- "Neutrofer" → "Sulfato Ferroso"
- "Ninho" → "Leite em Pó"
- "Neuleptil" → "Periciazina"
- "Nestonutri" → "Suplemento Alimentar"

EXEMPLOS DE UNIFICAÇÃO:
- "Soro fisiológico 100ml" e "Soro fisiológico 500ml" → "Soro fisiológico"
- "Dipirona 500mg" e "Dipirona 1g" → "Dipirona"
- "Papel A4 500 folhas" e "Papel A4 100 folhas" → "Papel A4"
- "Álcool 70% 1L" e "Álcool 70% 100ml" → "Álcool 70%"
- "Protetor Solar FPS 30 50ml" e "Protetor Solar FPS 30 200ml" → "Protetor Solar FPS 30"
- "Luva M cx 100un" e "Luva M cx 50un" → "Luva M"
- "Fralda G 20un" e "Fralda G 60un" → "Fralda G"

EXEMPLOS DE MARCAS:
- "Neutrogena Sun Fresh FPS 30 200ml" → "Protetor Solar FPS 30"
- "Novalgina 500mg 20 comp" → "Novalgina"
- "Nescau 400g" e "Nescau 1kg" → "achocolatado em pó"

IMPORTANTE:
- SEMPRE converta marcas para nomes genéricos dos produtos
- Analise o contexto para identificar o produto real
- Use termos técnicos/genéricos em vez de marcas
- Mantenha apenas características funcionais (FPS, Zero Lactose, etc)
- Use letras maiúsculas no início de cada palavra importante
- NUNCA faça distinção por quantidade ou unidade de medida
- Mantenha apenas características que alteram a função (tamanho P/M/G, FPS, etc)
- Unifique SEMPRE produtos com mesmo nome base
- Remova TODAS as informações de quantidade/volume/peso

Item para análise:
- Produto: {item}
- Grupo: {grupo}
- Já processado antes?: {ja_processado}
- Produtos similares já processados: {produtos_similares}

Responda APENAS no seguinte formato JSON:
{{"item_normalizado": "Nome Genérico Com Capitalize", "categoria": "categoria do item"}}
"""


class DocumentFilter:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-o4-mini",
            temperature=0,
        )

        self.prompt = PromptTemplate(
            input_variables=["item", "grupo", "ja_processado", "produtos_similares"],
            template=ITEM_ANALYSIS_TEMPLATE,
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt, verbose=True)

        # Dicionário para armazenar itens normalizados e suas variantes
        self.normalized_items = (
            {}
        )  # chave: item_normalizado, valor: {código, categoria, variantes[]}
        # Contador para geração de códigos sequenciais
        self.current_code = 25000

    def _get_next_code(self) -> int:
        """Retorna o próximo código disponível."""
        code = self.current_code
        self.current_code += 1
        return code

    def _get_similar_items(self, item: str) -> str:
        """Retorna itens similares já processados."""
        similares = []
        # Normaliza o item atual para comparação
        palavras_item = set(item.lower().split())

        # Remove palavras comuns que não ajudam na comparação
        palavras_para_remover = {
            "ml",
            "mg",
            "g",
            "kg",
            "comp",
            "comprimido",
            "cx",
            "c/",
            "com",
            "amp",
            "ampola",
            "gts",
            "gotas",
            "caps",
            "capsula",
            "un",
            "unidade",
            "pct",
            "pacote",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "0",
        }
        palavras_item = palavras_item - palavras_para_remover

        for norm_item, details in self.normalized_items.items():
            for variante in details["variantes"]:
                # Normaliza a variante para comparação
                palavras_variante = (
                    set(variante.lower().split()) - palavras_para_remover
                )

                # Calcula a interseção das palavras
                palavras_comuns = palavras_item.intersection(palavras_variante)

                # Se houver pelo menos 2 palavras em comum e uma delas for a primeira palavra
                if len(palavras_comuns) >= 2:
                    similares.append(f"{variante} → {norm_item}")
                # Ou se a primeira palavra for exatamente igual
                elif (list(palavras_item)[0] if palavras_item else "") == (
                    list(palavras_variante)[0] if palavras_variante else ""
                ):
                    similares.append(f"{variante} → {norm_item}")

        return json.dumps(similares, ensure_ascii=False) if similares else "[]"

    def _is_similar_to_existing(
        self, item: str, item_normalizado: str
    ) -> tuple[bool, str | None]:
        """
        Verifica se um item é similar a algum item já normalizado.
        Retorna uma tupla (é_similar, item_normalizado_existente).
        """
        palavras_item = set(item.lower().split())
        palavras_norm = set(item_normalizado.lower().split())

        # Remove palavras comuns que não ajudam na comparação
        palavras_para_remover = {
            "ml",
            "mg",
            "g",
            "kg",
            "comp",
            "comprimido",
            "cx",
            "c/",
            "com",
            "amp",
            "ampola",
            "gts",
            "gotas",
            "caps",
            "capsula",
            "un",
            "unidade",
            "pct",
            "pacote",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "0",
        }
        palavras_item = palavras_item - palavras_para_remover
        palavras_norm = palavras_norm - palavras_para_remover

        for norm_item, details in self.normalized_items.items():
            palavras_norm_existente = (
                set(norm_item.lower().split()) - palavras_para_remover
            )

            # Verifica similaridade com o item normalizado existente
            palavras_comuns_norm = palavras_norm.intersection(palavras_norm_existente)
            if len(palavras_comuns_norm) >= 2:
                return True, norm_item

            # Verifica similaridade com as variantes
            for variante in details["variantes"]:
                palavras_variante = (
                    set(variante.lower().split()) - palavras_para_remover
                )
                palavras_comuns = palavras_item.intersection(palavras_variante)

                if len(palavras_comuns) >= 2:
                    return True, norm_item
                # Ou se a primeira palavra for exatamente igual
                elif (list(palavras_item)[0] if palavras_item else "") == (
                    list(palavras_variante)[0] if palavras_variante else ""
                ):
                    return True, norm_item

        return False, None

    def _process_item(self, row: pd.Series) -> Dict[str, str]:
        """Processa um único item usando a chain do LangChain."""
        item = str(row["Produto/Serviço"])
        grupo = str(row["Grupo"])

        print(f"\n{'='*50}")
        print(f"Processando item: {item}")
        print(f"Grupo: {grupo}")
        print(f"{'='*50}")

        try:
            print("→ Verificando itens similares...")
            produtos_similares = self._get_similar_items(item)
            print(f"→ Produtos similares encontrados: {produtos_similares}")

            print("\n→ Enviando para processamento LLM...")
            response = self.chain.invoke(
                {
                    "item": item,
                    "grupo": grupo,
                    "ja_processado": str(
                        any(
                            item in details["variantes"]
                            for details in self.normalized_items.values()
                        )
                    ),
                    "produtos_similares": produtos_similares,
                }
            )

            if not response or not response.get("text"):
                raise ValueError("Resposta vazia do LLM")

            print("✓ Resposta recebida do LLM")
            print(f"→ Resposta bruta: {response['text']}")

            result = eval(response["text"])  # Converte a string JSON em dicionário
            print(f"→ Item normalizado: {result['item_normalizado']}")
            print(f"→ Categoria: {result['categoria']}")

            # Garante que o item_normalizado está com capitalize
            item_normalizado = " ".join(
                word.capitalize() for word in result["item_normalizado"].split()
            )

            # Verifica se já existe um item normalizado igual ou similar
            print("\n→ Verificando similaridade com itens existentes...")
            is_similar, existing_norm_item = self._is_similar_to_existing(
                item, item_normalizado
            )

            if is_similar and existing_norm_item:
                print(f"✓ Item similar encontrado: {existing_norm_item}")
                # Adiciona como variante do item similar existente
                self.normalized_items[existing_norm_item]["variantes"].append(item)
                return {
                    "codigo": self.normalized_items[existing_norm_item]["codigo"],
                    "item_normalizado": existing_norm_item,
                    "categoria": result["categoria"],
                    "grupo_original": grupo,
                    "unidade": row["UND"],
                    "variantes": self.normalized_items[existing_norm_item]["variantes"],
                }
            elif item_normalizado in self.normalized_items:
                print(f"✓ Item idêntico encontrado: {item_normalizado}")
                # Adiciona como variante do item normalizado existente
                self.normalized_items[item_normalizado]["variantes"].append(item)
                return {
                    "codigo": self.normalized_items[item_normalizado]["codigo"],
                    "item_normalizado": item_normalizado,
                    "categoria": result["categoria"],
                    "grupo_original": grupo,
                    "unidade": row["UND"],
                    "variantes": self.normalized_items[item_normalizado]["variantes"],
                }
            else:
                print("✓ Novo item único - criando novo registro")
                # Cria novo item normalizado
                novo_item = {
                    "codigo": self._get_next_code(),
                    "item_normalizado": item_normalizado,
                    "categoria": result["categoria"],
                    "grupo_original": grupo,
                    "unidade": row["UND"],
                    "variantes": [item],
                }
                self.normalized_items[item_normalizado] = {
                    "codigo": novo_item["codigo"],
                    "categoria": result["categoria"],
                    "variantes": [item],
                }
                return novo_item

        except Exception as e:
            print(f"\n❌ ERRO no processamento LLM:")
            print(f"Tipo de erro: {type(e).__name__}")
            print(f"Mensagem de erro: {str(e)}")

            if "insufficient_quota" in str(e):
                print("⚠️  ERRO DE QUOTA DA API OPENAI - Verifique sua chave API")
                raise Exception(
                    "Erro de quota da API OpenAI - Processamento interrompido"
                )
            elif "invalid_api_key" in str(e):
                print("⚠️  CHAVE API INVÁLIDA - Verifique sua chave API")
                raise Exception(
                    "Chave API OpenAI inválida - Processamento interrompido"
                )

            print("\n→ Usando fallback para processamento com erro...")
            # Em caso de erro, retorna o item original com capitalize
            item_normalizado = " ".join(word.capitalize() for word in item.split())
            return {
                "codigo": self._get_next_code(),
                "item_normalizado": item_normalizado,
                "categoria": "não classificado",
                "grupo_original": grupo,
                "unidade": row["UND"],
                "variantes": [item],
            }

    def processa_tabela(self, caminho: str) -> pd.DataFrame:
        """
        Processa uma tabela Excel contendo itens para normalização e categorização.

        Args:
            caminho: Caminho para o arquivo Excel contendo os itens.

        Returns:
            DataFrame com as colunas: codigo, item_normalizado, categoria, grupo_original, unidade, variantes
        """
        print("\n" + "=" * 50)
        print("INICIANDO PROCESSAMENTO DA TABELA")
        print("=" * 50)

        # Carrega o DataFrame
        print(f"\n→ Carregando arquivo: {caminho}")
        df = pd.read_excel(caminho)
        print(f"✓ Total de linhas carregadas: {len(df)}")

        # Limita aos 30 primeiros itens
        df = df.iloc[30:100]

        print(f"→ Limitando aos primeiros 30 itens")

        # Verifica se existem as colunas necessárias
        print("\n→ Verificando colunas necessárias...")
        colunas_necessarias = ["Código", "Grupo", "Produto/Serviço", "UND"]
        colunas_faltantes = [
            col for col in colunas_necessarias if col not in df.columns
        ]

        if colunas_faltantes:
            print("❌ ERRO: Colunas faltantes encontradas!")
            raise ValueError(
                f"O arquivo Excel deve conter as colunas: {', '.join(colunas_faltantes)}"
            )
        print("✓ Todas as colunas necessárias encontradas")

        # Processa cada item
        resultados = []
        itens_processados = set()  # Conjunto para controlar itens já processados
        itens_pulados = 0  # Contador de itens pulados
        total_itens = 0  # Contador total de itens
        erros_processamento = 0  # Contador de erros

        print("\nIniciando processamento de itens...")

        for _, row in df.iterrows():
            if pd.notna(row["Produto/Serviço"]):
                total_itens += 1
                item = str(row["Produto/Serviço"])

                # Verifica se o item já foi processado através de uma variante
                ja_processado = False
                item_normalizado_encontrado = None
                for norm_item, details in self.normalized_items.items():
                    if item in details["variantes"]:
                        ja_processado = True
                        item_normalizado_encontrado = norm_item
                        break

                if ja_processado:
                    itens_pulados += 1
                    print(f"\n→ Item pulado (já processado): {item}")
                    print(
                        f"  Já existe como variante de: {item_normalizado_encontrado}"
                    )
                else:
                    try:
                        resultado = self._process_item(row)
                        if (
                            resultado is not None
                            and resultado["item_normalizado"] not in itens_processados
                        ):
                            resultados.append(resultado)
                            itens_processados.add(resultado["item_normalizado"])
                    except Exception as e:
                        erros_processamento += 1
                        if "API OpenAI" in str(e):
                            raise e
                        print(f"\n❌ Erro grave no processamento: {str(e)}")

        # Cria o DataFrame final
        df_resultado = pd.DataFrame(resultados)

        # Reordena as colunas
        colunas_ordem = [
            "codigo",
            "item_normalizado",
            "categoria",
            "grupo_original",
            "unidade",
            "variantes",
        ]
        df_resultado = df_resultado[colunas_ordem]

        # Imprime o relatório de unificação
        print("\n" + "=" * 50)
        print("RELATÓRIO FINAL DE PROCESSAMENTO")
        print("=" * 50)
        print(f"Total de itens analisados: {total_itens}")
        print(f"Itens pulados (já processados): {itens_pulados}")
        print(f"Itens únicos após normalização: {len(resultados)}")
        print(f"Erros de processamento: {erros_processamento}")
        print("=" * 50)

        if erros_processamento > 0:
            print("\n⚠️  ATENÇÃO: Houve erros durante o processamento!")
        else:
            print("\n✓ Processamento concluído com sucesso!")

        print("\nRelatório de Unificação de Itens:")
        for item_norm, details in self.normalized_items.items():
            print(f"\nItem Normalizado: {item_norm}")
            print(f"Código: {details['codigo']}")
            print(f"Categoria: {details['categoria']}")
            print("Variantes encontradas:")
            for var in details["variantes"]:
                print(f"  - {var}")
            print("-" * 50)

        return df_resultado


def main():
    """Função principal para demonstração do uso."""
    filtro = DocumentFilter()

    # Processa a tabela
    df_limpo = filtro.processa_tabela("tabelaInicial.xlsx")

    # Exibe os resultados
    print("\nItens processados (unificados):")
    print(df_limpo)

    # Salva o resultado
    df_limpo.to_excel("itens_processados.xlsx", index=False)
    print("\nResultados salvos em 'itens_processados.xlsx'")


if __name__ == "__main__":
    main()
