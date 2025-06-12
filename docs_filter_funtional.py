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
ITEM_ANALYSIS_TEMPLATE = """Você é um normalizador especializado em itens. Sua tarefa é analisar cada item, removendo duplicatas e corrigindo metonímias (quando a marca aparece em lugar do nome genérico).

REGRAS DE NORMALIZAÇÃO:

1. REGRA PRINCIPAL - Unificação e Normalização:
   - SEMPRE ignore TODAS as unidades de medida (ml, g, mg, kg, etc)
   - SEMPRE ignore TODAS as quantidades (10 comp, 20 comp, 3 ampolas, etc)
   - SEMPRE unifique itens com mesmo nome base, independente das medidas
   - NUNCA diferencie singular e plural (ex: "Fralda G" = "Fraldas G")
   - Mantenha apenas especificações que mudam a função do produto (ex: FPS, Zero Lactose, tamanho P/M/G)

2. REGRAS PARA MEDICAMENTOS:
   - Se o item menciona o PRINCÍPIO ATIVO, normalize para o NOME DA SUBSTÂNCIA
     Ex: "Dipirona 500mg" → "Dipirona"
   - Se o item é uma MARCA COMERCIAL, substitua pela SUBSTÂNCIA GENÉRICA
     Ex: "Novalgina" → "Dipirona"
     Ex: "Xarelto" → "Rivaroxabana"
   - SEMPRE priorize o princípio ativo sobre a marca comercial

3. REGRAS PARA COSMÉTICOS E MATERIAIS:
   - SEMPRE converta marcas para o termo genérico do produto
   - Analise o contexto para identificar o produto real
   - Use termos técnicos/genéricos em vez de marcas
   - Mantenha apenas características funcionais (FPS, Zero Lactose, etc)

4. REGRAS DE CATEGORIZAÇÃO:
   Categorias principais disponíveis:
   - MEDICAMENTO: Qualquer item farmacêutico com princípio ativo
   - COSMÉTICO: Produtos de beleza e cuidados com a pele
   - MATERIAL_MÉDICO: Materiais hospitalares e de enfermagem
   - HIGIENE: Produtos de higiene pessoal
   - ALIMENTO: Produtos alimentícios
   - MATERIAL_ESCRITÓRIO: Materiais de escritório
   - LIMPEZA: Produtos de limpeza
   - HORTIFRUTI: Frutas, verduras e legumes

   Subcategorias (exemplos):
   - MEDICAMENTO: Analgésico, Anti-inflamatório, Antibiótico, etc.
   - COSMÉTICO: Protetor Solar, Hidratante, Maquiagem, etc.
   - MATERIAL_MÉDICO: Curativo, Seringa, Luva, etc.
   - HIGIENE: Sabonete, Shampoo, Pasta Dental, etc.
   - ALIMENTO: Cereal, Laticínio, Carne, etc.
   - MATERIAL_ESCRITÓRIO: Papel, Caneta, Pilha, etc.
   - LIMPEZA: Detergente, Desinfetante, etc.
   - HORTIFRUTI: Fruta, Verdura, Legume, etc.

EXEMPLOS DE SAÍDA ESPERADA:
- "Neutrogena Sun Fresh FPS 30 200ml" → 
  {{"item_normalizado": "Protetor Solar FPS 30", "categoria": "COSMÉTICO", "subcategoria": "Protetor Solar"}}

- "Novalgina 500mg 20 comp" → 
  {{"item_normalizado": "Dipirona", "categoria": "MEDICAMENTO", "subcategoria": "Analgésico"}}

- "Nescau 400g" → 
  {{"item_normalizado": "Achocolatado em Pó", "categoria": "ALIMENTO", "subcategoria": "Achocolatado"}}

Item para análise:
- Produto: {item}
- Grupo: {grupo}
- Já processado antes?: {ja_processado}
- Produtos similares já processados: {produtos_similares}

Responda APENAS no seguinte formato JSON:
{{"item_normalizado": "Nome Genérico Com Capitalize", "categoria": "CATEGORIA_PRINCIPAL", "subcategoria": "Subcategoria"}}
"""


class DocumentFilter:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
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
            "gramas",
            "gr",
            "grs",
            "caixa",
            "bisnaga",
            "frasco",
            "embalagem",
            "cartela",
            "de",
            "da",
            "do",
            "para",
            "com",
            "em",
        }
        palavras_item = palavras_item - palavras_para_remover

        # Primeiro, tenta identificar a categoria do item atual usando o LLM
        try:
            response = self.chain.invoke(
                {
                    "item": item,
                    "grupo": "",
                    "ja_processado": "false",
                    "produtos_similares": "[]",
                }
            )
            if response and response.get("text"):
                try:
                    result = json.loads(response["text"])
                except json.JSONDecodeError:
                    cleaned_text = response["text"].strip().replace("\n", "")
                    result = eval(cleaned_text)

                categoria_atual = result["categoria"]
                subcategoria_atual = result["subcategoria"]

                # Agora procura por itens similares apenas na mesma categoria/subcategoria
                for norm_item, details in self.normalized_items.items():
                    if (
                        details["categoria"] == categoria_atual
                        and details["subcategoria"] == subcategoria_atual
                    ):

                        for variante in details["variantes_info"]:
                            palavras_variante = (
                                set(variante["item"].lower().split())
                                - palavras_para_remover
                            )
                            palavras_comuns = palavras_item.intersection(
                                palavras_variante
                            )

                            if len(palavras_comuns) >= 2:
                                primeira_palavra_igual = (
                                    list(palavras_item)[0] if palavras_item else ""
                                ) == (
                                    list(palavras_variante)[0]
                                    if palavras_variante
                                    else ""
                                )

                                if primeira_palavra_igual:
                                    similares.append(
                                        f"{variante['item']} → {norm_item} ({details['categoria']}/{details['subcategoria']})"
                                    )

        except Exception:
            # Se falhar ao obter a categoria via LLM, usa a lógica antiga
            for norm_item, details in self.normalized_items.items():
                for variante in details["variantes_info"]:
                    palavras_variante = (
                        set(variante["item"].lower().split()) - palavras_para_remover
                    )
                    palavras_comuns = palavras_item.intersection(palavras_variante)

                    if len(palavras_comuns) >= 2:
                        primeira_palavra_igual = (
                            list(palavras_item)[0] if palavras_item else ""
                        ) == (list(palavras_variante)[0] if palavras_variante else "")

                        if primeira_palavra_igual:
                            similares.append(f"{variante['item']} → {norm_item}")

        return json.dumps(similares, ensure_ascii=False) if similares else "[]"

    def _is_similar_to_existing(
        self, item: str, item_normalizado: str, categoria: str, subcategoria: str
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
            "gramas",
            "gr",
            "grs",
            "caixa",
            "bisnaga",
            "frasco",
            "embalagem",
            "cartela",
            "de",
            "da",
            "do",
            "para",
            "com",
            "em",
        }
        palavras_item = palavras_item - palavras_para_remover
        palavras_norm = palavras_norm - palavras_para_remover

        for norm_item, details in self.normalized_items.items():
            # Primeiro verifica se as categorias são compatíveis
            if (
                categoria != details["categoria"]
                or subcategoria != details["subcategoria"]
            ):
                continue

            palavras_norm_existente = (
                set(norm_item.lower().split()) - palavras_para_remover
            )

            # Verifica similaridade com o item normalizado existente
            palavras_comuns_norm = palavras_norm.intersection(palavras_norm_existente)

            # Se houver pelo menos 2 palavras significativas em comum
            if len(palavras_comuns_norm) >= 2:
                primeira_palavra_igual = (
                    list(palavras_item)[0] if palavras_item else ""
                ) == (
                    list(palavras_norm_existente)[0] if palavras_norm_existente else ""
                )

                if primeira_palavra_igual:
                    return True, norm_item

            # Verifica similaridade com as variantes
            for variante in details["variantes_info"]:
                palavras_variante = (
                    set(variante["item"].lower().split()) - palavras_para_remover
                )
                palavras_comuns = palavras_item.intersection(palavras_variante)

                if len(palavras_comuns) >= 2:
                    primeira_palavra_igual = (
                        list(palavras_item)[0] if palavras_item else ""
                    ) == (list(palavras_variante)[0] if palavras_variante else "")

                    if primeira_palavra_igual:
                        return True, norm_item

        return False, None

    def _process_item(self, row: pd.Series) -> Dict[str, str]:
        """Processa um único item usando a chain do LangChain."""
        item = str(row["Produto/Serviço"])
        grupo = str(row["Grupo"])
        codigo_original = str(row["Código"])

        print(f"\n{'='*50}")
        print(f"Processando item: {item}")
        print(f"Código original: {codigo_original}")
        print(f"Grupo: {grupo}")
        print(f"{'='*50}")

        try:
            # Verifica se o item já existe como variante
            for norm_item, details in self.normalized_items.items():
                if item in [v["item"] for v in details["variantes_info"]]:
                    print(f"→ Item já existe como variante de: {norm_item}")
                    return {
                        "codigo": details["codigo"],
                        "item_normalizado": norm_item,
                        "categoria": details["categoria"],
                        "subcategoria": details["subcategoria"],
                        "grupo_original": grupo,
                        "unidade": row["UND"],
                        "variantes": [v["item"] for v in details["variantes_info"]],
                        "variantes_info": details["variantes_info"],
                    }

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
                            item in [v["item"] for v in details["variantes_info"]]
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

            try:
                result = json.loads(response["text"])
            except json.JSONDecodeError:
                cleaned_text = response["text"].strip().replace("\n", "")
                result = eval(cleaned_text)

            print(f"→ Item normalizado: {result['item_normalizado']}")
            print(f"→ Categoria: {result['categoria']}")
            print(f"→ Subcategoria: {result['subcategoria']}")

            # Garante que o item_normalizado está com capitalize
            item_normalizado = " ".join(
                word.capitalize() for word in result["item_normalizado"].split()
            )

            # Verifica se já existe um item normalizado igual ou similar
            print("\n→ Verificando similaridade com itens existentes...")
            is_similar, existing_norm_item = self._is_similar_to_existing(
                item, item_normalizado, result["categoria"], result["subcategoria"]
            )

            # Cria o objeto de informação da variante
            variante_info = {
                "item": item,
                "codigo_original": codigo_original,
                "grupo_original": grupo,
            }

            if is_similar and existing_norm_item:
                print(f"✓ Item similar encontrado: {existing_norm_item}")
                # Adiciona como variante do item similar existente
                if item not in [
                    v["item"]
                    for v in self.normalized_items[existing_norm_item]["variantes_info"]
                ]:
                    self.normalized_items[existing_norm_item]["variantes_info"].append(
                        variante_info
                    )
                return {
                    "codigo": self.normalized_items[existing_norm_item]["codigo"],
                    "item_normalizado": existing_norm_item,
                    "categoria": result["categoria"],
                    "subcategoria": result["subcategoria"],
                    "grupo_original": grupo,
                    "unidade": row["UND"],
                    "variantes": [
                        v["item"]
                        for v in self.normalized_items[existing_norm_item][
                            "variantes_info"
                        ]
                    ],
                    "variantes_info": self.normalized_items[existing_norm_item][
                        "variantes_info"
                    ],
                }
            elif item_normalizado in self.normalized_items:
                print(f"✓ Item idêntico encontrado: {item_normalizado}")
                # Adiciona como variante do item normalizado existente
                if item not in [
                    v["item"]
                    for v in self.normalized_items[item_normalizado]["variantes_info"]
                ]:
                    self.normalized_items[item_normalizado]["variantes_info"].append(
                        variante_info
                    )
                return {
                    "codigo": self.normalized_items[item_normalizado]["codigo"],
                    "item_normalizado": item_normalizado,
                    "categoria": result["categoria"],
                    "subcategoria": result["subcategoria"],
                    "grupo_original": grupo,
                    "unidade": row["UND"],
                    "variantes": [
                        v["item"]
                        for v in self.normalized_items[item_normalizado][
                            "variantes_info"
                        ]
                    ],
                    "variantes_info": self.normalized_items[item_normalizado][
                        "variantes_info"
                    ],
                }
            else:
                print("✓ Novo item único - criando novo registro")
                # Cria novo item normalizado
                novo_item = {
                    "codigo": self._get_next_code(),
                    "item_normalizado": item_normalizado,
                    "categoria": result["categoria"],
                    "subcategoria": result["subcategoria"],
                    "grupo_original": grupo,
                    "unidade": row["UND"],
                    "variantes": [item],
                    "variantes_info": [variante_info],
                }
                self.normalized_items[item_normalizado] = {
                    "codigo": novo_item["codigo"],
                    "categoria": result["categoria"],
                    "subcategoria": result["subcategoria"],
                    "variantes_info": [variante_info],
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
            variante_info = {
                "item": item,
                "codigo_original": codigo_original,
                "grupo_original": grupo,
            }
            return {
                "codigo": self._get_next_code(),
                "item_normalizado": item_normalizado,
                "categoria": "NÃO CLASSIFICADO",
                "subcategoria": "Não Classificado",
                "grupo_original": grupo,
                "unidade": row["UND"],
                "variantes": [item],
                "variantes_info": [variante_info],
            }

    def processa_tabela(self, caminho: str) -> pd.DataFrame:
        """
        Processa uma tabela Excel contendo itens para normalização e categorização.

        Args:
            caminho: Caminho para o arquivo Excel contendo os itens.

        Returns:
            DataFrame com as colunas: codigo, item_normalizado, categoria, grupo_original, unidade, variantes, variantes_info
        """
        print("\n" + "=" * 50)
        print("INICIANDO PROCESSAMENTO DA TABELA")
        print("=" * 50)

        # Carrega o DataFrame
        print(f"\n→ Carregando arquivo: {caminho}")
        df = pd.read_excel(caminho)
        total_itens_original = len(df)
        print(f"✓ Total de linhas carregadas: {total_itens_original}")

        # Limita aos 30 primeiros itens
        # df = df.head(15)
        print(f"\n→ Limitando processamento aos primeiros 30 itens")
        print(f"✓ Itens selecionados para processamento: {len(df)}")
        print(f"✓ Itens não processados: {total_itens_original - len(df)}")

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
        itens_pulados = []  # Lista para armazenar detalhes dos itens pulados
        total_itens = 0  # Contador total de itens
        erros_processamento = []  # Lista para armazenar detalhes dos erros

        print("\nIniciando processamento de itens...")

        for _, row in df.iterrows():
            if pd.notna(row["Produto/Serviço"]):
                total_itens += 1
                item = str(row["Produto/Serviço"])
                codigo_original = str(row["Código"])

                # Verifica se o item já foi processado através de uma variante
                ja_processado = False
                item_normalizado_encontrado = None
                for norm_item, details in self.normalized_items.items():
                    if item in [v["item"] for v in details["variantes_info"]]:
                        ja_processado = True
                        item_normalizado_encontrado = norm_item
                        itens_pulados.append(
                            {
                                "item_original": item,
                                "codigo_original": codigo_original,
                                "normalizado_como": norm_item,
                                "codigo_normalizado": details["codigo"],
                                "categoria": details["categoria"],
                                "subcategoria": details["subcategoria"],
                            }
                        )
                        break

                if not ja_processado:
                    try:
                        resultado = self._process_item(row)
                        if resultado is not None:
                            # Verifica se já existe um item normalizado com o mesmo nome
                            item_norm = resultado["item_normalizado"]
                            if item_norm not in itens_processados:
                                resultados.append(resultado)
                                itens_processados.add(item_norm)
                            else:
                                # Se já existe, adiciona como variante ao item existente
                                for res in resultados:
                                    if res["item_normalizado"] == item_norm:
                                        if item not in [
                                            v["item"] for v in res["variantes_info"]
                                        ]:
                                            variante_info = {
                                                "item": item,
                                                "codigo_original": codigo_original,
                                                "grupo_original": str(row["Grupo"]),
                                                "categoria": res["categoria"],
                                                "subcategoria": res["subcategoria"],
                                            }
                                            res["variantes_info"].append(variante_info)
                                            res["variantes"].append(item)
                                        itens_pulados.append(
                                            {
                                                "item_original": item,
                                                "codigo_original": codigo_original,
                                                "normalizado_como": item_norm,
                                                "codigo_normalizado": res["codigo"],
                                                "categoria": res["categoria"],
                                                "subcategoria": res["subcategoria"],
                                            }
                                        )
                                        break
                    except Exception as e:
                        erros_processamento.append(
                            {
                                "item": item,
                                "codigo_original": codigo_original,
                                "erro": str(e),
                                "tipo": type(e).__name__,
                            }
                        )
                        if "API OpenAI" in str(e):
                            raise e
                        print(f"\n❌ Erro no processamento do item '{item}': {str(e)}")

        # Cria o DataFrame final
        df_resultado = pd.DataFrame(resultados)

        # Reordena as colunas
        colunas_ordem = [
            "codigo",
            "item_normalizado",
            "categoria",
            "subcategoria",
            "grupo_original",
            "unidade",
            "variantes",
            "variantes_info",
        ]
        df_resultado = df_resultado[colunas_ordem]

        # Imprime o relatório detalhado
        print("\n" + "=" * 50)
        print("RELATÓRIO DETALHADO DE PROCESSAMENTO")
        print("=" * 50)
        print(f"Total de itens analisados: {total_itens}")
        print(f"Itens únicos após normalização: {len(resultados)}")
        print(f"Itens pulados (já processados): {len(itens_pulados)}")
        print(f"Erros de processamento: {len(erros_processamento)}")

        if itens_pulados:
            print("\nDETALHES DOS ITENS PULADOS:")
            print("-" * 50)
            for item in itens_pulados:
                print(
                    f"Item original: {item['item_original']} (Código: {item['codigo_original']})"
                )
                print(f"→ Normalizado como: {item['normalizado_como']}")
                print(f"→ Código normalizado: {item['codigo_normalizado']}")
                print(f"→ Categoria: {item['categoria']}")
                print(f"→ Subcategoria: {item['subcategoria']}")
                print("-" * 30)

        if erros_processamento:
            print("\nDETALHES DOS ERROS DE PROCESSAMENTO:")
            print("-" * 50)
            for erro in erros_processamento:
                print(f"Item: {erro['item']} (Código: {erro['codigo_original']})")
                print(f"Tipo de erro: {erro['tipo']}")
                print(f"Mensagem: {erro['erro']}")
                print("-" * 30)

        print("\nRELATÓRIO DE UNIFICAÇÃO DE ITENS:")
        print("=" * 50)
        for item_norm, details in self.normalized_items.items():
            print(f"\nItem Normalizado: {item_norm}")
            print(f"Código: {details['codigo']}")
            print(f"Categoria: {details['categoria']}")
            print(f"Subcategoria: {details['subcategoria']}")
            print("Variantes encontradas:")
            for var in details["variantes_info"]:
                print(f"  - {var['item']} (Código original: {var['codigo_original']})")
                print(f"    Grupo original: {var['grupo_original']}")
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
