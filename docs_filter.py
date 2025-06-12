import os
import pandas as pd
from typing import Dict, List, Set
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json
import asyncio
from collections import defaultdict
import aiohttp
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Carrega variáveis de ambiente (incluindo OPENAI_API_KEY)
load_dotenv()

# Template do prompt para análise de itens em lote
BATCH_ANALYSIS_TEMPLATE = """Você é um normalizador especializado em itens. Sua tarefa é analisar cada item da lista fornecida, removendo duplicatas e corrigindo metonímias.

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

Analise os seguintes itens:
{items}

Responda APENAS no seguinte formato JSON (array de objetos):
[
    {{"item_original": "Item Original 1", "item_normalizado": "Nome Genérico 1", "categoria": "CATEGORIA_1", "subcategoria": "Subcategoria 1"}},
    {{"item_original": "Item Original 2", "item_normalizado": "Nome Genérico 2", "categoria": "CATEGORIA_2", "subcategoria": "Subcategoria 2"}}
]"""


class OptimizedDocumentFilter:
    def __init__(self, batch_size=50, state_file="processing_state.json"):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo-16k",
            temperature=0,
        )
        self.batch_size = batch_size
        self.state_file = state_file
        self.prompt = PromptTemplate(
            input_variables=["items"],
            template=BATCH_ANALYSIS_TEMPLATE,
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.normalized_items = {}
        self.current_code = 25000
        self.vectorizer = None  # Será inicializado sob demanda
        self.similarity_threshold = 0.85  # Aumentado para ser mais rigoroso
        self.processed_items_cache = {}
        self.name_mapping = {}
        self.similarity_cache = {}

        # Carrega o estado inicial
        self._load_state()

    def _load_state(self):
        """Carrega o estado do processamento do arquivo JSON."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                self.current_code = state.get("next_code", 25000)
                self.last_processed_index = state.get("last_processed_index", -1)
                print(
                    f"✓ Estado carregado: Último código: {self.current_code-1}, Último índice: {self.last_processed_index}"
                )
            else:
                self.last_processed_index = -1
                print("→ Arquivo de estado não encontrado. Iniciando do começo.")
        except Exception as e:
            print(f"❌ Erro ao carregar estado: {str(e)}")
            self.last_processed_index = -1

    def _save_state(self):
        """Salva o estado atual do processamento."""
        try:
            state = {
                "next_code": self.current_code,
                "last_processed_index": self.last_processed_index,
                "timestamp": pd.Timestamp.now().isoformat(),
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            print(
                f"✓ Estado salvo: Último código: {self.current_code-1}, Último índice: {self.last_processed_index}"
            )
        except Exception as e:
            print(f"❌ Erro ao salvar estado: {str(e)}")

    def _load_existing_results(self, output_file):
        """Carrega resultados existentes do arquivo Excel."""
        try:
            if os.path.exists(output_file):
                df_existing = pd.read_excel(output_file)
                # Reconstrói o estado interno baseado nos resultados existentes
                for _, row in df_existing.iterrows():
                    # Converte a string JSON em lista
                    try:
                        variantes_info = json.loads(
                            row["variantes_info"].replace("'", '"')
                        )
                    except:
                        variantes_info = []

                    self.normalized_items[row["item_normalizado"]] = {
                        "codigo": row["codigo"],
                        "categoria": row["categoria"],
                        "subcategoria": row["subcategoria"],
                        "variantes_info": variantes_info,
                    }
                    # Atualiza o mapeamento de nomes
                    normalized_name = self._normalize_text(row["item_normalizado"])
                    self.name_mapping[normalized_name] = row["item_normalizado"]
                print(
                    f"✓ Carregados {len(df_existing)} itens processados anteriormente"
                )
                return df_existing
            return pd.DataFrame(
                columns=[
                    "codigo",
                    "item_normalizado",
                    "categoria",
                    "subcategoria",
                    "grupo_original",
                    "unidade",
                    "variantes",
                    "variantes_info",
                ]
            )
        except Exception as e:
            print(f"❌ Erro ao carregar resultados existentes: {str(e)}")
            return pd.DataFrame(
                columns=[
                    "codigo",
                    "item_normalizado",
                    "categoria",
                    "subcategoria",
                    "grupo_original",
                    "unidade",
                    "variantes",
                    "variantes_info",
                ]
            )

    def _normalize_text(self, text: str) -> str:
        """Normaliza o texto removendo caracteres especiais e convertendo para minúsculo."""
        import re

        if not text:
            return ""

        # Remove caracteres especiais e números
        text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", "", text)
        # Converte para minúsculo
        text = text.lower()
        # Remove espaços extras
        text = " ".join(text.split())
        # Remove palavras comuns que não ajudam na comparação
        stop_words = [
            "de",
            "da",
            "do",
            "das",
            "dos",
            "e",
            "com",
            "para",
            "em",
            "mg",
            "ml",
            "g",
            "kg",
        ]
        words = text.split()
        words = [w for w in words if w not in stop_words]
        text = " ".join(words)
        return text

    def _initialize_vectorizer(self, items):
        """Inicializa ou atualiza o vectorizer com novos itens."""
        if not self.vectorizer:
            self.vectorizer = TfidfVectorizer(
                analyzer="word",
                stop_words=[
                    "ml",
                    "mg",
                    "g",
                    "kg",
                    "comp",
                    "cx",
                    "c/",
                    "amp",
                    "un",
                    "pct",
                ],
                ngram_range=(1, 2),  # Considera unigramas e bigramas
            )

        # Normaliza todos os itens antes de treinar o vectorizer
        normalized_items = [self._normalize_text(item) for item in items]
        self.vectorizer.fit(normalized_items)

    def _calculate_similarity(self, item1: str, item2: str) -> float:
        """Calcula a similaridade entre dois itens usando TF-IDF e cosine similarity."""
        # Verifica o cache primeiro
        cache_key = f"{item1}|{item2}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        # Normaliza os textos
        norm1 = self._normalize_text(item1)
        norm2 = self._normalize_text(item2)

        # Se algum dos textos está vazio após normalização, retorna 0
        if not norm1 or not norm2:
            return 0.0

        # Se os textos normalizados são idênticos, retorna 1
        if norm1 == norm2:
            return 1.0

        try:
            # Vetoriza os textos
            vectors = self.vectorizer.transform([norm1, norm2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

            # Armazena no cache
            self.similarity_cache[cache_key] = similarity
            return similarity
        except Exception as e:
            print(f"Erro ao calcular similaridade: {str(e)}")
            return 0.0

    def _find_existing_item(self, item_normalizado: str, categoria: str) -> str | None:
        """
        Procura por um item existente que seja similar ao item normalizado.
        Retorna o item existente se encontrado, None caso contrário.
        """
        # Primeiro, verifica pelo nome exato normalizado
        normalized_name = self._normalize_text(item_normalizado)
        if normalized_name in self.name_mapping:
            return self.name_mapping[normalized_name]

        # Se não encontrou pelo nome exato, procura por similaridade
        if len(self.normalized_items) > 0:
            try:
                # Encontra o item mais similar na mesma categoria
                max_similarity = 0
                most_similar_item = None

                for existing_item, details in self.normalized_items.items():
                    if details["categoria"] == categoria:
                        similarity = self._calculate_similarity(
                            item_normalizado, existing_item
                        )
                        if (
                            similarity > max_similarity
                            and similarity > self.similarity_threshold
                        ):
                            max_similarity = similarity
                            most_similar_item = existing_item

                return most_similar_item

            except Exception as e:
                print(f"Erro ao procurar item similar: {str(e)}")
                return None

        return None

    def _get_next_code(self) -> int:
        code = self.current_code
        self.current_code += 1
        return code

    async def _process_batch(self, batch: List[pd.Series]) -> List[Dict]:
        """Processa um lote de itens em paralelo."""
        # Extrai todos os itens do lote para treinar o vectorizer
        batch_items = [str(row["Produto/Serviço"]) for row in batch]
        existing_items = list(self.normalized_items.keys())
        all_items = batch_items + existing_items

        # Inicializa ou atualiza o vectorizer com todos os itens
        self._initialize_vectorizer(all_items)

        items_json = json.dumps(
            [
                {"item": str(row["Produto/Serviço"]), "grupo": str(row["Grupo"])}
                for row in batch
            ],
            ensure_ascii=False,
        )

        try:
            response = await asyncio.to_thread(self.chain.invoke, {"items": items_json})

            if not response or not response.get("text"):
                raise ValueError("Resposta vazia do LLM")

            results = json.loads(response["text"])
            processed_results = []

            # Processa cada item do lote
            for idx, result in enumerate(results):
                row = batch[idx]
                item = str(row["Produto/Serviço"])
                codigo_original = str(row["Código"])

                # Verifica cache
                if item in self.processed_items_cache:
                    processed_results.append(self.processed_items_cache[item])
                    continue

                # Verifica se já existe um item similar
                existing_item = self._find_existing_item(
                    result["item_normalizado"], result["categoria"]
                )

                if existing_item:
                    # Adiciona como variante do item existente
                    details = self.normalized_items[existing_item]
                    variantes_info = details["variantes_info"]
                    if isinstance(variantes_info, str):
                        try:
                            variantes_info = json.loads(
                                variantes_info.replace("'", '"')
                            )
                        except:
                            variantes_info = []

                    # Verifica se o item já existe como variante
                    item_exists = False
                    for variante in variantes_info:
                        if (
                            self._calculate_similarity(variante.get("item", ""), item)
                            > self.similarity_threshold
                        ):
                            item_exists = True
                            break

                    if not item_exists:
                        variante_info = {
                            "item": item,
                            "codigo_original": codigo_original,
                            "grupo_original": str(row["Grupo"]),
                        }
                        variantes_info.append(variante_info)
                        details["variantes_info"] = variantes_info

                    processed_result = {
                        "codigo": details["codigo"],
                        "item_normalizado": existing_item,
                        "categoria": details["categoria"],
                        "subcategoria": details["subcategoria"],
                        "grupo_original": str(row["Grupo"]),
                        "unidade": row["UND"],
                        "variantes": [v.get("item") for v in variantes_info],
                        "variantes_info": json.dumps(
                            variantes_info, ensure_ascii=False
                        ),
                    }
                else:
                    # Cria novo item
                    variantes_info = [
                        {
                            "item": item,
                            "codigo_original": codigo_original,
                            "grupo_original": str(row["Grupo"]),
                        }
                    ]

                    novo_item = {
                        "codigo": self._get_next_code(),
                        "item_normalizado": result["item_normalizado"],
                        "categoria": result["categoria"],
                        "subcategoria": result["subcategoria"],
                        "grupo_original": str(row["Grupo"]),
                        "unidade": row["UND"],
                        "variantes": [item],
                        "variantes_info": json.dumps(
                            variantes_info, ensure_ascii=False
                        ),
                    }

                    # Adiciona aos itens normalizados
                    self.normalized_items[result["item_normalizado"]] = {
                        "codigo": novo_item["codigo"],
                        "categoria": result["categoria"],
                        "subcategoria": result["subcategoria"],
                        "variantes_info": variantes_info,
                    }

                    # Adiciona ao mapeamento de nomes
                    normalized_name = self._normalize_text(result["item_normalizado"])
                    self.name_mapping[normalized_name] = result["item_normalizado"]

                    processed_result = novo_item

                self.processed_items_cache[item] = processed_result
                processed_results.append(processed_result)

            return processed_results

        except Exception as e:
            print(f"\n❌ ERRO no processamento do lote:")
            print(f"Tipo de erro: {type(e).__name__}")
            print(f"Mensagem de erro: {str(e)}")
            raise e

    async def processa_tabela_async(
        self,
        caminho: str,
        output_file: str = "itens_processados_incremental.xlsx",
        items_to_process: int = 100,
    ) -> pd.DataFrame:
        """Processa uma tabela Excel de forma assíncrona e incremental."""
        print("\n" + "=" * 50)
        print("INICIANDO PROCESSAMENTO DA TABELA (VERSÃO INCREMENTAL)")
        print("=" * 50)

        # Carrega o DataFrame original
        print(f"\n→ Carregando arquivo: {caminho}")
        df_original = pd.read_excel(caminho)
        total_itens_original = len(df_original)
        print(f"✓ Total de linhas carregadas: {total_itens_original}")

        # Carrega resultados existentes
        df_resultado = self._load_existing_results(output_file)

        # Determina o intervalo de processamento
        start_index = self.last_processed_index + 1
        end_index = min(start_index + items_to_process, total_itens_original)

        if start_index >= total_itens_original:
            print("\n✓ Todos os itens já foram processados!")
            return df_resultado

        # Seleciona o intervalo de itens para processar
        df = df_original.iloc[start_index:end_index].copy()
        total_itens = len(df)

        print(f"\n→ Processando do índice {start_index} até {end_index-1}")
        print(f"✓ Itens selecionados para processamento: {total_itens}")
        print(f"✓ Itens restantes após este lote: {total_itens_original - end_index}")

        # Processa em lotes
        resultados = []
        batches = [
            df[i : i + self.batch_size] for i in range(0, len(df), self.batch_size)
        ]

        with tqdm(total=len(batches), desc="Processando lotes") as pbar:
            for batch_idx, batch_df in enumerate(batches):
                try:
                    batch_results = await self._process_batch(
                        batch_df.to_dict("records")
                    )
                    resultados.extend(batch_results)

                    # Atualiza o último índice processado
                    self.last_processed_index = (
                        start_index + (batch_idx + 1) * self.batch_size - 1
                    )
                    if batch_idx == len(batches) - 1:  # Último lote
                        self.last_processed_index = end_index - 1

                    # Salva incrementalmente
                    df_increment = pd.DataFrame(batch_results)
                    df_resultado = pd.concat(
                        [df_resultado, df_increment], ignore_index=True
                    )
                    df_resultado.to_excel(output_file, index=False)

                    # Salva o estado
                    self._save_state()

                    pbar.update(1)
                except Exception as e:
                    print(f"\n❌ Erro no processamento do lote {batch_idx}: {str(e)}")
                    # Salva o progresso mesmo em caso de erro
                    if resultados:
                        df_increment = pd.DataFrame(resultados)
                        df_resultado = pd.concat(
                            [df_resultado, df_increment], ignore_index=True
                        )
                        df_resultado.to_excel(output_file, index=False)
                    self._save_state()
                    raise e

        print("\nRELATÓRIO FINAL:")
        print(f"Total de itens processados nesta execução: {total_itens}")
        print(f"Total de itens únicos após normalização: {len(self.normalized_items)}")
        print(f"Próximo índice a ser processado: {self.last_processed_index + 1}")

        return df_resultado


def main():
    """Função principal para demonstração do uso."""
    filtro = OptimizedDocumentFilter(batch_size=50)

    # Processa a tabela de forma assíncrona e incremental
    df_limpo = asyncio.run(
        filtro.processa_tabela_async(
            caminho="tabelaInicial.xlsx",
            output_file="itens_processados_incremental.xlsx",
            items_to_process=100,
        )
    )

    print("\nResultados salvos em 'itens_processados_incremental.xlsx'")


if __name__ == "__main__":
    main()
