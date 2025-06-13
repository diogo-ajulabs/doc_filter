import os
import pandas as pd
from typing import Dict, List, Set, TypedDict
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
import json
import asyncio
from collections import defaultdict
import aiohttp
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import _thread
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

# Carrega variáveis de ambiente (incluindo OPENAI_API_KEY)
load_dotenv()


# Define o modelo Pydantic para validação
class ItemAnalise(BaseModel):
    item_original: str = Field(
        description="O item exatamente como foi fornecido",
        default="Item não identificado",
    )
    item_normalizado: str = Field(
        description="O nome normalizado do item", default="Item não identificado"
    )
    categoria: str = Field(
        description="Uma das categorias principais listadas", default="OUTROS_SERVICOS"
    )
    subcategoria: str = Field(
        description="Uma subcategoria específica para o item", default="Itens Diversos"
    )

    @field_validator("categoria")
    @classmethod
    def validate_categoria(cls, v: str) -> str:
        if not v:
            return "OUTROS_SERVICOS"
        if not v.isupper() or " " in v:
            v = v.upper().replace(" ", "_")
        return v

    @field_validator("subcategoria")
    @classmethod
    def validate_subcategoria(cls, v: str | None) -> str:
        if not v:
            return "Itens Diversos"
        return v

    def model_dump(self) -> Dict:
        data = super().model_dump()
        # Garante que subcategoria nunca seja None
        if not data["subcategoria"]:
            # Define subcategoria padrão baseada na categoria
            categoria_to_subcategoria = {
                "MEDICAMENTO": "Medicamentos em Geral",
                "MATERIAL_MEDICO": "Materiais Hospitalares",
                "COSMETICO": "Produtos Cosméticos",
                "HIGIENE": "Produtos de Higiene",
                "ALIMENTO": "Gêneros Alimentícios",
                "MATERIAL_ESCRITORIO": "Materiais de Escritório",
                "LIMPEZA": "Produtos de Limpeza",
                "OUTROS_SERVICOS": "Itens Diversos",
            }
            data["subcategoria"] = categoria_to_subcategoria.get(
                data["categoria"], "Itens Diversos"
            )
        return data


class BatchAnaliseResponse(BaseModel):
    items: List[ItemAnalise]


# Template do prompt para análise de itens em lote
BATCH_ANALYSIS_TEMPLATE = """Analise os itens e retorne um JSON no formato:
{
    "items": [
        {
            "item_original": "Item exato",
            "item_normalizado": "Nome genérico",
            "categoria": "CATEGORIA_MAIUSCULA",
            "subcategoria": "Tipo específico"
        }
    ]
}

Regras:
1. Ignore medidas e quantidades
2. Use nomes genéricos
3. Categorias em MAIÚSCULAS_COM_UNDERSCORE
4. IMPORTANTE: Remova nomes de marcas e substitua por termos genéricos. Exemplos:
   - "Nescau" -> "Achocolatado em pó"
   - "Nivea hidratante" -> "Hidratante corporal"
   - "Bombril" -> "Esponja de aço"
   - "Cotonetes" -> "Hastes flexíveis"
   - "Band-aid" -> "Curativo adesivo"
   - "Gillette" -> "Lâmina de barbear"
   - "Xerox" -> "Cópia reprográfica"
   - "Post-it" -> "Bloco adesivo"
5. CRÍTICO - Tratamento especial para medicamentos:
   a) Se for marca comercial, substituir pelo composto ativo:
      - "Tylenol" -> "Paracetamol"
      - "Aspirina" -> "Ácido acetilsalicílico"
      - "Rivotril" -> "Clonazepam"
      - "Neosaldina" -> "Dipirona + Isometepteno + Cafeína"
      - "Dorflex" -> "Dipirona + Orfenadrina + Cafeína"
      - "Buscopan" -> "Butilescopolamina"
      - "Allegra" -> "Fexofenadina"
      - "Zoloft" -> "Sertralina"
   b) Se já for o composto ativo, manter como está:
      - "Dipirona" -> manter "Dipirona"
      - "Ibuprofeno" -> manter "Ibuprofeno"
      - "Amoxicilina" -> manter "Amoxicilina"
      - "Omeprazol" -> manter "Omeprazol"
      - "Losartana" -> manter "Losartana"
   c) Para combinações de compostos, listar todos separados por "+"
6. Categorias e suas subcategorias OBRIGATÓRIAS:
   - MEDICAMENTO -> "Medicamentos em Geral" ou uma das subcategorias:
     * "Medicamentos Controlados"
     * "Medicamentos Antibióticos"
     * "Medicamentos Anti-inflamatórios"
     * "Medicamentos Analgésicos"
     * "Medicamentos Cardiovasculares"
     * "Medicamentos Dermatológicos"
     * "Medicamentos Oftalmológicos"
     * "Medicamentos Pediátricos"
     * "Medicamentos em Geral" (para outros casos)
   - MATERIAL_MEDICO -> "Materiais Hospitalares"
   - COSMETICO -> "Produtos Cosméticos"
   - HIGIENE -> "Produtos de Higiene"
   - ALIMENTO -> "Gêneros Alimentícios"
   - MATERIAL_ESCRITORIO -> "Materiais de Escritório"
   - LIMPEZA -> "Produtos de Limpeza"
   - Outras categorias -> "Itens Diversos"

IMPORTANTE: A subcategoria é OBRIGATÓRIA e NUNCA pode ser null ou vazia.
Para MEDICAMENTOS, se não souber a subcategoria específica, use "Medicamentos em Geral".

Itens:
%(items)s"""


class OptimizedDocumentFilter:
    def __init__(
        self, batch_size=50, state_file="processing_state.json", num_threads=4
    ):
        self.batch_size = batch_size
        self.state_file = state_file
        self.num_threads = num_threads
        self.batch_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.thread_lock = threading.Lock()
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo-16k",
            temperature=0.4,
        )

        # Configura o parser de saída
        self.output_parser = PydanticOutputParser(pydantic_object=BatchAnaliseResponse)

        # Atualiza o prompt para incluir o formato do parser
        self.prompt = PromptTemplate(
            template=BATCH_ANALYSIS_TEMPLATE,
            input_variables=["items"],
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.normalized_items = {}
        self.current_code = 25000
        self.vectorizer = None  # Será inicializado sob demanda
        self.similarity_threshold = 0.92  # Aumentado para ser mais rigoroso
        self.processed_items_cache = {}
        self.name_mapping = {}
        self.similarity_cache = {}

        # Dicionário de categorias compatíveis
        self.compatible_categories = {
            "MEDICAMENTO": ["MEDICAMENTO"],
            "MATERIAL_MEDICO": ["MATERIAL_MEDICO"],
            "COSMETICO": ["COSMETICO", "HIGIENE"],
            "HIGIENE": ["HIGIENE", "COSMETICO"],
            "ALIMENTO": ["ALIMENTO"],
            "MATERIAL_ESCRITORIO": ["MATERIAL_ESCRITORIO"],
            "LIMPEZA": ["LIMPEZA"],
            "OUTROS_SERVICOS": ["OUTROS_SERVICOS"],
        }

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
                # Garante que todas as colunas de texto sejam strings
                text_columns = [
                    "item_normalizado",
                    "categoria",
                    "subcategoria",
                    "grupo_original",
                    "unidade",
                    "variantes",
                    "variantes_info",
                ]
                for col in text_columns:
                    if col in df_existing.columns:
                        df_existing[col] = df_existing[col].fillna("").astype(str)

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

    def _are_categories_compatible(self, category1: str, category2: str) -> bool:
        """Verifica se duas categorias são compatíveis para agrupamento."""
        if (
            category1 not in self.compatible_categories
            or category2 not in self.compatible_categories
        ):
            return False
        return category2 in self.compatible_categories[category1]

    def _normalize_text(self, text: str) -> str:
        """Normaliza o texto removendo caracteres especiais e convertendo para minúsculo."""
        import re

        if not text:
            return ""

        # Converte para string se for número
        if isinstance(text, (int, float)):
            text = str(text)

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
            "un",
            "und",
            "unidade",
            "cx",
            "caixa",
            "pct",
            "pacote",
            "kit",
            "conjunto",
            "sistema",
            "ref",
            "tipo",
            "marca",
            "modelo",
            "tamanho",
            "cor",
            "material",
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

        # Verifica comprimento mínimo das palavras
        words1 = set(norm1.split())
        words2 = set(norm2.split())

        # Se as palavras são muito diferentes em quantidade, retorna 0
        if len(words1) == 0 or len(words2) == 0:
            return 0.0

        if max(len(words1), len(words2)) / min(len(words1), len(words2)) > 2:
            return 0.0

        try:
            # Vetoriza os textos
            vectors = self.vectorizer.transform([norm1, norm2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

            # Penaliza diferenças grandes no comprimento do texto
            len_ratio = min(len(norm1), len(norm2)) / max(len(norm1), len(norm2))
            similarity = similarity * len_ratio

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
            existing_item = self.name_mapping[normalized_name]
            existing_details = self.normalized_items[existing_item]
            if self._are_categories_compatible(
                categoria, existing_details["categoria"]
            ):
                return existing_item

        # Se não encontrou pelo nome exato, procura por similaridade
        if len(self.normalized_items) > 0:
            try:
                # Encontra o item mais similar na mesma categoria
                max_similarity = 0
                most_similar_item = None

                for existing_item, details in self.normalized_items.items():
                    # Primeiro verifica se as categorias são compatíveis
                    if not self._are_categories_compatible(
                        categoria, details["categoria"]
                    ):
                        continue

                    # Calcula similaridade do nome normalizado
                    name_similarity = self._calculate_similarity(
                        item_normalizado, existing_item
                    )

                    # Se a similaridade do nome é alta, verifica também as variantes
                    if (
                        name_similarity > self.similarity_threshold * 0.9
                    ):  # 90% do threshold
                        # Verifica similaridade com cada variante
                        variantes_info = details["variantes_info"]
                        if isinstance(variantes_info, str):
                            try:
                                variantes_info = json.loads(
                                    variantes_info.replace("'", '"')
                                )
                            except:
                                variantes_info = []

                        # Calcula a similaridade máxima com qualquer variante
                        for variante in variantes_info:
                            variante_similarity = self._calculate_similarity(
                                item_normalizado, variante.get("item", "")
                            )
                            name_similarity = max(name_similarity, variante_similarity)

                    if (
                        name_similarity > max_similarity
                        and name_similarity > self.similarity_threshold
                    ):
                        max_similarity = name_similarity
                        most_similar_item = existing_item

                return most_similar_item

            except Exception as e:
                print(f"Erro ao procurar item similar: {str(e)}")
                return None

        return None

    def _is_duplicate(self, item: str, item_normalizado: str, categoria: str) -> bool:
        """
        Verifica se um item é duplicata de algum item existente.
        Retorna True se for duplicata, False caso contrário.
        """
        # Primeiro verifica se o item normalizado já existe exatamente igual
        if item_normalizado in self.normalized_items:
            existing_details = self.normalized_items[item_normalizado]
            if existing_details["categoria"] == categoria:
                return True

        # Procura por similaridade
        existing_item = self._find_existing_item(item_normalizado, categoria)
        if existing_item:
            return True

        return False

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

        # Formata os itens para o prompt
        items_list = []
        for idx, row in enumerate(batch):
            item = str(row["Produto/Serviço"])
            grupo = str(row["Grupo"])
            items_list.append(f"{idx + 1}. {item} (Grupo: {grupo})")

        items_text = "\n".join(items_list)

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Monta o prompt completo
                prompt_text = BATCH_ANALYSIS_TEMPLATE % {"items": items_text}

                # Chama o LLM diretamente
                messages = [
                    {
                        "role": "system",
                        "content": "Você é um assistente especializado em normalização de itens. Retorne apenas o JSON solicitado.",
                    },
                    {"role": "user", "content": prompt_text},
                ]

                response = await asyncio.to_thread(self.llm.invoke, messages)

                if not response or not response.content:
                    raise ValueError("Resposta vazia do LLM")

                # Pré-processa a resposta do LLM para garantir campos obrigatórios
                try:
                    import json

                    # Tenta limpar a resposta se ela estiver corrompida
                    content = response.content
                    if not content.strip().endswith("}"):
                        # Tenta encontrar o último item completo
                        last_complete = content.rfind('"}')
                        if last_complete > 0:
                            content = content[: last_complete + 2] + "]}"

                    try:
                        response_json = json.loads(content)
                    except json.JSONDecodeError as je:
                        print(f"⚠️ Erro ao decodificar JSON: {str(je)}")
                        # Cria um JSON válido com os dados que temos
                        response_json = {"items": []}
                        for row in batch:
                            item = {
                                "item_original": str(row["Produto/Serviço"]),
                                "item_normalizado": str(row["Produto/Serviço"]),
                                "categoria": "OUTROS_SERVICOS",
                                "subcategoria": "Itens Diversos",
                            }
                            response_json["items"].append(item)

                    # Garante que todos os itens tenham os campos obrigatórios
                    for item in response_json.get("items", []):
                        # Garante item_original
                        if "item_original" not in item:
                            item["item_original"] = item.get(
                                "item_normalizado", "Item não identificado"
                            )

                        # Garante item_normalizado
                        if "item_normalizado" not in item:
                            item["item_normalizado"] = item.get(
                                "item_original", "Item não identificado"
                            )

                        # Garante categoria
                        if "categoria" not in item:
                            item["categoria"] = "OUTROS_SERVICOS"

                        # Garante subcategoria
                        if "subcategoria" not in item:
                            item["subcategoria"] = "Itens Diversos"
                        elif item["subcategoria"] is None:
                            item["subcategoria"] = "Itens Diversos"

                    # Reconstrói a resposta JSON
                    response.content = json.dumps(response_json)

                except Exception as e:
                    print(f"⚠️ Erro ao processar resposta do LLM: {str(e)}")
                    # Cria um JSON válido com os dados que temos
                    response_json = {"items": []}
                    for row in batch:
                        item = {
                            "item_original": str(row["Produto/Serviço"]),
                            "item_normalizado": str(row["Produto/Serviço"]),
                            "categoria": "OUTROS_SERVICOS",
                            "subcategoria": "Itens Diversos",
                        }
                        response_json["items"].append(item)
                    response.content = json.dumps(response_json)

                # Usa o parser para validar e processar a resposta
                try:
                    parsed_response = self.output_parser.parse(response.content)
                    results = [item.model_dump() for item in parsed_response.items]
                except Exception as parse_err:
                    print(f"\n❌ Erro ao fazer parse da resposta:")
                    print(f"Erro: {str(parse_err)}")
                    print("Resposta recebida:")
                    print("-" * 50)
                    print(response.content)
                    print("-" * 50)
                    raise ValueError(
                        f"Falha ao fazer parse da resposta: {str(parse_err)}"
                    )

                processed_results = []

                # Processa cada item do lote
                for idx, result in enumerate(results):
                    try:
                        row = batch[idx]
                        item = str(row["Produto/Serviço"])
                        codigo_original = str(row["Código"])

                        # Verifica se o item_original corresponde ao item do lote
                        if result["item_original"] != item:
                            print(
                                f"\n⚠️ Aviso: Item {idx} tem item_original diferente do item fornecido:"
                            )
                            print(f"Item fornecido: {item}")
                            print(f"Item retornado: {result['item_original']}")
                            print("Corrigindo para usar o item fornecido...")
                            result["item_original"] = item

                        print(
                            f"\n→ Processando item: {item} (Código: {codigo_original})"
                        )

                        # Verifica cache
                        if item in self.processed_items_cache:
                            print(f"✓ Item encontrado no cache: {item}")
                            processed_results.append(self.processed_items_cache[item])
                            continue

                        # Verifica se é duplicata
                        if self._is_duplicate(
                            item, result["item_normalizado"], result["categoria"]
                        ):
                            print(f"⚠️ Item detectado como duplicata: {item}")
                            existing_item = self._find_existing_item(
                                result["item_normalizado"], result["categoria"]
                            )
                            if existing_item:
                                print(f"✓ Usando item existente: {existing_item}")
                                details = self.normalized_items[existing_item]

                                # Adiciona como variante do item existente
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
                                        self._calculate_similarity(
                                            variante.get("item", ""), item
                                        )
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
                                    "variantes": [
                                        v.get("item") for v in variantes_info
                                    ],
                                    "variantes_info": json.dumps(
                                        variantes_info, ensure_ascii=False
                                    ),
                                }
                                self.processed_items_cache[item] = processed_result
                                processed_results.append(processed_result)
                                continue

                        # Se não é duplicata, cria novo item
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
                        normalized_name = self._normalize_text(
                            result["item_normalizado"]
                        )
                        self.name_mapping[normalized_name] = result["item_normalizado"]

                        self.processed_items_cache[item] = novo_item
                        processed_results.append(novo_item)

                    except Exception as e:
                        print(f"\n❌ Erro ao processar item {idx} do lote:")
                        print(f"Tipo de erro: {type(e).__name__}")
                        print(f"Mensagem de erro: {str(e)}")
                        print(f"Dados do item que causou erro:")
                        print(f"Row data: {row if 'row' in locals() else 'N/A'}")
                        print(
                            f"Result data: {result if 'result' in locals() else 'N/A'}"
                        )
                        raise e

                return processed_results

            except Exception as e:
                print(f"\n❌ ERRO no processamento do lote:")
                print(f"Tipo de erro: {type(e).__name__}")
                print(f"Mensagem de erro: {str(e)}")
                if retry_count < max_retries - 1:
                    print(
                        f"Tentando novamente... (tentativa {retry_count + 1}/{max_retries})"
                    )
                    retry_count += 1
                    continue
                raise e

    def _thread_worker(self):
        """Função worker que será executada em cada thread."""
        while True:
            try:
                # Pega um lote da fila
                batch = self.batch_queue.get_nowait()
                if batch is None:  # Sinal para encerrar a thread
                    break

                # Processa o lote
                try:
                    batch_results = asyncio.run(self._process_batch(batch))
                    self.result_queue.put((True, batch_results))
                except Exception as e:
                    self.result_queue.put((False, str(e)))

                self.batch_queue.task_done()
            except queue.Empty:
                break

    async def processa_tabela_async(
        self,
        caminho: str,
        output_file: str = "itens_processados_incremental.xlsx",
        items_to_process: int = 100,
    ) -> pd.DataFrame:
        """Processa uma tabela Excel de forma assíncrona e incremental usando múltiplas threads."""
        print("\n" + "=" * 50)
        print(f"INICIANDO PROCESSAMENTO DA TABELA COM {self.num_threads} THREADS")
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

        # Divide em lotes
        batches = [
            df[i : i + self.batch_size] for i in range(0, len(df), self.batch_size)
        ]

        # Coloca os lotes na fila
        for batch_df in batches:
            self.batch_queue.put(batch_df.to_dict("records"))

        # Adiciona sinais de término para cada thread
        for _ in range(self.num_threads):
            self.batch_queue.put(None)

        # Inicia as threads
        threads = []
        for _ in range(self.num_threads):
            thread = threading.Thread(target=self._thread_worker)
            thread.start()
            threads.append(thread)

        # Processa resultados conforme eles chegam
        with tqdm(total=len(batches), desc="Processando lotes") as pbar:
            processed_batches = 0
            while processed_batches < len(batches):
                success, result = self.result_queue.get()

                if not success:
                    print(f"\n❌ Erro no processamento do lote: {result}")
                    # Salva o progresso mesmo em caso de erro
                    if len(df_resultado) > 0:
                        df_resultado.to_excel(output_file, index=False)
                    self._save_state()
                    raise Exception(f"Erro no processamento: {result}")

                # Para cada resultado do lote
                for item_result in result:
                    item_normalizado = item_result["item_normalizado"]

                    with self.thread_lock:
                        # Verifica se já existe uma linha com este item_normalizado
                        existing_mask = (
                            df_resultado["item_normalizado"] == item_normalizado
                        )
                        if existing_mask.any():
                            # Atualiza a linha existente
                            existing_idx = existing_mask.idxmax()
                            existing_row = df_resultado.loc[existing_idx]

                            # Combina as variantes existentes com as novas
                            try:
                                existing_variantes = json.loads(
                                    existing_row["variantes_info"].replace("'", '"')
                                )
                            except:
                                existing_variantes = []

                            new_variantes = json.loads(item_result["variantes_info"])

                            # Combina as variantes, evitando duplicatas
                            combined_variantes = []
                            existing_items = set()

                            # Adiciona variantes existentes
                            for var in existing_variantes:
                                item = var.get("item")
                                if item and item not in existing_items:
                                    combined_variantes.append(var)
                                    existing_items.add(item)

                            # Adiciona novas variantes
                            for var in new_variantes:
                                item = var.get("item")
                                if item and item not in existing_items:
                                    combined_variantes.append(var)
                                    existing_items.add(item)

                            # Atualiza a linha existente
                            df_resultado.at[existing_idx, "variantes"] = [
                                v.get("item") for v in combined_variantes
                            ]
                            df_resultado.at[existing_idx, "variantes_info"] = (
                                json.dumps(combined_variantes, ensure_ascii=False)
                            )
                        else:
                            # Se não existe, adiciona nova linha
                            df_resultado = pd.concat(
                                [df_resultado, pd.DataFrame([item_result])],
                                ignore_index=True,
                            )

                processed_batches += 1
                self.last_processed_index = (
                    start_index + processed_batches * self.batch_size - 1
                )

                # Salva incrementalmente
                df_resultado.to_excel(output_file, index=False)
                self._save_state()

                pbar.update(1)

        # Aguarda todas as threads terminarem
        for thread in threads:
            thread.join()

        print("\nRELATÓRIO FINAL:")
        print(f"Total de itens processados nesta execução: {total_itens}")
        print(f"Total de itens únicos após normalização: {len(df_resultado)}")
        print(f"Próximo índice a ser processado: {self.last_processed_index + 1}")

        return df_resultado


def main():
    """Função principal para demonstração do uso."""
    filtro = OptimizedDocumentFilter(batch_size=50, num_threads=4)

    # Processa a tabela de forma assíncrona e incremental
    df_limpo = asyncio.run(
        filtro.processa_tabela_async(
            caminho="tabelaInicial.xlsx",
            output_file="itens_processados_incremental.xlsx",
            items_to_process=22014,
        )
    )

    print("\nResultados salvos em 'itens_processados_incremental.xlsx'")


if __name__ == "__main__":
    main()
