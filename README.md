# 🦜 Nheengatu-Uirapuru: Alinhamento Semântico e IA Explicável

![Uirapuru Cover](outputs/22-UIRAPURU.jpg)
> **Identidade Visual:** Obra "Uirapuru" gentilmente creditada à **Associação Cultural Pintura Solidária** (Saiba mais em: [pinturasolidaria.org.br](https://pinturasolidaria.org.br/)).

**Dashboard Analítico de Alinhamento Hiperdimensional | Low-Resource Languages**

Este repositório contém a infraestrutura completa de dados, modelagem matemática e inferência visual desenvolvida para a pesquisa de alinhamento vetorial semântico bidirecional entre a Língua Portuguesa e o Nheengatu (língua indígena Tupi-Guarani de baixos recursos). 

A ferramenta utiliza Inteligência Artificial (modelos baseados em arquitetura Transformer) associada ao cálculo de Procrustes Ortogonal para provar isomorfismo topológico, exibindo os resultados através de uma interface baseada em **IA Explicável (XAI)** utilizando conceitos modernos de UI/UX (*Glass Morphism* e *Mesh Gradients*).

---

## 🎯 Objetivos da Pesquisa
1. **Mineração de Contexto (*Offset Mapping*):** Extrair tensores latentes contextualizados de 768 dimensões de ambos os idiomas utilizando as Redes Neurais `neuralmind/bert-base-portuguese-cased` e `dominguesm/canarim-bert-nheengatu`.
2. **Isomorfismo Geométrico:** Provar que duas línguas isoladas desenvolvem hiperplanos matemáticos de formato idêntico.
3. **Tradução Zero-Shot (Unsupervised):** Ensinar ao computador o alinhamento bilíngue através da rotação pura dos vetores (Procrustes), sem o uso de dicionários diretos de tradução durante a predição.
4. **Hubness Reduction:** Mitigação de buracos-negros topológicos através da métrica topológica iterativa **CSLS** (*Cross-Domain Similarity Local Scaling*).

---

## 🛠️ Tecnologias e Dependências
O projeto é modular e as dependências operacionais estão estritamente controladas:
* **Hugging Face (`transformers`, `datasets`):** Tokenização e alocação dinâmica dos pesos neurais.
* **PyTorch (`torch`):** Operações de tensores locais, cálculos em memória RAM e SVD (*Singular Value Decomposition*).
* **Scikit-Learn (`scikit-learn`):** Compressão visual da matriz densa via PCA (2D) e t-SNE não-linear.
* **Streamlit (`streamlit`):** *Framework* central para o painel SaaS de renderização XAI.
* **Plotly & Seaborn:** Exposição dos dados dimensionais em radares de dispersão (*Scatter Plots*).
* **Base64 / CSS Híbrido:** Renderização limpa da UI/UX eliminando recálculos DOM intrusivos.

---

## 📂 Arquitetura do Repositório

A base de código segue o padrão corporativo (*Code Freeze* / Pipeline Sequencial):

```text
nheengatu-uirapuru/
├── data/
│   ├── raw/                           # Dados brutos em Excel/CSV
│   └── processed/                     # Tensores .pt e Vocabulários JSON gerados
├── outputs/                           # Exportações de Gráficos e Imagem Base64
├── src/
│   ├── ingestion/                     # ETL, limpeza NFC e Cartesian Data Augmentation
│   ├── models/                        # Extratores Transformer, Offset Mapping e Matrizes In-Domain
│   └── alignment/                     # Filtragem CSLS, Problema de Procrustes e Plotagem t-SNE/PCA
├── app.py                             # O Frontend (SaaS Dashboard & XAI)
├── inference_engine.py                # O Backend (Classe de orquestração RAM para o Streamlit)
└── requirements.txt                   # Mapeamento do Ambiente

```

---

## ⚙️ Como Reproduzir a Pesquisa (Local)

**1. Clone o repositório e instale as dependências:**

```bash
git clone [https://github.com/SEU_USUARIO/nheengatu-uirapuru.git](https://github.com/SEU_USUARIO/nheengatu-uirapuru.git)
cd nheengatu-uirapuru
pip install -r requirements.txt

```

**2. A Fase de Extração (Transformando Texto em Tensores):**
Certifique-se de que os dados brutos estejam na pasta `data/raw/` e execute sequencialmente as rotas abaixo para extrair a matriz oculta:

```bash
python src/ingestion/pipeline_augment.py
python src/models/extraction_script.py
python src/models/build_matrices.py
python src/models/mine_context_pool.py

```

**3. A Fase de Geometria e Rotação Ortogonal:**
Este passo é crucial. Ele resolve o Problema de Schönemann-Procrustes e extrai a matriz $W$, responsável por rotacionar todo o universo semântico do Nheengatu para dentro da mente do Português:

```bash
python src/alignment/orthogonal_procrustes.py

```

*(Opcional: Para gerar os mapas estáticos na pasta `outputs/`, execute os scripts `visualize_pca.py` e `visualize_tsne.py` em `src/alignment/`).*

**4. Levantar o SaaS Dashboard (Terminal):**
Acesse as explicações lógicas e a navegação hiperdimensional na interface web:

```bash
streamlit run app.py

```

---

## 🌌 IA Explicável (XAI)

O aplicativo não retorna apenas a "Tradução" calculada, mas a justifica visualmente de duas formas:

* **Degradação Zero-Shot:** Aciona *Warnings* caso a predição tenha sido gerada puramente por gravidade semântica (cosseno inferior a 0.45).
* **Justificativa Metodológica (Hubness):** Compara a similaridade Top-1 contra a Top-5 para alertar o usuário e a banca de pesquisa caso a inferência caia em um "buraco negro semântico" (ex: verbos passivos e sufixos atraindo vetores desconhecidos).

---

> **Desenvolvido e Auditado para Inovação Corporativa e Bancas Acadêmicas.**
> *Low-Resource Language Processing (LRLP) | Teresina, Piauí, Brasil.*

```

***