# `reallift.utils.data_cleaning.clean_geo_data`

A função `clean_geo_data` atua como o **Portal de DataOps** do framework RealLift. Sua finalidade é preparar, padronizar e validar dados geo-espaciais brutos, garantindo que as séries temporais sejam estatisticamente robustas para os algoritmos de Inferência Causal (Controle Sintético e Log-Diff).

## Assinatura

```python
def clean_geo_data(
    data, 
    date_col: str, 
    imputation_method: str = 'interpolation', 
    constant_value: float = 1e-3, 
    verbose: bool = True,
    plot: bool = False,
    save_csv: bool = True,
    save_pdf: bool = False,
    file_name: str = 'cleaned_geo_data.csv',
    pdf_name: str = 'cleaning_report.pdf',
    max_zero_rate: float = None,
    top_n_geos: int = None,
    keep_top_quantiles: int = None,
    exclude_geos: list = None,
    quantile_bins: int = None,
    start_date: str = None,
    end_date: str = None,
    logo: str = None
) -> pd.DataFrame
```

---

## 1. Funcionamento do Pipeline

O pipeline de limpeza executa seis estágios sequenciais para transformar dados de faturamento/vendas ruidosos em matrizes matemáticas prontas para modelagem:

1.  **Padronização Temporal:** Converte diversos formatos de data para o padrão ISO (YYYY-MM-DD).
2.  **Ordenação Cronológica:** Reorganiza a linha do tempo e detecta lacunas de dias ausentes.
3.  **Segregação Geo-espacial:** Separa a dimensão temporal das métricas de cada geografia.
4.  **Imputação Algébrica (Anti-Log Crash):** Trata células com valor zero ou vazias (NaN) para evitar o erro matemático de $-\infty$ durante as transformações logarítmicas internas do algoritmo.
5.  **Scorecard de Qualidade:** Gera um diagnóstico detalhado sobre a esparsidade e a distribuição de volume de cada cidade.
6.  **Filtragem Estratégica:** Remove dinamicamente cidades de baixa qualidade ou baixo volume com base em regras de negócio (Pareto, Taxa de Zeros ou Ranking).

---

## 2. Parâmetros

### 2.1 Configuração de Dados e Tempo

| Parâmetro | Tipo | Descrição |
|:---|:---|:---|
| `data` | `DataFrame` / `str` | Fonte de dados: um DataFrame do Pandas ou o caminho para um arquivo CSV. |
| `date_col` | `str` | Nome da coluna que representa a dimensão temporal (datas). |
| `exclude_geos` | `list` | Lista de nomes de geografias para exclusão forçada (outliers conhecidos ou praças de teste). |
| `start_date` | `str` | *(Opcional)* Filtra o dataset para incluir apenas registros **a partir desta data** (inclusive). Formato: `'YYYY-MM-DD'`. |
| `end_date` | `str` | *(Opcional)* Filtra o dataset para incluir apenas registros **até esta data** (inclusive). Formato: `'YYYY-MM-DD'`. |

> [!TIP]
> **Filtro de Período:** Use `start_date` e `end_date` para restringir a análise a uma janela temporal específica dentro de um dataset maior. Por exemplo, se o CSV cobre 2 anos mas você quer preparar apenas o pré-teste de 90 dias para o DoE, defina `end_date` para a data de início da campanha.

### 2.2 Estratégia de Imputação

| Parâmetro | Tipo | Default | Descrição |
|:---|:---|:---|:---|
| `imputation_method` | `str` | `'interpolation'` | `'interpolation'`: Preenche lacunas via interpolação linear (tende a preservar a tendência). <br> `'constant'`: Preenche todos os vazios com o `constant_value`. |
| `constant_value` | `float` | `1e-3` | O "piso" de valor injetado. Essencial para evitar o erro de `log(0)`. |

### 2.3 Mecanismos de Seleção e Filtragem (DataOps)

| Parâmetro | Tipo | Descrição |
|:---|:---|:---|
| `max_zero_rate` | `float` | Remova geos com uma taxa de zeros/vazios superior ao limite (ex: `0.2` remove cidades com >20% de dias zerados). |
| `quantile_bins` | `int` | Número de fatias para a análise de distribuição de volume (ex: `4` para Quartis, `10` para Decis). |
| `keep_top_quantiles`| `int` | Mantém apenas os top N quantis (ex: `1` mantém apenas o Q1 — as cidades de maior volume). |
| `top_n_geos` | `int` | Seleciona as N melhores geografias baseando-se em um ranking combinado de Volume e Qualidade. |

### 2.4 Saídas e Relatório

| Parâmetro | Tipo | Default | Descrição |
|:---|:---|:---|:---|
| `save_csv` | `bool` | `True` | Exporta o DataFrame resultante para um CSV. |
| `file_name` | `str` | `'cleaned_geo_data.csv'` | Nome do arquivo CSV de saída. |
| `save_pdf` | `bool` | `False` | Gera um relatório PDF de auditoria (`cleaning_report.pdf`). |
| `pdf_name` | `str` | `'cleaning_report.pdf'` | Nome do arquivo PDF de saída. |
| `logo` | `str` | `None` | Caminho para um logotipo a ser incluído no cabeçalho do PDF. |

---

## 3. Entendendo o Scorecard de Diagnóstico

Quando o parâmetro `verbose=True` é utilizado, a função imprime no terminal o **Geos Scorecard**. Esta tabela é vital para auditar o impacto da limpeza nos seus dados:

| Coluna | Significado | Interpretação |
|:---|:---|:---|
| **Geo** | Nome da Praça | Identificação da unidade. |
| **Imputed** | N de células tratadas | Quantos dias foram "inventados" ou corrigidos via imputação. |
| **% Zeros** | Taxa de Esparsidade | Proporção de dias em que a cidade não teve vendas/faturamento. |
| **Sum Original** | Volume Bruto | Soma total dos valores antes da limpeza. |
| **% Imputed** | Impacto Volumétrico | Quanto do volume final da cidade é fruto de imputação. |

> [!IMPORTANT]
> **A Regra de Ouro da Imputação**
> Se uma cidade possui `% Imputed` acima de 5%, os pesos do Controle Sintético podem ser distorcidos por dados "sintetizados" demais. Recomenda-se filtrar essas praças usando o `max_zero_rate`.

---

## 4. Análise de Quantis (Princípio de Pareto)

O RealLift utiliza segmentação por volume para ajudar na escolha do pool de doadores. No terminal, você verá a análise de quantis:

- **Q1 (Top Volume):** Geralmente as capitais e grandes centros. São ótimos tratamentos, mas doadores perigosos se forem muito dominantes.
- **Q4/Q10 (Cauda Longa):** Cidades pequenas e ruidosas. Devem ser filtradas no DoE para evitar que o algoritmo tente polir o erro usando cidades irrelevantes.

---

## 5. Exemplo de Uso

O workflow padrão de DataOps antes de iniciar um experimento é:

```python
from reallift.utils.data_cleaning import clean_geo_data

# 1. Limpeza e Filtração: Mantendo apenas as praças ricas (Q1 e Q2) e estáveis (<10% zeros)
#    restrita ao período pré-teste
df_clean = clean_geo_data(
    data="vendas_raw.csv",
    date_col="data",
    start_date="2024-01-01",     # Filtra apenas a partir desta data
    end_date="2025-12-31",       # Filtra até esta data (inclusive)
    max_zero_rate=0.10,          # Remove geos com mais de 10% de dias zerados
    quantile_bins=5,             # Divide o volume em Quintis
    keep_top_quantiles=2,        # Mantém apenas Q1 e Q2 (Top 40% em volume)
    imputation_method='interpolation'
)

# 2. O resultado 'df_clean' agora está pronto para o DoE
# clusters = design_of_experiments(data=df_clean, ...)
```

---

## 6. Visualização Forense (`plot=True`)

Ao ativar `plot=True`, a função gera um gráfico comparativo de **"Antes vs Depois"** para as 20 maiores geografias. Isso permite que sua equipe de dados valide visualmente se a estratégia de interpolação está preservando as tendências sazonais ou se está criando padrões artificiais indesejados.
