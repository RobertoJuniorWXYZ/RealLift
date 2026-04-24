# `reallift.pipelines.geo_pipeline.design_of_experiments`

A função `design_of_experiments` (DoE) atua como a interface mestre para planejamento de testes. Seu uso é mandatório **antes** de iniciar qualquer campanha, pois projeta matematicamente as premissas de custo, tempo e sensibilidade (MDE).

## Assinatura

```python
def design_of_experiments(
    filepath: str,
    date_col: str,
    start_date: str = None,
    end_date: str = None,
    geos: list = None,
    pct_treatment: float | list = None,
    fixed_treatment: list = None,
    mde: float = None,
    experiment_days: int | list = [21, 28, 30, 35],
    n_folds: int = 5,
    search_mode: str = "ranking",
    experiment_type: str = "synthetic_control",
    use_elasticnet: bool = False,
    n_jobs: int = None,
    verbose: bool = True
) -> dict
```

## Parâmetros Principais

| Parâmetro | Tipo | Default | Descrição |
|:---|:---|:---|:---|
| `filepath` | `str` | **Obrigatório** | Caminho para o arquivo CSV com os dados históricos. |
| `date_col` | `str` | **Obrigatório** | Nome da coluna de data. |
| `pct_treatment` | `float` \| `list` | `[0.1, 0.2, 0.3]` | Porcentagem(ns) de geos para tratamento (ex: `0.2` para 20%). |
| `experiment_days` | `list` | `[21, 28, 30, 35]` | Janelas de tempo para cálculo do MDE. |
| `use_elasticnet` | `bool` | `False` | Se `True`, utiliza ElasticNet para pré-filtragem de doadores (recomendado para alta dimensionalidade). |
| `n_jobs` | `int` | `None` | Número de processos paralelos para o screening inicial. |
| `experiment_type` | `str` | `"synthetic_control"` | Tipo de modelo: `"synthetic_control"` ou `"matched_did"`. |


## Arquitetura da Pipeline (Design Level)

Para embasar a viabilidade do teste, a pipeline processa metadados pré-intervenção orquestrando três pilares:

1. **Agrupamento Ótimo (`discover_geo_clusters`)**: Identifica as melhores combinações de geos para tratamento e controle. No modo `synthetic_control`, utiliza otimização convexa ElasticNet. No modo `matched_did`, agrupa baseando-se unicamente na métrica de correlação média sob pesos idênticos (1/N).
2. **Avaliação Pragmática (`validate_geo_clusters`)**: Executa Cross-Validation (Backtesting) em janelas rolantes para garantir a estabilidade holística.
3. **Cálculo de Requisitos (`estimate_duration`)**: Projeta o MDE (Efeito Mínimo Detectável) para diferentes durações (ex: 21, 30, 60 dias), permitindo escolher o cenário com melhor custo-benefício.

## Relatório Técnico (*Verbosity*)

O terminal exibe um relatório detalhado para cada cenário (ex: 10%, 20% de tratamento) incluindo:

- **EXPERIMENTAL SCOPE**: Mostra a cobertura total do mercado (Geos Distintos, Tratamentos e Controles distintos / Total de Geos).
- **TEST POOL**: Lista as unidades geográficas selecionadas para receber o tratamento (nomes exibidos integralmente, sem truncamento).
- **CONTROL DESIGN (DONOR POOL & WEIGHTS)**: Exibe cada cluster com seus respectivos geos doadores e pesos de importância. Útil para detectar se o modelo está equilibrado.
- **CROSS-VALIDATION SUMMARY**: Tabela com métricas de backtesting por cluster (R² Treino/Teste, MAPE e WAPE), com largura de colunas dinâmica ajustada ao nome mais longo.
- **MDE COMPARISON**: Tabela final comparando todos os cenários. Inclui as colunas: `Distinct` (total de geos distintos), **`Controls`** (controles distintos por cenário), `MDE`, `R²`, `MAPE` e `WAPE` (em formato percentual).

## Retorno (*Output*)

O dicionário retornado aglutina as inferências de todos os cenários testados:

```python
{
    "experiment_type": "synthetic_control",
    "scenarios": [
        {
            "pct_treatment": 0.10,
            "n_treatment": 3,
            "treatment_pool": ["geo_A", "geo_B", "geo_C"], # Lista agregada de todos os tratamentos
            "clusters": [...],      # Agrupamentos encontrados
            "duration": {...},      # Curva de MDE e Power
            "validation": pd.DataFrame # Resultados de Backtesting (R2, MAPE)
        },
        ...
    ],
    "comparison": pd.DataFrame      # Tabela comparativa consolidada
}
```
