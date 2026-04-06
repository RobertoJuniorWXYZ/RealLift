# `reallift.pipelines.geo_pipeline.run_geo_experiment`

Na biblioteca RealLift, a função `run_geo_experiment` é a principal *pipeline* analítica de ponta-a-ponta. Ela centraliza todas as etapas necessárias para consolidar o resultado de um teste A/B Geográfico **após (ou durante) a intervenção**.

## Assinatura

```python
def run_geo_experiment(
    filepath: str,
    date_col: str,
    treatment_start_date: str,
    treatment_end_date: str = None,
    doe: dict = None,
    scenario: int = None,
    start_date: str = None,
    end_date: str = None,
    geos: list = None,
    n_treatment: int = 1,
    fixed_treatment: list = None,
    mde: float = 0.015,
    experiment_days: int | list = [21, 60],
    n_folds: int = 5,
    random_state: int = None,
    plot: bool = True,
    verbose: bool = True
) -> dict
```

## Integração com Design of Experiments (DoE)

Uma das maiores vantagens da `run_geo_experiment` é a capacidade de ler diretamente o objeto retornado pela função `design_of_experiments`. 

- **`doe`**: Se você passar o dicionário retornado pelo DoE, a pipeline irá ignorar os parâmetros `geos`, `n_treatment` e `fixed_treatment`, utilizando exatamente os agrupamentos validados no planejamento.
- **`scenario`**: Índice do cenário escolhido (ex: 1 para 10% de tratamento, 2 para 20%, etc.).

## Janelas de Análise

- **`treatment_start_date`**: Data exata em que a campanha começou. Divide o mundo em "Treino" e "Teste".
- **`treatment_end_date`**: (Novo) Se você quiser analisar apenas um pedaço do período pós-intervenção (ex: os primeiros 14 dias de uma campanha que ainda está rodando), utilize este parâmetro para fechar a janela de Lift.

## Etapas da Pipeline

1. **Descoberta/Recuperação de Clusters**: Recupera do DoE ou descobre via ElasticNet os melhores controles sintéticos.
2. **Validação Cruzada (`validate_geo_clusters`)**: Atesta a robustez histórica das séries.
3. **Poder Estatístico (`estimate_duration`)**: Valida se o teste tem fôlego para detectar o MDE.
4. **Controle Sintético (`run_synthetic_control`)**: Calcula o Lift Absoluto e Percentual com intervalos de confiança via **Bootstrap**.
5. **Placebo Robusto (`run_placebo_tests`)**: Aplica a metodologia de **Razão MSPE** para garantir que o efeito é único da região tratada.
6. **Diagnósticos Visuais**: Renderiza gráficos de séries temporais, lift acumulado e distribuição de placebo.

## Retorno (*Output*)

```python
{
    "clusters": [...],       # Agrupamentos utilizados
    "results": [
        {
            "cluster": {...},
            "validation": {...},
            "duration": {...},
            "synthetic": {...},  # Resultados de Lift e MSPE
            "placebo": {...}    # p-valor empírico robusto
        }
    ],
    "consolidated": {...}    # Visão agregada de todos os clusters
}
```
