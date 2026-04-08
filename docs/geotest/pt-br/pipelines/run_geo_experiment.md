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

- **`doe`**: Se você passar o dicionário retornado pelo DoE, a pipeline irá herdar tanto a modalidade analítica (Controle Sintético vs. Matched DiD) quanto ignorar os parâmetros `geos`, `n_treatment` e `fixed_treatment`, utilizando exatamente os agrupamentos validados no planejamento.
- **`scenario`**: Índice do cenário escolhido dentro do dicionário `doe` (ex: 1 para o cenário com 10% de tratamento, 2 para 20%, etc.).

## Janelas de Análise

- **`treatment_start_date`**: Data exata em que a campanha começou. Divide o mundo em "Treino" e "Teste".
- **`treatment_end_date`**: (Novo) Se você quiser analisar apenas um pedaço do período pós-intervenção (ex: os primeiros 14 dias de uma campanha que ainda está rodando), utilize este parâmetro para fechar a janela de Lift.

## Etapas da Pipeline

1. **Descoberta/Recuperação de Clusters**: Recupera do DoE ou descobre via ElasticNet os melhores controles sintéticos.
2. **Validação Cruzada (`validate_geo_clusters`)**: Atesta a robustez histórica das séries.
3. **Poder Estatístico (`estimate_duration`)**: Valida se o teste final (agora com prazo determinado) tem real significância sobre os dados obtidos.
4. **Cálculo Causal**: Dependendo da metodologia herdada do DoE, ativa o `run_synthetic_control` (Otimização Convexa) ou o `run_matched_did` (Diferenças em Diferenças). Calcula o Lift Absoluto e Percentual com intervalos de confiança via **Pooled Bootstrap**.
5. **Placebo Robusto (`run_placebo_tests`)**: Aplica testes exatos de permutação sobre os controles para calcular a **Razão MSPE** e atestar que a quebra de tendência é singular e não um efeito cíclico do mercado. Mantém total coerência metodológica espelhando o método de cálculo do Passo 4.
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
