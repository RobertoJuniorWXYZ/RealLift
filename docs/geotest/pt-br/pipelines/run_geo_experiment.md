# `reallift.pipelines.geo_pipeline.run_geo_experiment`

Na biblioteca RealLift, a função `run_geo_experiment` é a principal *pipeline* analítica de ponta-a-ponta para o cálculo de incrementalidade. Ela centraliza todas as etapas necessárias para consolidar o resultado de um teste A/B Geográfico **após (ou durante) a intervenção**.

## Assinatura

```python
def run_geo_experiment(
    filepath: str,
    date_col: str,
    treatment_start_date: str,
    geos: list = None,
    n_treatment: int = 1,
    fixed_treatment: list = None,
    mde: float = 0.01,
    max_days: list = [21, 60],
    n_folds: int = 1,
    random_state: int = None,
    plot: bool = True,
    verbose: bool = True
) -> dict
```

## Arquitetura da Pipeline

Diferente de métodos isolados, a `run_geo_experiment` orquestra o encadeamento de 6 submódulos matemáticos sequenciais para cada cluster (grupo cidade controle vs cidade tratamento) encontrado:

1. **Descoberta de Clusters (`find_best_geo_clusters`)**: Acha os *matches* perfeitos de controle usando dados estritamente anteriores a `treatment_start_date`.
2. **Validação Cruzada Temporal (`validate_geo_groups`)**: Roda *Time Series Cross-Validation* para atestar a robustez fora da amostra (OOF R2 e MAPE).
3. **Poder e Duração (`estimate_duration`)**: Calcula o retroceder estatístico para atestar se a curva tem poder para encontrar o Efeito Mínimo Detectável (MDE) desejado.
4. **Controle Sintético Estrutural (`run_synthetic_control`)**: Projeta o preditor contrafactual do período de teste cruzando as estimativas otimizadas com os dados pós-tratamento para auferir o Impacto Real Absoluto e Relativo.
5. **Testes de Placebo em In-Space (`run_placebo_tests`)**: Realiza re-amostragens recursivas atribuindo intervenções faltas às cidades de controle para mapear p-valores empíricos e a significância formal geométrica.
6. **Diagnósticos Visuais (`plot_*`)**: Compila renderizações matemáticas baseadas no `matplotlib` para reportar publicamente as predições de efeito na série.

## Parâmetros Principais

- **`treatment_start_date`** *(str)*: Recorte crucial na base `YYYY-MM-DD`. Divide os dados usados em *Treinamento de Otimização* (pré-start) e *Avaliação de Lift* (pós-start).
- **`mde`** *(float, default=0.01)*: Minimum Detectable Effect (Lift esperado). Ex: `0.01` equivale a estimar 1% de deslocamento de *baseline* para validar *Power*.
- **`max_days`** *(int/list)*: O teto de tempo que o teste pode durar comercialmente para a análise de *Power*.
- **`n_folds`** *(int)*: Quantas janelas deslizantes avaliar durante a checagem cruzada da acurácia retroativa do Sintético.

## Retorno (*Output*)

A função compila um rico log formatado no STDOUT sumarizando as tabelas do Experimento e devolve a hierarquia integral dos metadados:

```python
{
    "summary": {"clusters": [...]},  # Metadados globais da Otimização Inicial
    "results": [
        {
            "cluster": {...},       # Composição (Tratamento vs Controles selecionados)
            "validation": {...},    # Métricas CV (R2 Out-of-Fold, MAPE)
            "duration": {...},      # Estatísticas de MDE, Poder alcançado
            "synthetic": {...},     # O delta central: lift_total_absoluto, limites do bootstrap
            "placebo": {...}        # Resultados de p-value das simulações falsas
        }
    ]
}
```
