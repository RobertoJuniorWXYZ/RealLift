# `reallift.geo.duration.estimate_duration`

A função `estimate_duration` fundamenta os pilares de *Power Analysis* e Dimensionamento Amostral. Ela é responsável por quantificar a inércia estatística do grupo de controle e determinar a viabilidade temporal do experimento para detectar um determinado efeito (MDE).

## Assinatura

```python
def estimate_duration(
    filepath: str,
    date_col: str,
    treatment_geo: str | list = None,
    control_geos: list = None,
    clusters: list = None,
    mde: float = 0.01,
    alpha: float = 0.05,
    power_target: float = 0.8,
    experiment_days: int | list = [21, 60],
    start_date: str = None,
    end_date: str = None,
    cluster_idx: int = None,
    consolidated: bool = False,
    cluster_residuals: list = None,
    verbose: bool = True
) -> dict
```

## Modos de Operação

A função opera em três modos distintos dependendo dos parâmetros fornecidos:

### 1. Estimativa de Duração (Padrão)
Quando um `mde` é fornecido (ex: 0.02), a função calcula o poder estatístico para cada dia no intervalo `experiment_days` e identifica o **menor número de dias** necessário para atingir o `power_target` (geralmente 80%).

### 2. Auto-MDE (MDE Inverso)
Se `mde=None`, a função inverte a lógica: para cada duração possível, ela calcula qual é o **Menor Efeito Detectável (MDE)** que pode ser garantido com 80% de poder. Útil quando o tempo de veiculação é fixo e você quer saber a sensibilidade do teste.

### 3. Modo Multi-Cluster (Orquestração DoE)
Se o parâmetro `clusters` for fornecido (saída da `discover_geo_clusters`), a função executa automaticamente:
1. Uma análise individual para cada cluster de tratamento.
2. Uma análise **Consolidada**, que utiliza a variância média dos resíduos de todos os clusters para projetar o poder do experimento como um todo.

## Fundamentação Matemática

1.  **Transformação Log-Diff**: Aplica $\log(Y_t) - \log(Y_{t-1})$ para focar no crescimento relativo e garantir estacionaridade.
2.  **Purificação via OLS**: Realiza uma regressão linear entre Tratamento e Controles para remover sazonalidades comuns. O desvio padrão dos resíduos desta regressão ($\sigma_{reg}$) representa o "ruído puro" que o MDE precisa superar.
3.  **Cálculo de Poder**: Utiliza a distribuição normal para encontrar o ponto onde o efeito injetado ($\Delta$) se torna distinguível do ruído histórico:
    $$ (1-\beta) = \Phi \left( \frac{\Delta}{\sigma_{reg} / \sqrt{n}} - Z_{1-\alpha/2} \right) $$

## Retorno (*Output*)

Retorna um dicionário contendo o sumário executivo e a curva de poder detalhada:

```python
{
    "summary": {
        "mde": 0.01,                 # MDE utilizado ou calculado
        "best_days": 21,             # Sugestão de duração ideal
        "best_power": 0.82,          # Poder atingido no best_days
        "sigma": 0.045,              # Ruído residual (log-diff)
        "delta_abs": 1250.50,        # Impacto incremental absoluto estimado por dia
        "estimated_days_needed": 19, # Projeção teórica exata via T-Student
        "auto_mde": False,           # Indica se o modo Auto-MDE foi usado
        "consolidated": False        # Indica se é uma visão consolidada de múltiplos clusters
    },
    "power_curve": pd.DataFrame,     # Tabela dia a dia de Power e MDE
    "residuals": pd.Series           # Série temporal de resíduos (útil para auditoria)
}
```
