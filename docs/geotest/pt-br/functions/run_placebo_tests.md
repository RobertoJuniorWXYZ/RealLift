# `reallift.geo.placebo.run_placebo_tests`

A elegante matriz rotacional `run_placebo_tests` incorpora o que chamamos convencionalmente em Data Science de testes *In-Space Placebo* e/ou Validações Falsificacionistas de Efeito Exato (*Falsification Test*). É a provação fundamental exigida pelo método científico para evidenciar estatisticamente que se não existisse nenhuma ação real da sua campanha publicitária sobre o mercado, você **nunca acharia um Lift aleatório daquela agressividade por mera causalidade** induzindo desvios nas outras cidades neutras.

## Assinatura

```python
def run_placebo_tests(
    filepath: str,
    date_col: str,
    control_geos: list,
    treatment_start_date: str,
    observed_lift: float,
    n_placebos: int = 10,
    random_state: int = None,
    cluster_idx: int = None,
    plot: bool = True,
    verbose: bool = True
) -> dict
```

## Mecânica de Razão MSPE (Post/Pre)

Em vez de comparar apenas o Lift absoluto, o RealLift adota a metodologia de **Razão MSPE (Mean Squared Prediction Error)**, conforme proposto por Abadie et al. (2010). Esta abordagem é superior pois normaliza o erro do período de intervenção pelo erro de ajuste histórico (pré-teste) de cada geo:

1.  **Iteração Placebo**: Para cada geo no pool de controle, o modelo tenta criar um controle sintético para ela, tratando-a como se fosse o alvo do experimento.
2.  **Cálculo da Razão**: Para cada teste (incluindo o real), calculamos:
    $$
    Ratio = \frac{MSPE_{Post}}{MSPE_{Pre}}
    $$
3.  **Vantagem**: Se uma cidade é naturalmente ruidosa e o modelo não encaixou bem no pré-teste ($MSPE_{Pre}$ alto), um desvio alto no pós-teste não será considerado tão "anormal". Por outro lado, se o encaixe foi perfeito e houve um desvio súbito após a data da campanha, a Razão será altíssima, indicando um efeito único.

### Densidade Probabilística (Empirical p-value formulation)

O $p_{value}$ agora representa a probabilidade de encontrarmos uma Razão MSPE tão alta quanto a observada no geo de tratamento original por puro acaso:

$$
P_{Value\_Empirico} = \frac{\sum_{i=1}^{P} \mathbf{I} \big(Ratio_{placebo\_i} \ge Ratio_{observed}\big)}{P}
$$

*(Onde $\mathbf{I}(\cdot)$ assume valor 1 caso a condição seja verdadeira).*

Resultados de $p \le 0.10$ acenderão métricas de sucesso visual (`✔ High confidence:`), corroborando que a anomalia gerada pela sua campanha é estatisticamente rara e não explicável pelo ruído natural das outras regiões.
