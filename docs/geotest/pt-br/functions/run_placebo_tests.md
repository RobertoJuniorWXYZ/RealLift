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
    verbose: bool = True,
    experiment_type: str = "synthetic_control"
) -> dict
```

## Mecânica de Razão MSPE (Post/Pre)

Em vez de comparar apenas o Lift absoluto, o RealLift adota a metodologia de **Razão MSPE (Mean Squared Prediction Error)**, conforme proposto por Abadie et al. (2010). Esta abordagem é superior pois normaliza o erro do período de intervenção pelo erro de ajuste histórico (pré-teste) daquela geo específica.

*Nota Estrutural importante: A função de placebo herda a rigorosidade do modelo selecionado (parâmetro `experiment_type`). Se você operou um painel de Diferenças-em-Diferenças (`matched_did`) a função placebo garantirá que as simulações falsas obedeçam a restrição de pesos uniformes 1/N. Caso o modelo mestre tenha sido Controle Sintético (`synthetic_control`), o placebo usará pesos otimizados via Convexidade.*

1.  **Iteração Placebo**: Para cada geo no pool de controle, o modelo tenta criar a simulação de controle exata que usaria no passo avaliado, tratando a geo isolada como se fosse o alvo irreal de uma campanha publicitária.
2.  **Cálculo da Razão**: Para cada teste (incluindo o real), calculamos:
    $$
    Ratio = \frac{MSPE_{Post}}{MSPE_{Pre}}
    $$
3.  **Vantagem Crítica (Proteção contra Dados Voláteis)**: O uso da razão nos protege em cenários modernos de dados e de *Matched DiD* sujos. Caso uma localidade placebo do passado já operasse sob forte ruído natural e a predição da tendência não fosse excelente ($MSPE_{Pre}$ alto), um desvio alto no pós-teste seria minimizado em sua divisão, não estritamente considerado tão "excepcional". Por outro lado, caso presenciassemos uma tendência estática (pré-teste perto de zero), o erro mínimo do MSPE dispararia a Razão ao identificar as quebras. Isso efetivamente impede falsos positivos punindo localidades ruidosas.

### Densidade Probabilística (Empirical p-value formulation)

O $p_{value}$ agora representa a probabilidade de encontrarmos uma Razão MSPE tão alta quanto a observada no geo de tratamento original por puro acaso:

$$
P_{Value\_Empirico} = \frac{\sum_{i=1}^{P} \mathbf{I} \big(Ratio_{placebo\_i} \ge Ratio_{observed}\big)}{P}
$$

*(Onde $\mathbf{I}(\cdot)$ assume valor 1 caso a condição seja verdadeira).*

Resultados de $p \le 0.10$ acenderão métricas de sucesso visual (`✔ High confidence:`), corroborando que a anomalia gerada pela sua campanha é estatisticamente rara e não explicável pelo ruído natural das outras regiões.
