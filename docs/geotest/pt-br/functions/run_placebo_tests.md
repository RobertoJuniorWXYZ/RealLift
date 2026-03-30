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

## Mecânica Estocástica de Inversão de Espaços Falsos

A execução dessa preposição instiga rigorosamente a emulação de permuta matemática chamada *Leave-One-Out Placebo Refutation*:
1. Um for-loop entra sistematicamente na base matricial iterando o balde das populações classificáveis de Controle genuínas não-testadas do seu GeoLift.
2. Promove-se artificialmente e momentaneamente a coroa de "Cidade Tratada ($Y$ alvo falsa)" para a $N_{ésima}$ cidade de controle do grid, repulando as coadjuvantes que sobraram nos laços para modelarem a contrafactual de falso-controle para ela.
3. Roda internamente um bloco cego assíncrono hiper-custoso da mesma matriz `run_synthetic_control`. Como a cidade placebo não sofreu nenhuma campanha real subjacente em `treatment_start_date`, espera-se logicamente que a margem delta residual (o pseudo-*Lift* captado) seja extremamente tendenciosa ao 0 absoluto.

### Densidade Probabilística (*Empirical p-value formulation*)
Ao final da estafante malha de inferência iterada randômica no teto limite tolerável `n_placebos`, junta-se as flutuações amostrais falsas num *Array* $|L_{placebo}|$ em formato não-sinalizado absoluto.
Deriva-se então a proeminência exata com intersecção ao seu verdadeiro e único Lift factual de projeto capturado ($|L_{observed}|$) pela equação baseada em simulação Monte Carlo restritiva:

$$ P_{Value\_Empirico} = \frac{\sum_{i=1}^{P} \mathbf{I} \big(|L_{placebo\_i}| \ge |L_{observed}|\big)}{P} $$

*(Onde $\mathbf{I}(\cdot)$ assume representatividade modular unitária 1 de indicador relacional caso falhe localmente).*

Quanto mais contrafactuais sintéticos aleatórios cruzarem sem motivos o altíssimo valor financeiro e numérico provocado no Lift verdadeiro da cidade-alvo real sua, maior se agitará o $p_{value}$, destitíndo estatisticamente a exclusividade causal e provando a quebra formal do rigor A/B.

Resultados menores que logarítimo estrito $\alpha = 0.1$ acenderão métricas de sucesso visual (`✔ High confidence:`), corroborando seu lucro numérico gerado como altamente único e impermeável às tendências flutuantes de variáveis inorgânicas naturais no ambiente de *hold-outs*.
