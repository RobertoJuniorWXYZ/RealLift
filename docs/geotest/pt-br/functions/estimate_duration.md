# `reallift.geo.duration.estimate_duration`

A função `estimate_duration` fundamenta diretamente os pilares estatísticos de *Power Analysis* e Dimensionamento Amostral do GeoLift. Exigido um Elevador Mínimo Detectável percentual (MDE) pela equipe financeira (ou estratégia), essa lógica quantifica a inércia basal da cidade controle somada e requer formalmente **quantos dias ininterruptos** o teste necessita existir em veiculação post-treatment para cimentar $\ge 80\%$ de poder fidedigno (minimizar a Taxa do Tipo II de Falsos Negativos).

## Assinatura

```python
def estimate_duration(
    filepath: str,
    date_col: str,
    treatment_geo: str | list,
    control_geos: list,
    mde: float = 0.01,
    alpha: float = 0.05,
    power_target: float = 0.8,
    max_days: list = [21, 60],
    treatment_start_date: str = None,
    cluster_idx: int = None,
    verbose: bool = True
) -> dict
```

## Fundamentação Matemática

1. **Transformação Log-Diff**: Para padronizar permanentemente a escala elástica do erro em Power Analysis de long-tail, a geometria do painel absorve a transformação do limiar $\log(Y_t) - \log(Y_{t-1})$. Unidades de intervenção múltiplas assumem seus blocos unificados por médias globais.
2. **Resíduo de Regressão Basal OLS**: Processa um estimador canônico OLS (`LinearRegression`) cruzando o Tratamento com os Controles nos deltas subjacentes logarítmicos. O OLS purga a variância sistêmica sazonal comum e nos fornece o **puro ruído assíncrono indetectável**, representado pelo desvio padrão de *Mismatch* $\sigma_{reg}$.
3. **Poder sobre Distribuição Normal**: Retrógrado converte a razão `mde` infundida numa métrica de deslocamento ($Y$) Logarítmico (Delta Efeito Absoluto $\Delta$). E modela iterativamente o limiar Gaussian da Força $(1-\beta)$ a cada período de exposição ($n$ dias):

$$ (1-\beta) = \Phi \left( \frac{\Delta}{\sigma_{reg} / \sqrt{n}} - Z_{1-\alpha/2} \right) $$

O solver itera contiguamente o $n$ dia a dia, constrito na margem declarada em `max_days`. O ciclo interrompe quando a integral cruzada da Força suplanta definitivamente a variável `power_target` ($\approx 0.80$).

## Retorno (*Output*)

Essa interface constrói um panorama completo do delineamento ideal:

```python
{
    "summary": {
        "mde": 0.01,                 # MDE configurado em decimal
        "best_days": 21,             # Solução do solver de dias mínimos
        "best_power": 0.8123         # Efetivo poder atingido no melhor dia (81%)
        "estimated_days_needed": ... # Predição da função normal (t-student) forçando extrapolação se não houver best_days
    },
    "power_curve": DataFrame         # Representação integral e iterativa gaussiana dia/dia
}
```
