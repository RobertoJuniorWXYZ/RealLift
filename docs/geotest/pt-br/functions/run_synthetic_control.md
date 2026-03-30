# `reallift.geo.synthetic.run_synthetic_control`

O apogeu da inferência causal em Geo Experiments, a sub-redator `run_synthetic_control` instila abordagens frequentistas/bayesianas pesadas para formalizar o **Causal Lift (Efeito Incremental)**. Aplicada logo em repouso da conclusão formal da intervenção territorial, ela orquestra o *Synthetic* contrafactual da curva total paralela associada ao *Bootstrap Significance*.

## Assinatura

```python
def run_synthetic_control(
    filepath: str,
    date_col: str,
    treatment_geo: str | list,
    control_geos: list,
    treatment_start_date: str,
    random_state: int = None,
    cluster_idx: int = None,
    plot: bool = True,
    verbose: bool = True
) -> dict
```

## Fundamentação Matemática e Algorítmica

### Otimização Convexa Linear (Treino Período Pré)
As colunas e tensores indexados em observação sofrem cisão perante a *boundary* interposta pelo `treatment_start_date`. A matriz contendo as datas anteriores ($Y_{pre}$ e $X_{pre}$) passa por escalonamento rigoroso em referencial normativo paramétrico às médias.
Os hiper-campos então despacham a problemática como uma fronteira de redução de perdas unida para dentro dos blocos internos do pacote *SciPy cvxpy*, engatilhado com a engine C++ Conic Solver (SCS). O objetivo é isolar interceptos isométricos e pesos convexos puristas (onde todos os pesos $w_i \ge 0$ são não negativos, com soma final trancada à obrigatoriedade algorítmica imperfeita de $\sum w = 1.0$).

### Projeção Diferencial Contrafactual (Efeito Absoluto Pós)
O solver descobre a "Super Cifra" de pesos globais ($W$). Esses pesos são transportados e transpostos imutavelmente atrelados à variação temporal futura da base inexplorada do grupo de controles pós-período:

$$ \hat{Y}_{post} = \sum X_{post} \cdot W_{SCS} + \alpha_{SCS} $$

Subtraindo por conseguinte a contagem bruta vendida real ($Y_t$ oficial da cidade alvo) perante sua projeção sintética fictícia construída pelos pares do estado, atinge-se limpidamente o valor exato incremental do Projeto de Intelecção Causal ($E_{Lift} = Y_{post} - \hat{Y}_{post}$). 

### Limítrofe Analítico *Bootstrap* (Empírico)
Refutando as *T-Test distributions* baseadas na prerrogativa inatingível de regressão perfeita que a maior parte dos cientistas falha, a biblioteca foca num núcleo re-amostral por métricas não-paramétricas ($Bootstrap Significance$).
Em paralelo, arrays randômicos gigantes baseados na semente `random_state` engaiolam o deltas de resíduo do efeito. Permutação em amostragem por reposição traça estocasticamente percentis com Intervalos de Confiança rigorosíssimos (95% *CI*), apontando limites de inferência superiores e inferiores ($CI_{Upper}$ e $CI_{Lower}$) independentes das distorções diárias da métrica orgânica.

## Retorno (*Output*)

O terminal exibe relatórios profundos do T-Test paramétrico e seus arrays limpos da operação:

```python
{
    "weights": {"geo_B": 0.5, "geo_C": 0.5},
    "lift_total": 5212.14,           # Faturamento Puro Oculto Adicionado no Post
    "lift_mean_abs": 150.0,          # Delta absoluto incremental pro dia-médio
    "lift_mean_pct": 0.041,          # Delta percentual incremental (ex: 4.1% a mais de volume)
    "bootstrap": { ... }             # Metadados de Confidence Intervals 95% empíricos
    "plotting_data": { ... }         # Curvas prontas em matriz para gerar gráficos matplotlib
}
```
