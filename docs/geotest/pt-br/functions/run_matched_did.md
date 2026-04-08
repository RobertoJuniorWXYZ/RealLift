# `reallift.geo.did.run_matched_did`

A função `run_matched_did` compõe a "metade gêmea" do controle causal da biblioteca RealLift. Projetada rigorosamente sobre os alicerces originais do Diferenças-em-Diferenças (DiD), esta via metodológica assume o poder computacional contornando as limitações do *Controle Sintético Convexo*. 

Ao abandonar pesos contínuos dinâmicos em favor da **média perfeitamente uniforme (`[1/N]`)**, o DiD protege inferências executadas sobre baselines altamente instáveis (onde a otimização de matriz L2 falharia através da Supressão de Variância).

## Assinatura

```python
def run_matched_did(
    filepath: str,
    date_col: str,
    treatment_geo: str | list,
    control_geos: list,
    treatment_start_date: str,
    treatment_end_date: str = None,
    start_date: str = None,
    end_date: str = None,
    random_state: int = None,
    cluster_idx: int = None,
    plot: bool = True,
    verbose: bool = True
) -> dict
```

## Fundamentação Matemática e Algorítmica (Panel Data)

O clássico problema de "Discrepância de Volume" entre regiões não impede a série temporal no RealLift devido ao excelente sistema algorítmico interno conhecido como **Efeito Fixo de Unidade** (Normalização).

### 1. Construção Vetorial Nivelada 
Ao operar séries temporais contínuas (Event-Study), as curvas dos doadores raramente possuem a mesma massa de escala da Praça Tratada principal.

O algorítmo inicialmente divide os dados pela média do período pré-intervenção para resgatar não as "vendas", mas sim a curva de *oscilação de mercado*.

$$
X_{norm} = \frac{X_{pre}}{\bar{X}_{pre\_mean}}
$$

### 2. A Interceptação e o Baseline (*Parallel Trends*)

Enquanto a via Convexa necessita descobrir restritamente pesos complexos, o modo diferencial define **Pesos Homogêneos (Axioma de Grupo Controle)**:

$$ w = \left[ \frac{1}{N_{\text{controls}}}, \dots, \frac{1}{N_{\text{controls}}} \right] $$

E projeta magicamente o índice do mercado para as proporções volumétricas base isoladas do local tratado original, assegurando que ambos comecem no mesmo intercepto $\alpha$ antes que a data do tratamento cause ruptura nos gráficos!

$$
Baseline = \left( \sum (X_{norm} \cdot w) \right) \cdot \bar{Y}_{pre\_mean}
$$

### 3. Impacto Relativo Estimado e Reamostragem

Tal como o Controle Sintético, todo Lift absoluto medido dia contra dia no pós-teste é analisado através de **Reamostragem Empírica (Bootstrap Iterativo)** contra a tendência neutra de ruído. Sem envolver a equação gaussiana tradicional de estatística teórica.

## Retorno (*Output*)

A abstração da API é mantida **idêntica** estruturalmente ao retorno de `run_synthetic_control` para permitir roteamentos limpos da pipeline abstrata do `design_of_experiments`.

```python
{
    "weights": {"geo_B": 0.5, "geo_C": 0.5}, # DiD Clássico (1/N uniformes)
    "alpha": 0.0,                            # Normalizado diretamente
    "lift_total": 451.2,                     # Impacto incremental acumulado
    "lift_mean_pct": 0.065,                  # Lift médio percentual (6.5%)
    "pre_mspe": 0.051,                       # Erro na predição de tendência anterior (Painel)
    "post_mspe": 0.982,                      # Distância da ruptura na intervenção
    "mspe_ratio": 19.25,                     # Permissão Placebo Pós/Pré
    "bootstrap": { ... },                    # Confidence Intervals
    "df": pd.DataFrame,                      # Matriz pós-transformação consolidada
    "plotting_data": { ... }                 # Facilita visualizações desacopladas do verbose
}
```
