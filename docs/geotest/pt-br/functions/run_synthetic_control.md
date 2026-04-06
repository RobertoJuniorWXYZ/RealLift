# `reallift.geo.synthetic.run_synthetic_control`

A função `run_synthetic_control` é o núcleo de inferência causal do RealLift. Ela utiliza Otimização Convexa para construir um contrafactual (Controle Sintético) que mimetiza o comportamento da unidade tratada no período anterior à intervenção, permitindo calcular o impacto incremental real.

## Assinatura

```python
def run_synthetic_control(
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

## Fundamentação Matemática e Algorítmica

### 1. Otimização Convexa (Período Pré-Teste)
O algoritmo isola os dados anteriores ao `treatment_start_date` e busca os pesos $w$ e o intercepto $\alpha$ que minimizam o erro quadrático entre o real e o sintético:

$$ \min_{w, \alpha} \| Y_{pre} - (X_{pre} \cdot w + \alpha) \|^2 $$

**Restrições Convexas:**
- Todos os pesos são não-negativos ($w_i \ge 0$).
- A soma dos pesos é exatamente 1 ($\sum w = 1$).

O **Intercepto Convexo ($\alpha$)** é uma inovação pragmática que absorve diferenças sistemáticas de nível (vendas médias diferentes), permitindo que os pesos foquem puramente em alinhar a *correlação* e o *comportamento* das séries.

### 2. Projeção Contrafactual (Período Pós-Teste)
Uma vez encontrados os pesos ideais, eles são aplicados aos dados do período de intervenção. O **Lift** é a diferença acumulada entre o que a geografia realmente vendeu ($Y_{post}$) e o que o modelo previu que ela venderia caso não houvesse marketing ($\hat{Y}_{post}$):

$$ \text{Lift} = \sum (Y_{post} - \hat{Y}_{post}) $$

### 3. Significância via Bootstrap
Em vez de depender de premissas paramétricas rígidas, o RealLift utiliza **re-amostragem (Bootstrap)** sobre os resíduos históricos para gerar intervalos de confiança de 95% e determinar se o Lift observado é estatisticamente diferente de zero.

## Retorno (*Output*)

Retorna um dicionário com toda a inteligência do experimento:

```python
{
    "weights": {"geo_A": 0.6, "geo_B": 0.4}, # Pesos convexos dos doadores
    "alpha": 12.5,                           # Valor do intercepto (correção de nível)
    "lift_total": 5212.0,                    # Impacto incremental total acumulado
    "lift_mean_pct": 0.045,                  # Lift médio percentual (4.5%)
    "pre_mspe": 0.0012,                      # Erro quadrático médio no pré-teste
    "post_mspe": 0.2540,                     # Erro quadrático médio no pós-teste
    "mspe_ratio": 211.6,                     # Razão Post/Pre (métrica de robustez)
    "bootstrap": { ... },                    # Intervalos de confiança e p-valor boot
    "df": pd.DataFrame,                      # Base de dados processada com real e sintético
    "plotting_data": { ... }                 # Estruturas prontas para visualização
}
```
