# `reallift.geo.split.find_best_geo_clusters`

Na biblioteca RealLift, a função `find_best_geo_clusters` é o motor principal para o design experimental de testes incrementais baseados em geografia (Geo Experiments). Sua finalidade é prever e identificar a combinação ótima de sub-regiões de controle para formar um **Controle Sintético** de alta correlação e baixo erro estocástico em relação a uma região de tratamento alvo, **antes do período da intervenção.**

## Assinatura

```python
def find_best_geo_clusters(
    filepath: str,
    date_col: str,
    geos: list = None,
    n_treatment: int = 3,
    fixed_treatment: list = None,
    treatment_start_date: str = None,
    verbose: bool = True
) -> list[dict]
```

## Fundamentação Matemática e Algorítmica

O processo de descoberta de "clusters" geográficos resolve um problema de otimização combinatória unida à inferência de controle sintético estrito. O algoritmo opera nas seguintes fases principais:

### 1. Transformação de Séries Temporais (`log_diff_transform`)
Dado que séries temporais de métricas de negócio (conversões, faturamento, acessos) exibem forte sazonalidade e tendências não-estacionárias, os dados brutos ($Y_t$) das geografias avaliadas são transformados parametricamente via logaritmos e diferenciações. Essa aplicação estabiliza a variância da série temporal, remove distorções de magnitude e força a estacionariedade, possibilitando que os modelos subsequentes foquem em alinhar variações e não apenas os valores absolutos.

### 2. Definição Iterativa da *Control Pool* (Piscina de Controle)
Para cada combinação de cidades a receber tratamento causal, uma "piscina de controle" ($X$) mutuamente exclusiva é instanciada. Durante execuções com o parâmetro `fixed_treatment`, o algoritmo isola iterativamente cada geografia alvo. É garantido topologicamente que nenhuma unidade sendo testada cause interferência ou colisão nos tensores precursores de seus próprios controles.

### 3. Eliminação e Regularização Dinâmica (`ElasticNet`)
Não se pode assumir empiricamente que todas as unidades na piscina sejam boas preditoras simuláveis — algo que invariavelmente leva a casos de *Overfitting* Espúrio. Consequentemente, uma regressão linear fortemente regularizada (`sklearn.linear_model.ElasticNet`) varre as covariáveis em um *grid-search* ($\alpha$ e *L1 ratio*). 

O modelo lida com a dispersão minimizando a função objetiva:

$$ \min_{w} \frac{1}{2n_{samples}} \|Xw - y\|_2^2 + \alpha \cdot \text{l1\_ratio} \cdot \|w\|_1 + 0.5 \cdot \alpha \cdot (1 - \text{l1\_ratio}) \cdot \|w\|_2^2 $$

**A Regra do Positivo:** Qualquer característica (cidade/unidade candidata) a qual o método de redução assinale um peso de regressão $w \le 0$ é permanentemente descartada. O dogma da construção de Controles Sintéticos dita que uma região irmã não deve possuir correlações espelhadas irreais (espectro negativo) frente o alvo.

### 4. Otimização Convexa do Controle Sintético
As covariáveis que sobrevivem à restrição avançam para uma fase estrita de Otimização Convexa (via pacote `cvxpy`). As matrizes são equalizadas por escala e um *solver* encontra valores de composição vetorial não-restritivos que respeitem irredutivelmente:

1. **Restrição de Não-Negatividade Estrita**: $w_i \ge 0$
2. **Aditividade Absoluta (Soma 1)**: $\sum_{i} w_i = 1$

Essa etapa condensa a liga da sua linha de base. Unidades com representatividade ínfima ou variância ruidosa têm seus vetores suprimidos para $\approx 0$, e ao caírem abaixo de `0.001` de significância estatística, são expurgadas pelo algorítmo.

### 5. Estimação Residual e Ranqueamento (The Fit)
O modelo final é parametrizado exclusivamente com aquelas de alta aderência, devolvendo duas estatísticas basilares de *Goodness-Of-Fit*:
- **Resíduo Padrão (`std_residual`)**: O desvio padrão dos resíduos matemáticos das frações da predição $\sigma(y - \hat{y})$.
- **Correlação Cruzada de Pearson (`correlation`)**: Sincronismo da trajetória da covariância histórica linear.

Todas as alternativas são ranqueadas dividindo o fator de erro pela correlação (`std_residual / (correlation + 1e-6)`), beneficiando curvas idênticas que nunca desviam de forma severa.

---

## Detalhamento de Parâmetros

- **`filepath`** *(str)*: Caminho absoluto ou relativo para o arquivo CSV de entrada (fator diário por geografias).
- **`date_col`** *(str)*: Coluna string/datetime representando a data de cada observação sequencial.
- **`geos`** *(list, opcional)*: Especifica ou restringe globalmente a gama total de praças passíveis de leitura na base de dados.
- **`n_treatment`** *(int, default=3)*: Utilizado caso `fixed_treatment` for `None`. O software gerará partições combinatórias contendo esse *N* número de lugares em cada simulação A/B pseudo-aleatória.
- **`fixed_treatment`** *(list, opcional)*: Lista nominal das geografias que sua equipe ativamente escolheu intervir. Ignora o combinatório `n_treatment` e analisa o cenário ótimo focado unicamente para salvaguardar os dados dos alvos escolhidos.
- **`treatment_start_date`** *(str, opcional)*: Parâmetro valioso no formato `YYYY-MM-DD`. Truca todos os dados retroalimentares apartir desse dia, certificando de testar as sinergias passadas **sem influência** do próprio tratamento (Data Leak).
- **`verbose`** *(bool, default=True)*: Retorna os logs progressivos e a listagem de sucesso na saída padrão (STDOUT).

---

## Retorno e Estrutura

*(list)*
Retorna lista de dicionários mapeados com a melhor parametrização combinada. A recomendação vencedora é sempre residente no index `0`. A resposta segue este *Schema*:

```python
[
    {
        "treatment": ["geo_target_A"], # Geografia Tratada Central
        "control": ["geo_B", "geo_C"], # Composição Sintética Ideal de Controles
        "correlation": 0.8419,         # Sincronia de Pearson entre Sintético e Tratado
        "std_residual": 0.052,         # Resíduo Padrão (RMSE equivalente)
        "n_controls": 2,               # Cardinalidade da Otimização final
        "alpha": 0.01,                 # Tuning do melhor viés hiperparamétrico 
        "l1_ratio": 0.5                # Razão de esparso/denso escolhido pelo Ridge/Lasso
    }
]
```

## Exemplo de Uso

```python
from reallift.geo.split import find_best_geo_clusters

# Exemplo Clássico: Forçando agrupamentos de controle para unidades de Campanha Ativas.
melhores_clusters = find_best_geo_clusters(
    filepath="base_historica_vendas.csv",
    date_col="data_faturamento",
    fixed_treatment=["sao_paulo", "rio_de_janeiro"], 
    treatment_start_date="2026-04-10"
)

for sub_testes in melhores_clusters:
     print(f"Alvo: {sub_testes['treatment'][0]} => Explicado por: {sub_testes['control']} ")
```
