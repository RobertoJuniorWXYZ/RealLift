# `reallift.geo.discovery.discover_geo_clusters`

Na biblioteca RealLift, a função `discover_geo_clusters` é o motor principal para o design experimental de testes incrementais baseados em geografia (Geo Experiments). Sua finalidade é prever e identificar a combinação ótima de sub-regiões de controle para formar um **Controle Sintético** de alta correlação e baixo erro estocástico em relação a uma região de tratamento alvo, **antes do período da intervenção.**

## Assinatura

```python
def discover_geo_clusters(
    filepath: str,
    date_col: str,
    geos: list = None,
    n_treatment: int = 3,
    fixed_treatment: list = None,
    start_date: str = None,
    end_date: str = None,
    use_elasticnet: bool = True,
    search_mode: str = "auto",
    verbose: bool = True,
    show_results: bool = True
) -> list
```

## Como Funciona: Passo a Passo Intuitivo

O processo de descoberta de "clusters" geográficos resolve um problema de otimização para encontrar quem "imita" a sua região de tratamento da melhor forma antes da intervenção. O algoritmo opera nas seguintes fases principais:

### 1. Preparação dos Dados
A função organiza os dados de série temporal: agrupa por data somando métricas duplicadas e garante que os dados estejam ordenados cronologicamente. Opcionalmente, se `treatment_start_date` for fornecido, a função filtra os dados para enxergar apenas o período *antes* da intervenção (evitando *data leak*).

### 2. Definindo "Quem é de Quem" (Tratamentos vs. Doadores)
O comportamento de busca muda dependendo de como a função é chamada:
- **Sem tratamentos fixos (`fixed_treatment = None`)**: O algoritmo trabalha no **modo exploratório**. Ele testa todas as combinações possíveis de `n_treatment` geografias operando como tratadas contra o resto como grupo de controle (*donor pool*).
- **Com tratamentos fixos (`fixed_treatment = [...]`)**: O algoritmo trabalha no **modo direcionado**. Ele avalia cada região definida de forma **individual**, buscando os melhores doadores exclusivos para cada uma. As outras regiões de tratamento são isoladas na "lista proibida" para não contaminarem os controles umas das outras.

### 3. Transformação da Série Temporal (`log_diff_transform`)
Para evitar que o modelo seja enganado por tendências de longo prazo ou distorções de escala, aplica-se uma transformação de log e diferença. Assim, o modelo foca em alinhar a volatilidade e o *crescimento relativo dia após dia*, tornando a série estacionária.

### 4. Filtrando a "Multidão" (ElasticNet)
Com muitas cidades no controle, comparar todas geraria ruído (*overfitting*). A função utiliza um modelo de regressão **ElasticNet** que avalia os doadores e naturalmente "zera" a relevância das cidades que não ajudam a explicar o tratamento. 

*Fórmula da Otimização ElasticNet minimizada:*

$$
\min_{w} \frac{1}{2n} ||Xw - y||^2_2 + \alpha \cdot L1_{ratio} \cdot ||w||_1 + 0.5 \cdot \alpha \cdot (1 - L1_{ratio}) \cdot ||w||_2^2
$$

**A Regra do Positivo:** Se uma cidade candidata recebe um peso negativo (ou seja, ela cresce quando o alvo cai), ela é permanentemente descartada. Controles sintéticos exigem similaridade real, e não correlações espelhadas irreais.

### 5. Otimizando os Pesos (O Controle Sintético Real)
As cidades que sobrevivem ao filtro anterior passam por uma otimização matemática rigorosa (via `cvxpy`) baseada nos dados originais. O objetivo é encontrar os **pesos percentuais** perfeitos que:
1. Sejam maiores ou iguais a zero ($w_i \ge 0$).
2. Somem exatamente 100% ($\sum_{i} w_i = 1$).

Se alguma cidade receber um peso insignificante (abaixo de 0.1%), ela é podada do *donor pool* definitivo.

### 6. Avaliação e Ranqueamento (Synthetic Error Ratio)
O algoritmo final avalia as oscilações preditas em relação às reais com o grupo vencedor e ranqueia as opções baseando-se em duas variáveis base:
- **Resíduo Padrão (`std_residual`)**: Mede o erro absoluto histórico diário entre o real e o simulado. Quanto mais "largo" o erro, menos precisa é a cidade (queremos baixo).
- **Correlação (`correlation`)**: Mede o nível de sincronismo dos movimentos dos "picos e vales" das linhas de predição (queremos alto).

Para resolver desempates clássicos da Inferência Causal (ex: modelos muito correlatos porém com altos níveis absolutos de erro que alargam o intervalo de confiança do Geo-Teste), a função cria uma heurística penalizadora chamada **Synthetic Error Ratio**:

*Métrica de Erro Sintético (Synthetic Error Ratio)*: `std_residual / (correlation + 1e-6)`

O algoritmo divide a largura da banda de erro pela correlação, forçando o ranqueamento priorizar severamente modelos cujas curvas possuam o menor desvio padrão histórico, *desde que* sigam juntas. Por isso o "Cluster 0" garantidamente representará o Controle Sintético historicamente estático e metodologicamente mais seguro de simulação e menos ruidoso (menor Synthetic Error Ratio).

> [!NOTE]
> **Nota Acadêmica sobre a Heurística Operacional**
> Na estatística clássica (Abadie et al., 2010), a otimização de Controles Sintéticos visa exclusivamente a minimização do Erro Quadrático Médio de Predição (RMSPE) no período pré-tratamento. Contudo, em Causalidade Aplicada a negócios e marketing, a minimização absoluta do RMSPE sofre de um fenômeno conhecido como *supressão de variância*, onde a modelagem pode selecionar matrizes "fixas" ou "planas" que historicamente têm erro passivo baixo, mas não reagem de forma equivalente a choques sazonais exógenos.
> 
> O **Synthetic Error Ratio** não se propõe a ser um novo teorema fundamental, mas sim atua como uma **Função de Perda Regularizadora** (Loss Function) puramente pragmática. Assumindo a Correlação como o "Sinal" (comportamento estrutural latente) e o Resíduo como o "Ruído" (variância estocástica inútil), a divisão matemática simula a maximização inversa do *Signal-to-Noise Ratio (SNR)*. Isso força empiricamente a exclusão dos famigerados "Controles Zumbis" e garante Robustez Fora da Amostra (Out-of-Sample) frente à instabilidade do mundo corporativo moderno.
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
        "control_weights": [0.65, 0.35], # Pesos percentuais (somam 1.0) de cada controle no sintético
        "correlation": 0.8419,         # Sincronia de Pearson entre Sintético e Tratado
        "std_residual": 0.052,         # Resíduo Padrão (RMSE equivalente)
        "synthetic_error_ratio": 0.0617, # Heurística de penalização (std_residual / correlation)
        "n_controls": 2,               # Cardinalidade da Otimização final
        "alpha": 0.01,                 # Tuning do melhor viés hiperparamétrico 
        "l1_ratio": 0.5                # Razão de esparso/denso escolhido pelo Ridge/Lasso
    }
]
```

## Exemplo de Uso

```python
from reallift.geo.discovery import discover_geo_clusters

# Exemplo Clássico: Forçando agrupamentos de controle para unidades de Campanha Ativas.
melhores_clusters = discover_geo_clusters(
    filepath="base_historica_vendas.csv",
    date_col="data_faturamento",
    fixed_treatment=["sao_paulo", "rio_de_janeiro"], 
    start_date="2026-01-01"
)

for sub_testes in melhores_clusters:
     print(f"Alvo: {sub_testes['treatment'][0]} => Explicado por: {sub_testes['control']} ")
```
