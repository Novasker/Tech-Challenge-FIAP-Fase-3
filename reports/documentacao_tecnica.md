# Documentação técnica - Fase 3

Este documento complementa o README. Aqui estão os detalhes de implementação: tabelas, dicionário de dados, pipeline, decisões analíticas e problemas que apareceram durante o desenvolvimento.

## Ambiente

- Databricks Free Edition (computação serverless), catálogo `workspace`
- Schemas usados: `bronze`, `silver`, `gold`, `staging`
- As camadas bronze, silver e gold da Fase 2 foram reconstruídas neste workspace
- A chave de acesso ao BigQuery (Base dos Dados) fica num secret scope (`gcp-bigquery`, chave `service-account-key`) e nunca no repositório
- Python com scikit-learn, pandas, numpy, matplotlib e shap (ver `requirements.txt`)

## Tabelas usadas e criadas

Criadas nesta fase:

- `bronze.ibge_population`: população por município e ano (IBGE, via Base dos Dados, anos de 2021 a 2025)
- `gold.base_modelagem_alunos`: base de modelagem no nível do aluno
- `gold.risco_municipios`: risco e tendência por município

Lidas, vindas da Fase 2:

- `silver.alunos_tratado`, `silver.resultado_municipio`
- `bronze.municipio_directory`
- `gold.indicador_alfabetizacao_municipio`

## Dicionário de dados

### gold.base_modelagem_alunos

Um aluno por linha. Só alunos que preencheram o caderno, das redes Municipal e Estadual. 3.354.637 linhas, particionada por `ano`.

| Coluna | Descrição |
|---|---|
| ano | Ano da avaliação (2023 ou 2024) |
| id_municipio | Código IBGE do município (texto, 7 dígitos) |
| rede | Rede de ensino (Municipal ou Estadual), decodificada pelo dicionário da Base dos Dados |
| id_escola | Identificador da escola |
| id_aluno | Identificador do aluno |
| caderno | Versão do caderno de prova |
| serie | Série (sempre 2º ano) |
| alfabetizado | Alvo. "1" se a proficiência é de 743 ou mais, "0" caso contrário |
| peso_aluno | Peso amostral do Saeb. Não usado como variável |
| nome_uf | Estado |
| nome_regiao | Região do país |
| amazonia_legal | 1 se o município está na Amazônia Legal, 0 caso contrário |
| previous_year_municipal_rate | Taxa de alfabetização do município no ano anterior (nula para 2023) |
| possui_meta_municipal | Se o município tem meta municipal definida (nula para 2023) |

Colunas removidas de propósito: `proficiencia` (define o alvo), `presenca` e `preenchimento_caderno` (viram constantes depois do filtro).

### Variáveis criadas no código (pandas)

| Coluna | Descrição |
|---|---|
| target | `alfabetizado` convertido para inteiro (0 ou 1) |
| school_size | Número de alunos avaliados na escola naquele ano |
| municipality_size | Número de alunos avaliados no município naquele ano |

Os dois tamanhos são só contagens, calculados antes da amostragem para não serem distorcidos por ela.

### gold.risco_municipios

Um município da rede municipal por linha.

| Coluna | Descrição |
|---|---|
| id_municipio, municipio_nome, nome_uf, nome_regiao, amazonia_legal | Identificação e território |
| meta_alfabetizacao_2030 | Meta municipal de 2030 |
| rate_2023, rate_2024 | Taxa de alfabetização de cada ano |
| students_2023, students_2024 | Alunos avaliados em cada ano |
| reliable | Verdadeiro se tem pelo menos 50 alunos nos dois anos |
| yearly_change | rate_2024 menos rate_2023 |
| current_gap | Meta de 2030 menos rate_2024 |
| projected_2030 | rate_2024 mais 6 vezes a variação anual, limitado entre 0 e 100 |
| projected_gap | Meta de 2030 menos projected_2030 |
| will_miss_goal | Verdadeiro se projected_2030 fica abaixo da meta |

## Código

Estrutura do pacote `src/`:

- `preprocessing/build_dataset.py`: lê a tabela gold, cria `school_size`, `municipality_size` e `target`, converte `amazonia_legal` para texto e devolve uma amostra em pandas (15%, semente 42)
- `preprocessing/features.py`: monta o pré-processamento
- `modeling/pipelines.py`: junta pré-processamento e modelo num único `Pipeline` (regressão logística ou gradient boosting)
- `evaluation/validation.py`: divisão por município e validação cruzada agrupada

Notebook `notebooks/02_modeling.py`: carrega os dados, roda os experimentos A, B e C, treina o modelo final, escolhe o corte, calcula a importância das variáveis (permutação e SHAP), o risco por município e os grupos.

### Pré-processamento

- Numéricas (`school_size`, `municipality_size`): imputação pela mediana, `log1p`, padronização
- Categóricas (`rede`, `nome_uf`, `amazonia_legal`): imputação pelo valor mais frequente e one-hot encoding (categorias nunca vistas são ignoradas)
- Tudo dentro do `Pipeline`, então é ajustado só com os dados de treino de cada divisão

### Validação

- Teste final: 20% dos municípios separados (`GroupShuffleSplit`), usados só no fim
- Validação cruzada: 5 divisões agrupadas por município (`GroupKFold`)
- Ajuste de parâmetros: `GridSearchCV` agrupado, 3 divisões, em 100 mil alunos do treino, métrica AUC

## Resultados dos experimentos

| Experimento | Variáveis | AUC de validação |
|---|---|---|
| A | aluno, território, tamanhos | 0,630 (logística e boosting) |
| B | A mais taxa do ano anterior (só 2024) | 0,650 (logística), 0,654 (boosting) |
| C | A mais população | 0,630 (logística), 0,628 (boosting) |

Modelo final: regressão logística com as variáveis do experimento A, parâmetro `C` igual a 1.

Teste final (municípios não vistos): AUC 0,625, F1 0,734, acurácia 0,614 (baseline de 0,599). Com corte de 0,6: acurácia balanceada 0,588, recall dos não alfabetizados 0,613, precisão 0,484.

Importância das variáveis (SHAP médio absoluto): estado 0,333, tamanho do município 0,084, Amazônia Legal 0,063, rede 0,047, tamanho da escola 0,015.

## Decisões analíticas

- **Definição do alvo.** `alfabetizado` é exatamente `proficiencia >= 743`. Conferimos nos dados: entre os alunos com nota, 1.984.546 estão acima do corte e todos são alfabetizados, e 1.370.115 estão abaixo e todos são não alfabetizados. Por isso a nota não pode entrar como variável.
- **Alunos sem prova.** 513.338 alunos não preencheram o caderno (13,3%). Todos estão marcados como não alfabetizados e não têm nota. Foram removidos do treino, para o modelo não aprender "não fez a prova".
- **Rede Privada.** Só 24 alunos, sem informação suficiente. Removida.
- **Colunas descartadas.** `caderno` (taxas entre 58,7% e 59,9%, sem sinal), `serie` (constante), `nome_regiao` (redundante com o estado), `school_size` mostrou importância quase zero mas ficou no pipeline do experimento A por ter sido definida desde o início.
- **Taxa do município.** Só entra a do ano anterior. A taxa do mesmo ano é calculada com os próprios alunos do treino e vazaria o alvo.
- **Divisão por município.** Uma divisão aleatória por aluno colocaria alunos da mesma cidade (com as mesmas variáveis territoriais) no treino e no teste, e infla as métricas.
- **Modelo B.** Usa só 2024, então a comparação com o A é aproximada.
- **Corte de decisão.** Escolhido nos dados de treino (previsões fora da amostra) maximizando a acurácia balanceada, entre 0,40 e 0,75. O teste só confirmou.
- **Modelo escolhido.** Regressão logística, por ter a mesma validação do boosting, ser mais simples e quase não decorar o treino. O ajuste de parâmetros mudou o AUC de teste do boosting em apenas 0,002.
- **Risco por município.** Primeira versão projetava a variação por seis anos sem filtro e gerou distâncias da meta acima de 500 pontos, impossíveis. Corrigimos exigindo 50 alunos por município e ano, limitando a projeção entre 0 e 100 e usando a distância atual até a meta como ranking principal.

## Limitação conhecida de metodologia

`school_size` e `municipality_size` são calculados com todos os alunos antes da divisão entre treino e teste, então as contagens incluem alunos do conjunto de teste. Como são só contagens, sem usar o alvo, o efeito é pequeno, mas num cenário de produção elas deveriam ser calculadas só com os dados disponíveis na hora da predição.

## Problemas encontrados e como foram resolvidos

- **Conta Azure desativada.** O workspace da Fase 2 deixou de funcionar. Migramos para o Databricks Free Edition (catálogo `workspace`) e reconstruímos bronze, silver e gold rodando de novo os notebooks da Fase 2.
- **Credencial do BigQuery.** O secret scope precisou ser recriado no novo workspace. Criamos pelo SDK do Databricks dentro de um notebook, lendo o arquivo da chave de um volume temporário, que depois foi apagado.
- **Imports do pacote `src`.** O Python não achava `src.preprocessing`. Corrigimos descobrindo a raiz do repositório pelo caminho do notebook, colocando-a no começo do `sys.path` e limpando módulos `src` em cache.
- **Variáveis perdidas ao reiniciar o notebook.** O notebook foi organizado em células de cima para baixo, com configurações no topo, para rodar com Run all.
- **Memória.** Treinar com 3,35 milhões de alunos não cabe no ambiente gratuito. Usamos uma amostra de 15%.

## Avisos que aparecem na execução e podem ser ignorados

- `BigQuery Storage module not found`: a leitura usa o endpoint REST, mais lento, mas correto
- SHAP: aviso de que o fundo de 2.000 linhas foi reduzido para 100, sem efeito nos resultados

## Referências

- Base dos Dados (indicador criança alfabetizada, diretório de municípios e população do IBGE)
- INEP, Pesquisa Alfabetiza Brasil (2023)
- Compromisso Nacional Criança Alfabetizada, MEC
