# Bitcoin ML Finance

Sistema de processamento e análise de dados de trading de Bitcoin com foco em machine learning e engenharia de features avançadas.

## Estrutura do Projeto

```
.
├── src/                      # Código fonte principal
│   ├── data_pipeline/        # Pipeline de dados
│   │   ├── downloaders/      # Scripts para download de dados
│   │   ├── processors/       # Processamento e otimização
│   │   └── validators/       # Validação de integridade
│   ├── features/             # Engenharia de features
│   ├── notebooks/            # Jupyter notebooks
│   └── utils/                # Utilitários
├── data/                     # Dados (não versionado)
│   ├── raw/                  # Dados brutos baixados
│   ├── processed/            # Dados processados
│   └── optimized/            # Dados otimizados
├── config/                   # Arquivos de configuração
├── logs/                     # Logs de execução
├── tests/                    # Testes
├── docs/                     # Documentação
└── article/                  # Artigo acadêmico

```

## Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/degen-ml-finance.git
cd degen-ml-finance

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instale as dependências
pip install -r requirements.txt
```

## Uso

### 1. Download de Dados

```bash
# Modo interativo (recomendado)
python main.py download

# Ou use o script diretamente
python src/data_pipeline/downloaders/binance_downloader.py
```

### 2. Otimização de Arquivos Parquet

```bash
python main.py optimize --source data/raw/dataset-raw-monthly-compressed/spot \
                       --target data/optimized/spot \
                       --max-size 10
```

### 3. Validação de Dados

```bash
# Validação rápida
python main.py validate --quick

# Validação completa com relatórios
python main.py validate --advanced --output-dir reports/
```

### 4. Geração de Features

```bash
# Gerar imbalance dollar bars
python main.py features --type imbalance
```

## Pipeline de Dados

1. **Download**: Baixa dados históricos de trading da Binance
2. **Otimização**: Combina arquivos pequenos em chunks de 10GB
3. **Validação**: Verifica integridade e qualidade dos dados
4. **Features**: Gera barras avançadas (imbalance dollar bars)

## Principais Funcionalidades

- Download automatizado de dados spot e futures da Binance
- Conversão e otimização de arquivos Parquet
- Validação completa de integridade de dados
- Geração de imbalance dollar bars
- Processamento distribuído com Dask
- Otimização de performance com Numba

## Requisitos

- Python 3.8+
- 16GB+ RAM (recomendado para processamento de grandes volumes)
- 100GB+ de espaço em disco para dados

## Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Crie um Pull Request

## Licença

[Especifique a licença aqui]