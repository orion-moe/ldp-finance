# LDP Finance

Pipeline de dados Bitcoin para ML baseado em "Advances in Financial Machine Learning" (Lopez de Prado).

## Instalacao

```bash
pip install -r requirements.txt
```

## Uso

```bash
python main.py  # Menu interativo
```

Comandos diretos: `download`, `optimize`, `validate`, `features` (use `--help` para opcoes).

## Estrutura

```
src/
├── data_pipeline/
│   ├── downloaders/    # Download de dados Binance
│   ├── converters/     # ZIP -> Parquet
│   ├── processors/     # Otimizacao/merge Parquet
│   └── validators/     # Validacao de dados
├── features/
│   └── bars/           # Dollar bars e imbalance bars
└── ml_pipeline/
    ├── core/           # Config, data loading, visualizations
    ├── steps/          # Pipeline steps (frac diff, AR, CUSUM, triple barrier)
    ├── feature_engineering/  # Features de microestrutura
    └── models/         # Random Forest, AR

data/
└── {symbol}-{type}/    # Ex: btcusdt-spot/
    ├── raw-zip-*/      # ZIPs baixados
    ├── raw-parquet-*/  # Parquet diario
    └── raw-parquet-merged-*/  # Parquet otimizado
```

## Funcoes Principais

| Modulo | Classe/Funcao | Descricao |
|--------|---------------|-----------|
| `downloaders/binance_downloader.py` | `BinanceDataDownloader` | Download paralelo de tick data |
| `converters/zip_to_parquet_streamer.py` | `ZipToParquetStreamer` | Conversao streaming ZIP->Parquet |
| `processors/parquet_optimizer.py` | `EnhancedParquetOptimizer` | Merge e compressao de arquivos |
| `bars/standard_dollar_bars.py` | `process_files_and_generate_bars` | Geracao de dollar bars (Numba) |
| `bars/imbalance_bars.py` | `ImbalanceBarsGenerator` | Bars baseados em order flow |
| `search_rf_classifier.py` | `main` | Pipeline ML completo |

## Referencia

Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
