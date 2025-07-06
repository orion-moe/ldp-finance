# Pipeline Modificado - Guia de Uso

## 🔄 Novo Fluxo do Pipeline

O pipeline foi reorganizado para incluir validação robusta de CSV e prevenção de corrupção de dados:

### Pipeline Steps:
1. **📥 Download ZIP data and extract to CSV** - Download com verificação CHECKSUM + extração
2. **🔍 Validate CSV integrity and convert to Parquet** - Validação de CSV + conversão segura
3. **🔧 Optimize Parquet files** - Otimização robusta com prevenção de corrupção
4. **✅ Validate optimized data integrity** - Validação final dos dados otimizados
5. **📊 Generate features** - Geração de features para ML
6. **🚪 Exit**

## 📋 Detalhes de Cada Etapa

### Step 1: Download e Extração
**O que faz:**
- Downloads paralelos com verificação SHA256/CHECKSUM
- Extração automática de todos os arquivos .zip para .csv
- Verificação de integridade dos CSV extraídos
- Opção de limpeza automática dos ZIPs após extração bem-sucedida

**Melhorias:**
- ✅ Verificação de checksum obrigatória
- ✅ Extração integrada no processo
- ✅ Validação básica dos CSV extraídos
- ✅ Rastreamento de progresso para downloads e extrações

### Step 2: Validação de CSV e Conversão
**O que faz:**
- Validação detalhada da integridade dos arquivos CSV
- Detecção automática de formato (com/sem header)
- Verificação de:
  - Colunas obrigatórias (time, price, qty)
  - Valores nulos em colunas críticas
  - Preços inválidos (≤0)
  - Formato de timestamp
- Conversão segura CSV → Parquet com tipos otimizados
- Verificação dos arquivos Parquet gerados

**Melhorias:**
- 🛡️ Validação abrangente antes da conversão
- 🔍 Detecção de problemas de dados
- 📊 Relatórios detalhados de qualidade
- ✅ Verificação pós-conversão

### Step 3: Otimização de Parquet
**O que faz:**
- Combina arquivos pequenos em chunks maiores (padrão 10GB)
- Mantém ordem cronológica dos dados
- Otimiza para melhor performance de I/O
- Cleanup automático em caso de erro

**Melhorias:**
- 🔄 Processamento eficiente de arquivos grandes
- 📋 Logs detalhados de cada operação
- ✅ Verificação de dados após otimização
- 💾 Redução do número de arquivos

### Step 4: Validação Final
**Opções disponíveis:**
1. Quick validation - Verificação rápida básica
2. Advanced validation - Relatórios detalhados
3. Missing dates validation - Verificação de gaps temporais
4. **🛡️ Comprehensive integrity validation** (NOVO) - Validação completa com score de qualidade

**Melhorias:**
- 📊 Score de qualidade de dados (0-100)
- 🔍 Detecção de anomalias com Numba
- 📄 Relatórios JSON detalhados
- ⚡ Processamento paralelo

### Step 5: Geração de Features
- Permanece inalterado
- Geração de imbalance dollar bars
- Processamento com Dask distribuído

## 🚀 Como Usar

### Execução Interativa
```bash
python main.py
```

### Execução em Lote
```bash
# Exemplo completo para BTCUSDT spot monthly
python main.py download --symbol BTCUSDT --type spot --granularity monthly --start 2024-01 --end 2024-03 --workers 5

# Validação e conversão de CSV (nova funcionalidade)
python src/data_pipeline/converters/csv_to_parquet.py --symbol BTCUSDT --type spot --granularity monthly --cleanup --verify

# Otimização de parquet
python src/data_pipeline/processors/parquet_optimizer.py --source datasets/dataset-raw-monthly-compressed/spot --target datasets/dataset-raw-monthly-compressed-optimized/spot --max-size 10

# Validação integral
python src/data_pipeline/validators/data_integrity_validator.py --directory datasets/dataset-raw-monthly-compressed-optimized/spot --output reports/integrity_report.json --verbose
```

## 🛡️ Benefícios do Novo Pipeline

### Prevenção de Corrupção
- **Validação em múltiplas camadas**: CSV → Parquet → Otimizado
- **Checksums e verificações**: Em cada etapa crítica
- **Fail-safe mechanisms**: Rollback automático em caso de erro
- **Arquivos temporários**: Operações seguras com staging

### Observabilidade
- **Logs detalhados**: Rastreamento completo de operações
- **Métricas de qualidade**: Score e relatórios de integridade
- **Progress tracking**: Estado persistente para retomar operações
- **Relatórios JSON**: Dados estruturados para análise

### Performance
- **Processamento Numba**: Operações críticas otimizadas
- **Paralelização**: Downloads e validações concorrentes
- **Batch processing**: Operações em lote eficientes
- **Memory efficient**: Streaming para arquivos grandes

## ⚠️ Notas Importantes

1. **Sempre execute as etapas em ordem**: O pipeline foi projetado para ser sequencial
2. **Verifique logs em caso de erro**: Logs detalhados estão em `datasets/logs/`
3. **Use modo robusto por padrão**: Especialmente para dados críticos
4. **Mantenha backups**: O sistema pode criar backups automáticos se configurado
5. **Monitore o espaço em disco**: Validações podem usar espaço temporário adicional

## 🔧 Troubleshooting

### Problema: CSV com formato inconsistente
**Solução**: Use a validação da Etapa 2 para identificar e corrigir problemas

### Problema: Arquivos Parquet corrompidos
**Solução**: Use o otimizador robusto (Etapa 3) que detecta e previne corrupção

### Problema: Dados faltando após otimização
**Solução**: Verificação automática de row count e checksums previne perda de dados

### Problema: Performance lenta
**Solução**: Ajuste o número de workers e use processamento em lote