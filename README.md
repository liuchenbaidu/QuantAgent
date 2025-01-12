# QuantAgent
quant agent 

## 配置
1. 复制 `config.yaml.example` 到 `config.yaml`
2. 在 `config.yaml` 中填入您的 DeepSeek API key

## 功能列表

### 1. qlib 和agent 技术结合，进行数据获取下载
- 支持获取以下ETF数据：
  - 中证红利ETF (515180.SH)
  - 纳斯达克ETF (513100.SH)
  - 黄金ETF (518880.SH)
  - 城投债ETF (511800.SH)
- 自动存入qlib数据库
- 支持获取近1年数据

### 2. 根据提示统计分析，回测结果和图示 

### 3. 根据提示进行策略优化，回测结果和图示 


# 开发的经验：
2025.1.12 获取数据不用用agent，直接用qlib获取，然后存入数据库. qlib获取数据还没有尝试

