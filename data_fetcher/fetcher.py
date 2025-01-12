import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import os
import time
from pathlib import Path
import yaml

def load_config():
    """加载配置文件"""
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("找不到配置文件 config.yaml")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class ETFDataFetcher:
    def __init__(self):
        # 加载配置
        config = load_config()
        self.etf_codes = config['etf_codes']
        
        # 创建数据存储目录
        self.data_dir = "data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
    def fetch_single_etf(self, name: str, code: str, start_date: str, end_date: str) -> str:
        """获取单个ETF的数据"""
        try:
            print(f"正在获取 {name} ({code}) 的数据...")
            
            df = ak.fund_etf_hist_em(
                symbol=code.split('.')[0],
                period="daily",
                start_date=start_date,
                end_date=end_date
            )
            
            if df is None or df.empty:
                return f"未能获取到 {code} 的数据"
                
            # 数据处理
            df = self._process_data(df)
            
            # 保存数据
            file_path = os.path.join(self.data_dir, f"{code.split('.')[0]}.csv")
            df.to_csv(file_path, index=False)
            
            return f"成功获取并保存 {name} 的数据，共 {len(df)} 条记录"
            
        except Exception as e:
            return f"获取 {name} 数据时出错: {str(e)}"
            
    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理数据的内部方法"""
        column_mapping = {
            '日期': 'date',
            '开盘': 'open',
            '最高': 'high',
            '最低': 'low',
            '收盘': 'close',
            '成交量': 'volume'
        }
        df = df.rename(columns=column_mapping)
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')
    
    def fetch_all_data(self, start_date: str = None, end_date: str = None):
        """获取所有ETF数据"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
            
        for name, code in self.etf_codes.items():
            result = self.fetch_single_etf(name, code, start_date, end_date)
            print(result)
            time.sleep(1)  # 避免请求过快 