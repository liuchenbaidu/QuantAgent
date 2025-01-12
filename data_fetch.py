from datetime import datetime, timedelta
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import akshare as ak

def setup_logging():
    log_dir = Path("logs")
    if not log_dir.exists():
        log_dir.mkdir()
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"etf_data_{datetime.now().strftime('%Y%m%d')}.log"),
            logging.StreamHandler()
        ]
    )

class TdxDataReader:
    """通达信数据读取器"""
    
    def __init__(self):
        self.data_dir = Path("data")
        if not self.data_dir.exists():
            self.data_dir.mkdir()
            
        # ETF代码映射
        self.etf_codes = {
            "515180": {"name": "中证红利ETF", "type": "stock"},
            "513100": {"name": "纳斯达克ETF", "type": "stock"},
            "518880": {"name": "黄金ETF", "type": "commodity"},
            "511800": {"name": "城投债ETF", "type": "bond", "yahoo_symbol": "511800.SS"}  # 添加雅虎财经代码
        }
    
    def _download_yahoo_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从雅虎财经下载数据"""
        try:
            # 创建雅虎财经的Ticker对象
            ticker = yf.Ticker(symbol)
            
            # 下载历史数据
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval="1d"
            )
            
            # 重置索引，将日期变为列，并处理时区
            df = df.reset_index()
            if df['Date'].dt.tz is not None:
                df['Date'] = df['Date'].dt.tz_localize(None)
            
            # 重命名列
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            return df
            
        except Exception as e:
            logging.error(f"从雅虎财经下载数据失败: {str(e)}")
            return pd.DataFrame()
    
    def _download_tdx_file(self, code: str, etf_type: str) -> bool:
        """下载数据文件"""
        try:
            if etf_type == "bond":
                # 使用雅虎财经获取城投债ETF数据
                yahoo_symbol = self.etf_codes[code]["yahoo_symbol"]
                start_date = "2018-01-01"
                end_date = datetime.now().strftime("%Y-%m-%d")
                
                df = self._download_yahoo_data(yahoo_symbol, start_date, end_date)
                
                if not df.empty:
                    file_path = self.data_dir / f"{code}.csv"
                    df.to_csv(file_path, index=False)
                    return True
            else:
                # 其他ETF使用原有的akshare接口
                df = ak.fund_etf_hist_em(
                    symbol=code,
                    period="daily",
                    start_date="20180101",
                    end_date=datetime.now().strftime("%Y%m%d"),
                    adjust="hfq"
                )
                if df is not None and not df.empty:
                    file_path = self.data_dir / f"{code}.csv"
                    df.to_csv(file_path, index=False)
                    return True
            return False
            
        except Exception as e:
            logging.error(f"下载数据文件失败: {str(e)}")
            return False
    
    def _process_bond_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理债券ETF数据"""
        try:
            # 打印原始列名，帮助调试
            logging.info(f"原始数据列名: {df.columns.tolist()}")
            
            # 重命名列（根据实际的列名进行调整）
            column_mapping = {
                '日期': 'date',
                'date': 'date',  # 如果已经是英文列名
                '开盘': 'open',
                'open': 'open',
                '最高': 'high',
                'high': 'high',
                '最低': 'low',
                'low': 'low',
                '收盘': 'close',
                'close': 'close',
                '成交量': 'volume',
                'volume': 'volume',
                '成交额': 'amount',
                'amount': 'amount'
            }
            
            # 只重命名存在的列
            rename_dict = {col: column_mapping[col] for col in df.columns if col in column_mapping}
            df = df.rename(columns=rename_dict)
            
            # 确保必要的列存在
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logging.error(f"缺少必要的列: {missing_columns}")
                logging.error(f"现有的列: {df.columns.tolist()}")
                raise ValueError(f"数据格式不正确，缺少列: {missing_columns}")
            
            # 确保数据类型正确，并移除时区信息
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 按日期排序
            df = df.sort_values('date')
            
            return df
            
        except Exception as e:
            logging.error(f"处理债券数据失败: {str(e)}")
            logging.error(f"数据列名: {df.columns.tolist()}")
            raise
    
    def _calculate_adjusted_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算等比后复权价格"""
        try:
            # 首先确保数据格式正确
            # 重命名列（如果需要）
            column_mapping = {
                '日期': 'date',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume'
            }
            
            # 检查是否需要重命名列
            if '日期' in df.columns:
                df = df.rename(columns=column_mapping)
            
            # 确保日期列是datetime类型，并处理时区
            if df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)
            else:
                df['date'] = pd.to_datetime(df['date'])
            
            # 确保价格列是数值类型
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 从最新日期开始往前计算复权因子
            df = df.sort_values('date', ascending=False).copy()  # 添加.copy()避免SettingWithCopyWarning
            
            # 初始化复权因子
            df['factor'] = 1.0
            
            # 计算除权因子
            for i in range(1, len(df)):
                if df.iloc[i]['close'] != 0:  # 避免除以零
                    ratio = df.iloc[i-1]['close'] / df.iloc[i]['close']
                    if abs(ratio - 1) > 0.1:  # 价格变动超过10%视为除权
                        df.iloc[i:]['factor'] *= ratio
            
            # 计算复权价格
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col] * df['factor']
            
            # 删除复权因子列
            df = df.drop('factor', axis=1)
            
            # 恢复按日期升序排序
            df = df.sort_values('date')
            
            return df
            
        except Exception as e:
            logging.error(f"计算复权价格失败: {str(e)}")
            logging.error(f"数据列名: {df.columns.tolist()}")
            logging.error(f"数据样例:\n{df.head()}")
            raise
    
    def fetch_adjusted_data(self):
        """获取所有ETF的等比后复权数据"""
        try:
            for code, info in self.etf_codes.items():
                name = info['name']
                etf_type = info['type']
                
                logging.info(f"\n开始获取 {name} ({code}) 的数据...")
                
                try:
                    # 下载数据
                    if not self._download_tdx_file(code, etf_type):
                        continue
                    
                    # 读取数据
                    df = pd.read_csv(self.data_dir / f"{code}.csv")
                    if df.empty:
                        logging.warning(f"{name} 数据为空")
                        continue
                    
                    # 打印原始数据信息
                    logging.info(f"原始数据列名: {df.columns.tolist()}")
                    logging.info(f"数据样例:\n{df.head()}")
                    
                    # 数据基本验证
                    required_columns = ['date'] if etf_type == 'bond' else ['日期']
                    required_columns.extend(['open', 'high', 'low', 'close', 'volume'] if etf_type == 'bond' else 
                                         ['开盘', '最高', '最低', '收盘', '成交量'])
                    
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        logging.error(f"{name} 数据缺少必要的列: {missing_columns}")
                        logging.error(f"现有的列: {df.columns.tolist()}")
                        continue
                    
                    # 根据ETF类型进行不同的处理
                    if etf_type == "bond":
                        df = self._process_bond_data(df)
                    else:
                        df = self._calculate_adjusted_price(df)
                    
                    # 保存处理后的数据
                    file_path = self.data_dir / f"{code}.csv"
                    df.to_csv(file_path, index=False)
                    
                    # 输出数据信息
                    file_size = file_path.stat().st_size
                    date_range = (df['date'].max() - df['date'].min()).days
                    logging.info(f"数据保存成功:")
                    logging.info(f"- 文件大小: {file_size/1024/1024:.2f} MB")
                    logging.info(f"- 数据条数: {len(df)}")
                    logging.info(f"- 时间跨度: {date_range} 天")
                    logging.info(f"- 价格范围: {df['close'].min():.4f} - {df['close'].max():.4f}")
                    
                except Exception as e:
                    logging.error(f"处理 {name} 数据时出错: {str(e)}")
                    logging.error(f"详细错误信息:", exc_info=True)
                    continue
            
            logging.info("\n所有ETF数据获取完成！")
            
        except Exception as e:
            logging.error(f"获取数据时出错: {str(e)}", exc_info=True)
            raise

def main():
    setup_logging()
    logging.info("开始执行ETF等比后复权数据获取任务...")
    
    try:
        reader = TdxDataReader()
        reader.fetch_adjusted_data()
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        return 1
    
    logging.info("程序执行完成！")
    return 0

if __name__ == "__main__":
    exit(main()) 