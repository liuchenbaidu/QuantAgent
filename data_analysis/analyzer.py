import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Union
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

class ETFAnalyzer:
    """ETF数据分析器"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.analysis_dir = Path("analysis_results")
        if not self.analysis_dir.exists():
            self.analysis_dir.mkdir(parents=True)
        
        # 设置matplotlib中文字体
        self._setup_chinese_font()
        
        # ETF代码映射
        self.etf_names = {
            "515180": "中证红利",
            "513100": "纳斯达克",
            "518880": "黄金",
            "511800": "城投债"
        }
    
    def _setup_chinese_font(self):
        """设置中文字体"""
        try:
            # 优先使用微软雅黑
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            
            # 测试中文显示
            plt.figure(figsize=(1, 1))
            plt.text(0.5, 0.5, '测试')
            plt.close()
            
        except Exception as e:
            logging.warning(f"设置中文字体时出错: {str(e)}")
            # 如果没有合适的中文字体，使用默认字体
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """加载所有ETF数据"""
        data = {}
        for code in self.etf_names.keys():
            file_path = self.data_dir / f"{code}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                data[code] = df
            else:
                logging.warning(f"找不到文件: {file_path}")
        return data
    
    def calculate_returns(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """计算收益率"""
        returns = {}
        for code, df in data.items():
            # 确保日期列是 tz-naive 的
            if isinstance(df.index, pd.DatetimeIndex):
                df.index = df.index.tz_localize(None)
            else:
                df.index = pd.to_datetime(df.index).tz_localize(None)
            
            # 计算日收益率
            returns[code] = pd.DataFrame({
                'daily_return': df['close'].pct_change(),
                'log_return': np.log(df['close']/df['close'].shift(1)),
                'cumulative_return': (1 + df['close'].pct_change()).cumprod()
            }, index=df.index)  # 确保使用相同的索引
        
        return returns
    
    def analyze_volatility(self, returns: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """分析波动率"""
        volatility = {}
        for code, df in returns.items():
            # 计算年化波动率
            volatility[code] = {
                'annual_volatility': df['daily_return'].std() * np.sqrt(252),
                'max_drawdown': (df['cumulative_return'] / df['cumulative_return'].cummax() - 1).min()
            }
        return pd.DataFrame(volatility).T
    
    def plot_cumulative_returns(self, returns: Dict[str, pd.DataFrame]):
        """绘制累计收益率对比图"""
        plt.figure(figsize=(12, 6))
        for code, df in returns.items():
            plt.plot(df.index, df['cumulative_return'], label=self.etf_names[code])
        
        plt.title('ETF累计收益率对比', fontsize=12)
        plt.xlabel('日期', fontsize=10)
        plt.ylabel('累计收益率', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.tight_layout()  # 调整布局，防止标签被裁剪
        plt.savefig(self.analysis_dir / 'cumulative_returns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def calculate_correlation(self, returns: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """计算相关性矩阵"""
        try:
            # 合并所有ETF的日收益率，确保所有数据使用相同的时区处理
            daily_returns_data = {}
            common_index = None
            
            # 首先获取所有数据的索引
            indexes = [df.index for df in returns.values()]
            if indexes:
                # 获取所有索引的交集
                common_index = indexes[0]
                for idx in indexes[1:]:
                    common_index = common_index.intersection(idx)
                
                # 确保common_index是tz-naive的
                if common_index.tz is not None:
                    common_index = common_index.tz_localize(None)
            
            # 使用共同索引重新采样数据
            for code, df in returns.items():
                series = df['daily_return'].reindex(common_index)
                if series.index.tz is not None:
                    series.index = series.index.tz_localize(None)
                daily_returns_data[self.etf_names[code]] = series
            
            # 创建DataFrame
            daily_returns = pd.DataFrame(daily_returns_data)
            
            # 计算相关性矩阵
            corr_matrix = daily_returns.corr()
            
            # 绘制相关性热力图
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                       annot_kws={'size': 10})
            plt.title('ETF相关性矩阵', fontsize=12)
            plt.tight_layout()
            plt.savefig(self.analysis_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            return corr_matrix
            
        except Exception as e:
            logging.error(f"计算相关性时出错: {str(e)}")
            logging.error("数据信息:")
            for code, df in returns.items():
                logging.error(f"{code} 索引类型: {type(df.index)}")
                logging.error(f"{code} 时区信息: {df.index.tz if isinstance(df.index, pd.DatetimeIndex) else 'None'}")
            raise
    
    def run_analysis(self) -> Dict:
        """运行完整的分析流程"""
        try:
            # 加载数据
            logging.info("开始加载ETF数据...")
            data = self.load_data()
            for code, df in data.items():
                logging.info(f"已加载 {self.etf_names[code]} 数据: {len(df)} 条记录, "
                            f"时间范围: {df.index[0].strftime('%Y-%m-%d')} 到 "
                            f"{df.index[-1].strftime('%Y-%m-%d')}")
            
            # 计算收益率
            logging.info("\n开始计算收益率指标...")
            returns = self.calculate_returns(data)
            for code, df in returns.items():
                total_return = (df['cumulative_return'].iloc[-1] - 1) * 100
                annual_return = (total_return / (len(df) / 252))
                logging.info(f"{self.etf_names[code]} 总收益率: {total_return:.2f}%, "
                            f"年化收益率: {annual_return:.2f}%")
            
            # 分析波动率
            logging.info("\n开始分析波动率...")
            volatility = self.analyze_volatility(returns)
            for code in volatility.index:
                logging.info(f"{self.etf_names[code]} 年化波动率: "
                            f"{volatility.loc[code, 'annual_volatility']*100:.2f}%, "
                            f"最大回撤: {volatility.loc[code, 'max_drawdown']*100:.2f}%")
            
            # 绘制累计收益率图
            logging.info("\n生成累计收益率对比图...")
            self.plot_cumulative_returns(returns)
            
            # 计算相关性
            logging.info("\n计算ETF间相关性...")
            correlation = self.calculate_correlation(returns)
            for i in correlation.index:
                for j in correlation.columns:
                    if i < j:  # 只输出上三角矩阵
                        logging.info(f"{i} 和 {j} 的相关系数: {correlation.loc[i,j]:.2f}")
            
            # 生成分析报告
            report = {
                'volatility': volatility,
                'correlation': correlation,
                'plots': {
                    'cumulative_returns': str(self.analysis_dir / 'cumulative_returns.png'),
                    'correlation_matrix': str(self.analysis_dir / 'correlation_matrix.png')
                }
            }
            
            # 保存分析结果
            self._save_results(report)
            logging.info("\n基础分析完成，结果已保存")
            
            return report
            
        except Exception as e:
            logging.error(f"分析过程中出错: {str(e)}", exc_info=True)
            raise
    
    def _save_results(self, report: Dict):
        """保存分析结果"""
        # 保存波动率分析结果
        report['volatility'].to_csv(self.analysis_dir / 'volatility_analysis.csv')
        
        # 保存相关性矩阵
        report['correlation'].to_csv(self.analysis_dir / 'correlation_matrix.csv')
        
        # 生成简要报告
        with open(self.analysis_dir / 'analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write("ETF分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("1. 波动率分析\n")
            f.write("-" * 30 + "\n")
            f.write(report['volatility'].to_string())
            f.write("\n\n")
            
            f.write("2. 相关性分析\n")
            f.write("-" * 30 + "\n")
            f.write(report['correlation'].to_string())
            f.write("\n\n")
            
            f.write("3. 可视化结果\n")
            f.write("-" * 30 + "\n")
            f.write(f"累计收益率图: {report['plots']['cumulative_returns']}\n")
            f.write(f"相关性热力图: {report['plots']['correlation_matrix']}\n") 