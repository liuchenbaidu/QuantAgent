from data_analysis import ETFAnalyzer
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

class PortfolioOptimizer:
    def __init__(self):
        self.analyzer = ETFAnalyzer()
        self.analysis_dir = Path("analysis_results")
        
        # 设置matplotlib中文字体（复用ETFAnalyzer的方法）
        self.analyzer._setup_chinese_font()
        
    def setup_logging(self):
        log_dir = Path("logs")
        if not log_dir.exists():
            log_dir.mkdir()
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "portfolio_analysis.log"),
                logging.StreamHandler()
            ]
        )
    
    def optimize_portfolio(self, returns_data: dict, risk_free_rate=0.03):
        """优化投资组合权重"""
        # 合并所有ETF的日收益率
        returns_df = pd.DataFrame({
            code: data['daily_return'] for code, data in returns_data.items()
        })
        
        # 计算年化收益和协方差
        mean_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        
        # 定义目标函数（最大化夏普比率）
        def objective(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
            return -sharpe_ratio
        
        # 约束条件
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # 权重和为1
        )
        bounds = tuple((0, 1) for _ in range(len(returns_df.columns)))
        
        # 初始权重
        init_weights = np.array([1/len(returns_df.columns)] * len(returns_df.columns))
        
        # 优化
        result = minimize(
            objective, 
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def analyze_efficient_frontier(self, returns_data: dict, num_portfolios=1000):
        """分析有效前沿"""
        returns_df = pd.DataFrame({
            code: data['daily_return'] for code, data in returns_data.items()
        })
        
        mean_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        
        # 生成随机投资组合
        results = []
        for _ in range(num_portfolios):
            weights = np.random.random(len(returns_df.columns))
            weights = weights / np.sum(weights)
            
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            results.append({
                'Return': portfolio_return,
                'Risk': portfolio_std,
                'Weights': weights
            })
        
        # 绘制有效前沿
        results_df = pd.DataFrame(results)
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['Risk'], results_df['Return'], 
                   c=results_df['Return']/results_df['Risk'], 
                   marker='o', cmap='viridis')
        plt.colorbar(label='夏普比率')
        plt.xlabel('风险（年化标准差）', fontsize=10)
        plt.ylabel('收益（年化收益率）', fontsize=10)
        plt.title('投资组合有效前沿', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.analysis_dir / 'efficient_frontier.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return results_df
    
    def run_analysis(self):
        """运行完整的投资组合分析"""
        try:
            self.setup_logging()
            logging.info("开始投资组合分析...")
            
            # 运行基础分析
            logging.info("\n=== 第一部分：基础分析 ===")
            base_results = self.analyzer.run_analysis()
            
            # 加载数据
            logging.info("\n=== 第二部分：投资组合优化 ===")
            data = self.load_data()
            returns = self.calculate_returns(data)
            
            # 优化投资组合
            logging.info("\n开始优化投资组合权重...")
            optimal_weights = self.optimize_portfolio(returns)
            
            # 输出最优权重
            logging.info("\n最优投资组合权重：")
            for code, weight in zip(self.analyzer.etf_names.keys(), optimal_weights):
                logging.info(f"{self.analyzer.etf_names[code]}: {weight:.2%}")
            
            # 分析有效前沿
            logging.info("\n开始分析投资组合有效前沿...")
            frontier_results = self.analyze_efficient_frontier(returns)
            
            # 输出有效前沿分析结果
            logging.info("\n有效前沿分析结果：")
            logging.info(f"最高夏普比率: {(frontier_results['Return']/frontier_results['Risk']).max():.2f}")
            logging.info(f"最高收益率: {frontier_results['Return'].max():.2%}")
            logging.info(f"最低风险: {frontier_results['Risk'].min():.2%}")
            
            # 生成优化结果报告
            self._generate_optimization_report(optimal_weights, frontier_results)
            
            logging.info("\n分析完成！所有结果已保存到 analysis_results 目录")
            
        except Exception as e:
            logging.error(f"分析过程中出错: {str(e)}")
            raise
    
    def _generate_optimization_report(self, optimal_weights, frontier_results):
        """生成优化结果报告"""
        with open(self.analysis_dir / 'portfolio_optimization_report.txt', 'w', encoding='utf-8') as f:
            f.write("投资组合优化报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("1. 最优投资组合权重\n")
            f.write("-" * 30 + "\n")
            for code, weight in zip(self.analyzer.etf_names.keys(), optimal_weights):
                f.write(f"{self.analyzer.etf_names[code]}: {weight:.2%}\n")
            f.write("\n")
            
            f.write("2. 有效前沿分析\n")
            f.write("-" * 30 + "\n")
            f.write(f"分析的投资组合数量: {len(frontier_results)}\n")
            f.write(f"最高夏普比率: {(frontier_results['Return']/frontier_results['Risk']).max():.2f}\n")
            f.write(f"最高收益率: {frontier_results['Return'].max():.2%}\n")
            f.write(f"最低风险: {frontier_results['Risk'].min():.2%}\n")
            f.write("\n有效前沿图已保存到: analysis_results/efficient_frontier.png\n")

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """加载所有ETF数据"""
        data = {}
        for code in self.analyzer.etf_names.keys():
            file_path = self.analyzer.data_dir / f"{code}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)  # 确保移除时区信息
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
            df.index = pd.to_datetime(df.index).tz_localize(None)
            
            # 计算日收益率
            returns[code] = pd.DataFrame({
                'daily_return': df['close'].pct_change(),
                'log_return': np.log(df['close']/df['close'].shift(1))
            })
        return returns

    def calculate_correlation(self, returns: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """计算相关性矩阵"""
        # 合并所有ETF的日收益率
        daily_returns = pd.DataFrame({
            self.analyzer.etf_names[code]: df['daily_return'] 
            for code, df in returns.items()
        })
        
        # 确保所有日期列都是 tz-naive 的
        daily_returns.index = pd.to_datetime(daily_returns.index).tz_localize(None)
        
        # 计算相关性矩阵
        corr_matrix = daily_returns.corr()
        
        # 绘制相关性热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                   annot_kws={'size': 10})
        plt.title('ETF相关性矩阵', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.analyzer.analysis_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return corr_matrix

def main():
    optimizer = PortfolioOptimizer()
    optimizer.run_analysis()

if __name__ == "__main__":
    main() 