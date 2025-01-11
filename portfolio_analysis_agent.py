import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Union, ClassVar
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.chains import LLMChain
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.llms.base import LLM
import yaml
import requests
import seaborn as sns
from datetime import datetime
import os
from data_fetch_agent import DeepSeekLLM
from scipy.optimize import minimize

class PortfolioAnalysisTool:
    def __init__(self):
        self.data_dir = Path("data")
        self.result_dir = Path("analysis_results")
        if not self.result_dir.exists():
            self.result_dir.mkdir()
        
        # 移除 qlib 初始化代码
        # provider_uri = "~/.qlib/qlib_data/cn_data"
        # qlib.init(provider_uri=provider_uri, region="CN")
            
    def calculate_portfolio_returns(self, weights_str: str) -> str:
        """计算投资组合收益"""
        try:
            # 解析权重字符串，格式如: "515180:0.3,513100:0.2,518880:0.3,511800:0.2"
            weights = dict(w.split(':') for w in weights_str.split(','))
            weights = {k.strip(): float(v) for k, v in weights.items()}
            
            # ETF代码映射
            code_mapping = {
                "中证红利": "515180",
                "纳斯达克": "513100",
                "黄金": "518880",
                "城投债": "511800"
            }
            
            # 转换中文名称为代码
            weights = {code_mapping.get(k, k): v for k, v in weights.items()}
            
            # 读取数据
            dfs = {}
            for code in weights.keys():
                df = pd.read_csv(self.data_dir / f"{code}.csv")
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                dfs[code] = df
            
            # 计算每个ETF的日收益率
            returns = {}
            for code, df in dfs.items():
                returns[code] = df['close'].pct_change()
            
            # 合并所有收益率
            returns_df = pd.DataFrame(returns)
            
            # 计算投资组合收益率
            portfolio_returns = sum(returns_df[code] * weight for code, weight in weights.items())
            
            # 计算累计收益率
            cumulative_returns = (1 + portfolio_returns).cumprod()
            
            # 计算年化收益率
            days = (returns_df.index[-1] - returns_df.index[0]).days
            annual_return = (cumulative_returns.iloc[-1] ** (365/days) - 1) * 100
            
            # 计算最大回撤
            cummax = cumulative_returns.cummax()
            drawdown = (cumulative_returns - cummax) / cummax
            max_drawdown = drawdown.min() * 100
            
            # 计算夏普比率
            risk_free_rate = 0.03  # 假设无风险利率为3%
            excess_returns = portfolio_returns - risk_free_rate/365
            sharpe_ratio = np.sqrt(365) * excess_returns.mean() / portfolio_returns.std()
            
            # 绘制收益率曲线
            plt.figure(figsize=(12, 6))
            cumulative_returns.plot()
            plt.title('投资组合累计收益率')
            plt.xlabel('日期')
            plt.ylabel('累计收益率')
            plt.grid(True)
            plt.savefig(self.result_dir / 'portfolio_returns.png')
            plt.close()
            
            return f"""投资组合分析结果：
年化收益率: {annual_return:.2f}%
最大回撤: {max_drawdown:.2f}%
夏普比率: {sharpe_ratio:.2f}
收益率曲线已保存到 analysis_results/portfolio_returns.png"""
            
        except Exception as e:
            return f"分析过程中出错: {str(e)}"
    
    def analyze_correlation(self, *args) -> str:
        """分析ETF之间的相关性"""
        try:
            # 读取所有ETF数据
            returns = {}
            etf_names = {
                "515180": "中证红利",
                "513100": "纳斯达克",
                "518880": "黄金",
                "511800": "城投债"
            }
            
            for file in self.data_dir.glob("*.csv"):
                code = file.stem
                df = pd.read_csv(file)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                returns[etf_names.get(code, code)] = df['close'].pct_change()
            
            # 计算相关性矩阵
            corr_matrix = pd.DataFrame(returns).corr()
            
            # 绘制相关性热力图
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title('ETF相关性矩阵')
            plt.tight_layout()
            plt.savefig(self.result_dir / 'correlation_matrix.png')
            plt.close()
            
            # 生成相关性描述
            description = "相关性分析结果：\n"
            for i in corr_matrix.index:
                for j in corr_matrix.columns:
                    if i < j:  # 只输出上三角矩阵的值
                        description += f"{i} 和 {j} 的相关系数: {corr_matrix.loc[i,j]:.2f}\n"
            
            description += "\n相关性热力图已保存到 analysis_results/correlation_matrix.png"
            return description
            
        except Exception as e:
            return f"相关性分析过程中出错: {str(e)}"
    
    def optimize_weights(self, *args) -> str:
        """使用机器学习方法优化投资组合权重"""
        try:
            # 读取数据
            etf_codes = ["515180", "513100", "518880", "511800"]
            returns_data = {}
            
            for code in etf_codes:
                df = pd.read_csv(self.data_dir / f"{code}.csv")
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                returns_data[code] = df['close'].pct_change().dropna()
            
            returns_df = pd.DataFrame(returns_data)
            
            # 定义目标函数（最大化夏普比率）
            def objective(weights):
                weights = np.array(weights)
                portfolio_return = np.sum(returns_df.mean() * weights) * 252
                portfolio_volatility = np.sqrt(
                    np.dot(weights.T, np.dot(returns_df.cov() * 252, weights))
                )
                sharpe_ratio = portfolio_return / portfolio_volatility
                return -sharpe_ratio  # 最小化负夏普比率 = 最大化夏普比率
            
            # 定义约束条件
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
            )
            bounds = tuple((0, 1) for _ in range(len(etf_codes)))  # 权重在0和1之间
            
            # 使用scipy优化
            initial_weights = [1./len(etf_codes)] * len(etf_codes)
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            # 获取最优权重
            optimal_weights = result.x
            
            # 计算最优组合的表现
            portfolio_returns = np.sum(returns_df * optimal_weights, axis=1)
            annual_return = portfolio_returns.mean() * 252 * 100
            annual_volatility = portfolio_returns.std() * np.sqrt(252) * 100
            sharpe_ratio = -result.fun  # 取负号转回来
            
            # 计算最大回撤
            cum_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cum_returns.cummax()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min() * 100
            
            # 绘制最优组合的收益曲线
            plt.figure(figsize=(12, 6))
            cum_returns.plot()
            plt.title('最优投资组合累计收益率')
            plt.xlabel('日期')
            plt.ylabel('累计收益率')
            plt.grid(True)
            plt.savefig(self.result_dir / 'optimal_portfolio_returns.png')
            plt.close()
            
            # 生成结果报告
            etf_names = {
                "515180": "中证红利",
                "513100": "纳斯达克",
                "518880": "黄金",
                "511800": "城投债"
            }
            
            report = "机器学习优化结果：\n\n"
            report += "最优权重配置：\n"
            for code, weight in zip(etf_codes, optimal_weights):
                report += f"{etf_names[code]}: {weight:.2%}\n"
            
            report += f"\n投资组合表现：\n"
            report += f"年化收益率: {annual_return:.2f}%\n"
            report += f"年化波动率: {annual_volatility:.2f}%\n"
            report += f"夏普比率: {sharpe_ratio:.2f}\n"
            report += f"最大回撤: {max_drawdown:.2f}%\n"
            report += f"\n收益率曲线已保存到 analysis_results/optimal_portfolio_returns.png"
            
            return report
            
        except Exception as e:
            return f"优化过程中出错: {str(e)}"

class PortfolioAnalysisPromptTemplate(StringPromptTemplate):
    template: ClassVar[str] = """你是一个专业的投资组合分析助手。

可用工具:
{tools}

你的任务: {input}

请按照以下格式回应:
思考: 你应该如何完成这个任务
行动: 要采取的行动名称
行动输入: 行动的输入参数
观察: 行动的结果

现在开始你的任务。
"""
    
    input_variables: List[str] = ["input", "tools"]
    
    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

class PortfolioAnalysisAgent:
    def __init__(self):
        config = self._load_config()
        self.llm = DeepSeekLLM(
            api_key=config['api_keys']['deepseek'],
            model_name="deepseek-chat",
            temperature=0
        )
        self.tools = self._setup_tools()
        self.output_parser = PortfolioOutputParser()
        self.agent_chain = self._setup_agent()
    
    def _load_config(self):
        with open("config.yaml", 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_tools(self) -> List[Tool]:
        analysis_tool = PortfolioAnalysisTool()
        return [
            Tool(
                name="calculate_portfolio_returns",
                func=analysis_tool.calculate_portfolio_returns,
                description="计算投资组合收益，输入格式：'ETF代码:权重,ETF代码:权重,...'，权重之和应为1"
            ),
            Tool(
                name="analyze_correlation",
                func=analysis_tool.analyze_correlation,
                description="分析ETF之间的相关性并生成热力图"
            ),
            Tool(
                name="optimize_weights",
                func=analysis_tool.optimize_weights,
                description="使用机器学习方法优化投资组合权重，自动寻找最优配置"
            )
        ]
    
    def _setup_agent(self) -> AgentExecutor:
        prompt = PortfolioAnalysisPromptTemplate(
            template=PortfolioAnalysisPromptTemplate.template,
            input_variables=["input", "tools"]
        )
        
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=self.output_parser,
            stop=["\n观察:"],
            allowed_tools=[tool.name for tool in self.tools]
        )
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True
        )
    
    def run(self, task: str) -> str:
        inputs = {
            "input": task,
            "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        }
        return self.agent_chain.run(inputs)

class PortfolioOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "行动:" not in text:
            return AgentFinish({"output": text}, text)
        
        action_match = text.split("行动:")[1].split("\n")[0].strip()
        action_input_match = text.split("行动输入:")[1].split("\n")[0].strip()
        
        return AgentAction(action_match, action_input_match, text)

def main():
    try:
        import seaborn
    except ImportError:
        print("正在安装必要的依赖...")
        os.system("pip install seaborn")
    
    agent = PortfolioAnalysisAgent()
    
    # 分析任务
    task = """
    请执行以下分析：
    1. 首先分析四个ETF之间的相关性
    2. 然后依次分析以下权重组合的年化收益和风险：
       a. 均衡配置：中证红利:0.25,纳斯达克:0.25,黄金:0.25,城投债:0.25
       b. 偏重中证红利：中证红利:0.4,纳斯达克:0.2,黄金:0.2,城投债:0.2
       c. 偏重黄金：中证红利:0.2,纳斯达克:0.2,黄金:0.4,城投债:0.2
    3. 使用机器学习方法寻找最优权重配置，并与上述手动配置进行比较
    
    请分析各个组合的表现并给出投资建议。
    """
    
    try:
        result = agent.run(task)
        print(result)
    except Exception as e:
        print(f"运行过程中出错: {str(e)}")
        print("详细错误信息:", e.__class__.__name__)

if __name__ == "__main__":
    main() 