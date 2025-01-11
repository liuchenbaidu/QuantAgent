import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import os
import time
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.chains import LLMChain
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union, Dict, ClassVar
from langchain.llms.base import LLM  # 导入基础LLM类
import yaml
from pathlib import Path
import requests

class DeepSeekLLM(LLM):
    """自定义 DeepSeek LLM 实现"""
    api_key: str
    model_name: str = "deepseek-chat"
    temperature: float = 0

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    def _call(self, prompt: str, **kwargs) -> str:
        """调用 DeepSeek API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            json=data,
            headers=headers
        )
        
        if response.status_code != 200:
            raise Exception(f"API调用失败: {response.text}")
            
        return response.json()["choices"][0]["message"]["content"]

def load_config():
    """加载配置文件"""
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("找不到配置文件 config.yaml")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class ETFDataTool:
    def __init__(self):
        # 创建数据存储目录
        self.data_dir = "data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
    def fetch_etf_data(self, code: str, start_date: str, end_date: str) -> str:
        """获取ETF数据的工具函数"""
        try:
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
            
            return f"成功获取并保存 {code} 的数据，共 {len(df)} 条记录"
            
        except Exception as e:
            return f"获取 {code} 数据时出错: {str(e)}"
            
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

class ETFDataPromptTemplate(StringPromptTemplate):
    template: ClassVar[str] = """你是一个专业的ETF数据分析助手。
    
    可用的ETF代码:
    - 中证红利ETF (515180.SH)
    - 纳斯达克ETF (513100.SH)
    - 黄金ETF (518880.SH)
    - 城投债ETF (511800.SH)
    
    你的任务: {input}
    
    可用工具:
    {tools}
    
    请按照以下格式回应:
    思考: 你应该如何完成这个任务
    行动: 要采取的行动名称
    行动输入: 行动的输入参数
    观察: 行动的结果
    
    现在开始你的任务。
    """
    
    # 添加 input_variables
    input_variables: List[str] = ["input", "tools"]
    
    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

class ETFDataOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """解析LLM输出"""
        if "行动:" not in text:
            return AgentFinish({"output": text}, text)
            
        action_match = text.split("行动:")[1].split("\n")[0].strip()
        action_input_match = text.split("行动输入:")[1].split("\n")[0].strip()
        
        return AgentAction(action_match, action_input_match, text)

class ETFDataAgent:
    def __init__(self):
        # 加载配置
        config = load_config()
        
        # 初始化自定义的 DeepSeek LLM
        self.llm = DeepSeekLLM(
            api_key=config['api_keys']['deepseek'],
            model_name="deepseek-chat",
            temperature=0
        )
        self.tools = self._setup_tools()
        self.output_parser = ETFDataOutputParser()  # 创建输出解析器实例
        self.agent_chain = self._setup_agent()
        
    def _setup_tools(self) -> List[Tool]:
        etf_tool = ETFDataTool()
        return [
            Tool(
                name="fetch_etf_data",
                func=etf_tool.fetch_etf_data,
                description="获取ETF数据，输入参数：ETF代码、开始日期、结束日期"
            )
        ]
        
    def _setup_agent(self) -> AgentExecutor:
        prompt = ETFDataPromptTemplate(
            template=ETFDataPromptTemplate.template,
            input_variables=["input", "tools"]
        )
        
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=self.output_parser,  # 使用输出解析器实例
            stop=["\n观察:"],
            allowed_tools=[tool.name for tool in self.tools]
        )
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True
        )
        
    def run(self, task: str) -> str:
        """运行Agent"""
        # 将任务包装在正确的输入格式中
        inputs = {
            "input": task,
            "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        }
        return self.agent_chain.run(inputs)

def main():
    # 首先检查是否安装了必要的依赖
    try:
        import langchain_community
        import yaml
        import requests  # 添加 requests 依赖检查
    except ImportError:
        print("正在安装必要的依赖...")
        os.system("pip install langchain-community pyyaml requests")
        
    agent = ETFDataAgent()
    
    # 设置时间范围
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
    
    # 执行数据获取任务
    task = f"""
    请获取以下ETF的近一年数据（从 {start_date} 到 {end_date}）：
    1. 中证红利ETF (515180.SH)
    2. 纳斯达克ETF (513100.SH)
    3. 黄金ETF (518880.SH)
    4. 城投债ETF (511800.SH)
    
    请依次获取每个ETF的数据并保存。
    """
    
    try:
        result = agent.run(task)
        print(result)
    except Exception as e:
        print(f"运行过程中出错: {str(e)}")
        print("详细错误信息:", e.__class__.__name__)

if __name__ == "__main__":
    main() 