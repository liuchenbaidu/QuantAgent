import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到 Python 路径
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from data_fetcher import ETFDataFetcher

def test_fetch_data():
    """测试数据获取功能"""
    try:
        # 初始化数据获取器
        fetcher = ETFDataFetcher()
        
        # 测试获取最近一个月的数据
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        
        print(f"测试获取数据: {start_date} 到 {end_date}")
        fetcher.fetch_all_data(start_date=start_date, end_date=end_date)
        
        # 验证数据文件是否存在
        data_dir = Path(project_root) / "data"
        expected_files = [
            "515180.csv",
            "513100.csv",
            "518880.csv",
            "511800.csv"
        ]
        
        for file_name in expected_files:
            file_path = data_dir / file_name
            if file_path.exists():
                file_size = file_path.stat().st_size
                print(f"文件 {file_name} 存在，大小: {file_size/1024:.2f} KB")
            else:
                print(f"错误: 文件 {file_name} 不存在")
        
        print("\n数据获取测试完成！")
        
    except Exception as e:
        print(f"测试过程中出错: {str(e)}")
        raise

def test_single_etf():
    """测试单个ETF数据获取"""
    try:
        fetcher = ETFDataFetcher()
        
        # 测试获取中证红利ETF的数据
        name = "中证红利ETF"
        code = "515180.SH"
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
        
        print(f"\n测试获取单个ETF: {name}")
        result = fetcher.fetch_single_etf(name, code, start_date, end_date)
        print(result)
        
    except Exception as e:
        print(f"单个ETF测试出错: {str(e)}")
        raise

if __name__ == "__main__":
    print("开始测试数据获取模块...")
    
    # 运行测试
    test_single_etf()
    test_fetch_data() 