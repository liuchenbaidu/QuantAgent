from data_fetch_agent import ETFDataAgent

def main():
    print("开始获取ETF数据...")
    agent = ETFDataAgent()
    agent.fetch_data()
    print("数据获取完成！")

if __name__ == "__main__":
    main() 