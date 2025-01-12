from .fetcher import ETFDataFetcher

def main():
    print("开始获取ETF数据...")
    fetcher = ETFDataFetcher()
    fetcher.fetch_all_data()
    print("数据获取完成！")

if __name__ == "__main__":
    main() 