import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from datetime import datetime

def setup_chinese_font():
    """设置中文字体"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

def plot_etf_data(code: str = "511800", name: str = "城投债ETF"):
    """绘制ETF数据图表"""
    # 设置中文字体
    setup_chinese_font()
    
    # 读取数据
    df = pd.read_csv(f"data/{code}.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    # 创建子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), height_ratios=[2, 1, 1])
    fig.suptitle(f"{name} 数据分析图", fontsize=16, y=0.95)
    
    # 1. 价格走势图
    ax1.plot(df['date'], df['close'], label='收盘价', color='blue')
    ax1.fill_between(df['date'], df['low'], df['high'], alpha=0.2, color='blue', label='日内波动')
    ax1.set_title('价格走势', fontsize=12)
    ax1.set_xlabel('日期')
    ax1.set_ylabel('价格')
    ax1.grid(True)
    ax1.legend()
    
    # 2. 成交量图
    ax2.bar(df['date'], df['volume'], color='green', alpha=0.6, label='成交量')
    ax2.set_title('成交量', fontsize=12)
    ax2.set_xlabel('日期')
    ax2.set_ylabel('成交量')
    ax2.grid(True)
    ax2.legend()
    
    # 3. 日收益率分布图
    returns = df['close'].pct_change().dropna()
    sns.histplot(returns, bins=50, ax=ax3, color='purple', alpha=0.6)
    ax3.set_title('日收益率分布', fontsize=12)
    ax3.set_xlabel('日收益率')
    ax3.set_ylabel('频次')
    ax3.grid(True)
    
    # 添加统计信息
    stats_text = (
        f"统计信息:\n"
        f"数据时间范围: {df['date'].min().strftime('%Y-%m-%d')} 到 {df['date'].max().strftime('%Y-%m-%d')}\n"
        f"平均价格: {df['close'].mean():.2f}\n"
        f"最高价: {df['high'].max():.2f}\n"
        f"最低价: {df['low'].min():.2f}\n"
        f"日均成交量: {df['volume'].mean():.0f}\n"
        f"日收益率标准差: {returns.std()*100:.2f}%"
    )
    fig.text(0.02, 0.02, stats_text, fontsize=10, va='bottom')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    save_dir = Path("analysis_results")
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    plt.savefig(save_dir / f"{code}_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"分析图已保存到: analysis_results/{code}_analysis.png")
    
    # 输出基本统计信息
    print("\n基本统计信息:")
    print(f"数据点数: {len(df)}")
    print(f"时间范围: {df['date'].min().strftime('%Y-%m-%d')} 到 {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"平均价格: {df['close'].mean():.2f}")
    print(f"价格范围: {df['low'].min():.2f} - {df['high'].max():.2f}")
    print(f"日收益率标准差: {returns.std()*100:.2f}%")

if __name__ == "__main__":
    plot_etf_data() 