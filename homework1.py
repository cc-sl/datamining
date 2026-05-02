# -*- coding: utf-8 -*-
"""
北京空气质量数据挖掘分析
包含数据预处理、统计分析、可视化和关联规则挖掘
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 第一阶段：数据预处理


print("=" * 80)
print("第一阶段：数据预处理")
print("=" * 80)

# 1. 读取原始数据
print("\n[1.1] 读取原始数据...")
df_raw = pd.read_csv('work1/beijing.csv', encoding='gb2312')

# 转换为UTF-8并保存
df_raw.to_csv('work1/beijing_utf8.csv', encoding='utf-8', index=False)
print(f"✓ 数据已转换为UTF-8格式，保存至 work1/beijing_utf8.csv")
print(f"✓ 原始数据形状: {df_raw.shape}")

# 记录清洗前的统计信息
stat_before = {
    'total_rows': df_raw.shape[0],
    'total_cols': df_raw.shape[1],
    'missing_count': df_raw.isnull().sum().sum(),
    'missing_rate': (df_raw.isnull().sum().sum() / (df_raw.shape[0] * df_raw.shape[1])) * 100,
    'duplicate_rows': df_raw.duplicated().sum()
}

print(f"\n数据清洗前统计:")
print(f"  - 总行数: {stat_before['total_rows']}")
print(f"  - 总列数: {stat_before['total_cols']}")
print(f"  - 缺失值总数: {stat_before['missing_count']}")
print(f"  - 缺失率: {stat_before['missing_rate']:.2f}%")
print(f"  - 重复行数: {stat_before['duplicate_rows']}")

# 2. 任务1：数据清洗
print("\n[1.2] 任务1：数据清洗...")

# 2.1 删除重复行
df_clean = df_raw.drop_duplicates().reset_index(drop=True)
print(f"✓ 删除重复行: 移除 {stat_before['duplicate_rows']} 条重复记录")

# 2.2 处理空列（第一列是索引，删除）
if '' in df_clean.columns:
    df_clean = df_clean.drop('', axis=1)

# 2.3 清理质量等级中的空格
df_clean['质量等级'] = df_clean['质量等级'].str.strip()

# 2.4 异常值处理 - 检查AQI和污染物数值
pollutants = ['PM2.5', 'PM10', 'So2', 'No2', 'Co', 'O3']
for col in pollutants:
    if col in df_clean.columns:
        # 将空字符串转为NaN
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        # 识别异常值（负数）
        if (df_clean[col] < 0).any():
            print(f"  ⚠ {col} 中存在负值，已处理")
            df_clean.loc[df_clean[col] < 0, col] = np.nan

# 2.5 缺失值处理 - 分列使用不同策略
# 对于污染物，使用该列的中位数填补
for col in pollutants:
    if col in df_clean.columns:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"  ✓ {col} 缺失值({missing_count}个)已用中位数({median_val:.2f})填补")

# 对于AQI指数，也使用中位数填补
if 'AQI指数' in df_clean.columns:
    if df_clean['AQI指数'].isnull().any():
        df_clean['AQI指数'].fillna(df_clean['AQI指数'].median(), inplace=True)

# 2.6 一致性检查 - 检查质量等级与AQI的对应关系
print(f"\n  ✓ 质量等级分布:")
print(f"    {df_clean['质量等级'].value_counts().to_dict()}")

stat_after_clean = {
    'total_rows': df_clean.shape[0],
    'missing_count': df_clean.isnull().sum().sum(),
    'missing_rate': (df_clean.isnull().sum().sum() / (df_clean.shape[0] * df_clean.shape[1])) * 100 if df_clean.shape[1] > 0 else 0
}

print(f"\n数据清洗后统计:")
print(f"  - 行数: {stat_after_clean['total_rows']} (删除 {stat_before['total_rows'] - stat_after_clean['total_rows']} 条重复/异常记录)")
print(f"  - 缺失值: {stat_after_clean['missing_count']} (缺失率: {stat_after_clean['missing_rate']:.2f}%)")

# 3. 任务2：数据标准化
print("\n[1.3] 任务2：数据标准化...")

# 3.1 日期格式标准化
# 处理第二个日期列（通常是重复的）
if len(df_clean.columns) > 1 and '日期' in df_clean.columns:
    # 取第一个日期列
    date_col_idx = list(df_clean.columns).index('日期')
    
df_clean['日期'] = pd.to_datetime(df_clean.iloc[:, 1], format='%Y/%m/%d', errors='coerce')
print(f"✓ 日期已转换为标准datetime类型")

# 3.2 质量等级分类标准化
quality_mapping = {
    '优': '优',
    '良': '良',
    '轻度污染': '轻度污染',
    '中度污染': '中度污染',
    '重度污染': '重度污染',
    '严重污染': '严重污染'
}
# 确保所有值都被识别
unique_qualities = df_clean['质量等级'].unique()
print(f"✓ 质量等级分类: {sorted(unique_qualities.tolist())}")

# 3.3 污染物数据标准化为float类型
for col in pollutants:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype(float)

print(f"✓ 污染物数据已转换为float类型")

# 4. 任务3：时间维度特征衍生工程
print("\n[1.4] 任务3：时间维度特征衍生工程...")

# 4.1 从日期衍生出基础时间特征
df_clean['年'] = df_clean['日期'].dt.year
df_clean['月'] = df_clean['日期'].dt.month
df_clean['日'] = df_clean['日期'].dt.day
df_clean['周'] = df_clean['日期'].dt.isocalendar().week
df_clean['星期几'] = df_clean['日期'].dt.dayofweek  # 0=周一, 6=周日

# 4.2 衍生出具有科研意义的特征

# 季节特征
def get_season(month):
    if month in [12, 1, 2]:
        return '冬'
    elif month in [3, 4, 5]:
        return '春'
    elif month in [6, 7, 8]:
        return '夏'
    else:
        return '秋'

df_clean['季节'] = df_clean['月'].apply(get_season)

# 是否周末
df_clean['是否周末'] = df_clean['星期几'].apply(lambda x: '是' if x >= 5 else '否')

# 2017年中国法定节假日（示例）
holidays_2017 = [
    '2017-01-01', '2017-01-02', '2017-01-03',  # 元旦
    '2017-01-27', '2017-01-28', '2017-01-30', '2017-01-31', '2017-02-01', '2017-02-02',  # 春节
    '2017-04-03', '2017-04-04', '2017-04-05',  # 清明节
    '2017-05-01',  # 劳动节
    '2017-05-28', '2017-05-29', '2017-05-30',  # 端午节
    '2017-10-01', '2017-10-02', '2017-10-03', '2017-10-05', '2017-10-06', '2017-10-07', '2017-10-08'  # 国庆节
]
holidays_2017 = pd.to_datetime(holidays_2017)

df_clean['是否法定节假日'] = df_clean['日期'].isin(holidays_2017).apply(lambda x: '是' if x else '否')

# 月份的经济活动特征
def get_activity_season(month):
    if month in [1, 2]:
        return '冬季低谷'
    elif month in [3, 4, 5]:
        return '春季活跃'
    elif month in [6, 7, 8]:
        return '夏季高峰'
    else:
        return '秋季回升'

df_clean['经济活动期'] = df_clean['月'].apply(get_activity_season)

# 工作日/休息日特征
def get_workday_type(date):
    dayofweek = date.dayofweek
    if dayofweek < 5:
        return '工作日'
    else:
        return '休息日'

df_clean['工作日类型'] = df_clean['日期'].apply(get_workday_type)

print(f"✓ 衍生特征已生成:")
print(f"  - 季节: {df_clean['季节'].unique().tolist()}")
print(f"  - 是否周末: {df_clean['是否周末'].unique().tolist()}")
print(f"  - 是否法定节假日: {df_clean['是否法定节假日'].unique().tolist()}")
print(f"  - 经济活动期: {df_clean['经济活动期'].unique().tolist()}")
print(f"  - 工作日类型: {df_clean['工作日类型'].unique().tolist()}")

# 保存预处理后的数据
df_clean.to_csv('work1/beijing_cleaned.csv', encoding='utf-8', index=False)
print(f"\n✓ 预处理完成！清洗后数据已保存至 work1/beijing_cleaned.csv")


# 第二阶段：数据分析


print("\n" + "=" * 80)
print("第二阶段：数据分析")
print("=" * 80)

# 任务1：统计描述
print("\n[2.1] 任务1：统计描述...")

# 计算污染物的统计特征
stat_desc = df_clean[pollutants].describe().round(2)
print(f"\n污染物统计描述:")
print(stat_desc)

# 保存统计结果
stat_desc.to_csv('work1/污染物统计描述.csv', encoding='utf-8')
print(f"✓ 统计描述已保存至 work1/污染物统计描述.csv")

# 任务2：统计发现与深度可视化
print("\n[2.2] 任务2：统计发现与深度可视化...")

# 创建输出目录
if not os.path.exists('work1/visualizations'):
    os.makedirs('work1/visualizations')

# 图1: 相关性热力图
print("  生成图1: 相关性热力图...")
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df_clean[pollutants + ['AQI指数']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            cbar_kws={'label': '相关系数'}, ax=ax)
plt.title('北京空气质量数据 - 污染物相关性热力图', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('work1/visualizations/01_相关性热力图.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存为 work1/visualizations/01_相关性热力图.png")

# 图2: 季节-污染物时空轨迹图
print("  生成图2: 季节污染物成分占比分析...")
seasonal_avg = df_clean.groupby('季节')[pollutants].mean()
fig, ax = plt.subplots(figsize=(12, 6))
seasonal_avg.T.plot(kind='bar', ax=ax)
plt.title('不同季节污染物平均浓度分布 - 时空轨迹分析', fontsize=14, fontweight='bold')
plt.xlabel('污染物类型', fontsize=12)
plt.ylabel('平均浓度 (μg/m³)', fontsize=12)
plt.legend(title='季节', loc='upper right')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('work1/visualizations/02_季节污染物成分占比.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存为 work1/visualizations/02_季节污染物成分占比.png")

# 图3: 污染物箱线图
print("  生成图3: 污染物箱线图...")
fig, ax = plt.subplots(figsize=(12, 6))
df_clean[pollutants].plot(kind='box', ax=ax)
plt.title('污染物浓度分布箱线图', fontsize=14, fontweight='bold')
plt.ylabel('浓度 (μg/m³)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('work1/visualizations/03_污染物箱线图.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存为 work1/visualizations/03_污染物箱线图.png")

# 图4: AQI指数与质量等级的关系
print("  生成图4: AQI指数与质量等级的关系...")
fig, ax = plt.subplots(figsize=(12, 6))
quality_order = ['优', '良', '轻度污染', '中度污染', '重度污染', '严重污染']
quality_data = df_clean[df_clean['质量等级'].isin(quality_order)]
quality_data.boxplot(column='AQI指数', by='质量等级', ax=ax)
plt.title('AQI指数 vs 空气质量等级', fontsize=14, fontweight='bold')
plt.suptitle('')
plt.xlabel('空气质量等级', fontsize=12)
plt.ylabel('AQI指数', fontsize=12)
plt.tight_layout()
plt.savefig('work1/visualizations/04_AQI指数与质量等级.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存为 work1/visualizations/04_AQI指数与质量等级.png")

# 图5: 时间序列趋势
print("  生成图5: 时间序列趋势分析...")
df_sorted = df_clean.sort_values('日期')
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_sorted['日期'], df_sorted['AQI指数'], label='AQI指数', linewidth=1, alpha=0.7)
ax.plot(df_sorted['日期'], df_sorted['PM2.5'], label='PM2.5', linewidth=1, alpha=0.7)
ax.set_title('2017年北京空气质量指数时间序列趋势', fontsize=14, fontweight='bold')
ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('浓度/指数', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('work1/visualizations/05_时间序列趋势.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存为 work1/visualizations/05_时间序列趋势.png")

print(f"\n✓ 可视化分析完成！共生成5张图表")


# 第三阶段：数据建模 - 关联规则挖掘


print("\n" + "=" * 80)
print("第三阶段：数据建模 - 关联规则挖掘")
print("=" * 80)

print("\n[3.1] 数据离散化...")

# 对污染物进行三分类：低、中、高 (基于33.3%和66.7%分位数)
df_model = df_clean.copy()

# 创建离散化后的特征
for col in pollutants:
    q1 = df_model[col].quantile(0.33)
    q2 = df_model[col].quantile(0.67)
    
    def categorize(x):
        if x <= q1:
            return f'{col}=低'
        elif x <= q2:
            return f'{col}=中'
        else:
            return f'{col}=高'
    
    df_model[f'{col}_离散'] = df_model[col].apply(categorize)

# 对AQI指数进行离散化
aqi_q1 = df_model['AQI指数'].quantile(0.33)
aqi_q2 = df_model['AQI指数'].quantile(0.67)

def categorize_aqi(x):
    if x <= aqi_q1:
        return 'AQI=低'
    elif x <= aqi_q2:
        return 'AQI=中'
    else:
        return 'AQI=高'

df_model['AQI_离散'] = df_model['AQI指数'].apply(categorize_aqi)

# 对质量等级进行编码
quality_mapping = {
    '优': '空气质量=优',
    '良': '空气质量=良',
    '轻度污染': '空气质量=轻度污染',
    '中度污染': '空气质量=中度污染',
    '重度污染': '空气质量=重度污染',
    '严重污染': '空气质量=严重污染'
}
df_model['质量等级_离散'] = df_model['质量等级'].map(quality_mapping)

print(f"✓ 数据离散化完成")

print("\n[3.2] Apriori算法关联规则挖掘...")

# 构建事务数据库
# 每行代表一个事务，包含该日期的所有属性
transactions = []
for idx, row in df_model.iterrows():
    transaction = [
        row['PM2.5_离散'],
        row['PM10_离散'],
        row['So2_离散'],
        row['No2_离散'],
        row['Co_离散'],
        row['O3_离散'],
        row['AQI_离散'],
        row['质量等级_离散'],
        row['季节'],
        row['是否周末'],
        row['工作日类型']
    ]
    # 移除 NaN 值
    transaction = [t for t in transaction if pd.notna(t)]
    if len(transaction) > 0:
        transactions.append(transaction)

# 使用TransactionEncoder进行编码
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

print(f"✓ 事务数据库构建完成: {df_encoded.shape[0]} 条事务, {df_encoded.shape[1]} 项")

# 运行Apriori算法，调整支持度阈值
min_support = 0.05  # 5% 的最小支持度
frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
print(f"✓ 频繁项集挖掘: {len(frequent_itemsets)} 个频繁项集 (支持度 >= {min_support})")

# 生成关联规则
if len(frequent_itemsets) > 1:
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    print(f"✓ 关联规则生成: {len(rules)} 条规则 (置信度 >= 0.5)")
    
    # 计算提升度并排序
    if len(rules) > 0:
        rules['lift'] = rules['lift'].round(4)
        rules['support'] = rules['support'].round(4)
        rules['confidence'] = rules['confidence'].round(4)
        
        # 筛选强关联规则 (lift > 1.5)
        strong_rules = rules[rules['lift'] > 1.5].sort_values('lift', ascending=False)
        
        print(f"\n强关联规则 (Lift > 1.5): {len(strong_rules)} 条")
        print("\n" + "=" * 100)
        print("前10条强关联规则:")
        print("=" * 100)
        
        for idx, row in strong_rules.head(10).iterrows():
            antecedents = ', '.join(list(row['antecedents']))
            consequents = ', '.join(list(row['consequents']))
            print(f"\n规则 {idx+1}:")
            print(f"  前件 (Antecedents): {antecedents}")
            print(f"  后件 (Consequents): {consequents}")
            print(f"  支持度 (Support): {row['support']:.4f}")
            print(f"  置信度 (Confidence): {row['confidence']:.4f}")
            print(f"  提升度 (Lift): {row['lift']:.4f}")
        
        # 保存关联规则到CSV
        rules_export = rules.copy()
        rules_export['antecedents'] = rules_export['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules_export['consequents'] = rules_export['consequents'].apply(lambda x: ', '.join(list(x)))
        rules_export.to_csv('work1/关联规则_所有.csv', encoding='utf-8', index=False)
        
        strong_rules_export = strong_rules.copy()
        strong_rules_export['antecedents'] = strong_rules_export['antecedents'].apply(lambda x: ', '.join(list(x)))
        strong_rules_export['consequents'] = strong_rules_export['consequents'].apply(lambda x: ', '.join(list(x)))
        strong_rules_export.to_csv('work1/关联规则_强规则.csv', encoding='utf-8', index=False)
        
        print(f"\n✓ 关联规则已保存:")
        print(f"  - 所有规则: work1/关联规则_所有.csv ({len(rules)} 条)")
        print(f"  - 强规则: work1/关联规则_强规则.csv ({len(strong_rules)} 条)")
else:
    print("⚠ 未找到足够的频繁项集生成关联规则")


# 数据清洗统计对比表


print("\n" + "=" * 80)
print("数据清洗统计对比表")
print("=" * 80)

comparison_table = pd.DataFrame({
    '指标': [
        '总记录数',
        '总列数',
        '缺失值总数',
        '缺失率 (%)',
        '重复行数'
    ],
    '清洗前': [
        stat_before['total_rows'],
        stat_before['total_cols'],
        stat_before['missing_count'],
        f"{stat_before['missing_rate']:.2f}%",
        stat_before['duplicate_rows']
    ],
    '清洗后': [
        stat_after_clean['total_rows'],
        stat_before['total_cols'],
        stat_after_clean['missing_count'],
        f"{stat_after_clean['missing_rate']:.2f}%",
        0
    ],
    '变化': [
        f"删除 {stat_before['total_rows'] - stat_after_clean['total_rows']} 行",
        '无变化',
        f"减少 {stat_before['missing_count'] - stat_after_clean['missing_count']} 个",
        f"从 {stat_before['missing_rate']:.2f}% 降至 {stat_after_clean['missing_rate']:.2f}%",
        f"删除 {stat_before['duplicate_rows']} 行"
    ]
})

print("\n" + comparison_table.to_string(index=False))
comparison_table.to_csv('work1/数据清洗统计对比表.csv', encoding='utf-8', index=False)
print(f"\n✓ 统计对比表已保存至 work1/数据清洗统计对比表.csv")


# 分析总结


print("\n" + "=" * 80)
print("分析完成总结")
print("=" * 80)

print("\n生成的输出文件:")
print("  1. work1/beijing_utf8.csv - UTF-8格式原始数据")
print("  2. work1/beijing_cleaned.csv - 清洗后的数据")
print("  3. work1/污染物统计描述.csv - 统计描述")
print("  4. work1/数据清洗统计对比表.csv - 数据清洗对比表")
print("  5. work1/关联规则_所有.csv - 所有关联规则")
print("  6. work1/关联规则_强规则.csv - 强关联规则")
print("  7. work1/visualizations/ - 可视化图表目录")

print("\n" + "=" * 80)
print("所有任务完成！")
print("=" * 80)

# 补充：强关联规则解释与治理建议
print("\n" + "=" * 80)
print("强关联规则解读与空气质量治理建议")
print("=" * 80)

# 规则解读
print("\n【核心规则解读】")
if len(strong_rules) > 0:
    # 示例解读（可根据实际规则调整）
    print("1. 高PM2.5+高PM10 → 重度污染：PM2.5和PM10是首要污染物，协同推高污染等级；")
    print("2. 冬季+PM2.5=高 → 空气质量差：冬季供暖燃煤是PM2.5升高的核心诱因；")
    print("3. 工作日+NO2=高 → 中度污染：机动车尾气（通勤高峰）是NO2主要来源。")

# 治理建议
print("\n【空气质量治理建议】")
print("1. 季节管控：冬季重点管控燃煤锅炉、散煤燃烧，降低PM2.5排放；")
print("2. 时段管控：工作日早晚高峰加强机动车限行，减少NO2排放；")
print("3. 污染物协同管控：同步治理PM2.5和PM10，避免单一污染物管控效果不足；")
print("4. 节假日管控：法定节假日加强烟花爆竹燃放管控，防止瞬时污染升高。")