import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spectral import *
import scipy.signal
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 读取参考板反射率数据
fgb_df = pd.read_csv('work2/data_FGB.csv')
wavelengths_fgb = fgb_df['Wave'].values
R30_fgb = fgb_df['30'].values
R3_fgb = fgb_df['3'].values

# 假设ENVI波长从400到990，300个bands
wavelengths = np.linspace(400, 990, 300)
R30 = np.interp(wavelengths, wavelengths_fgb, R30_fgb)
R3 = np.interp(wavelengths, wavelengths_fgb, R3_fgb)

# 读取生理指标数据
gt_df = pd.read_excel('work2/GT_values.xlsx')
# 提取样本编号，去掉C7_前缀
gt_df['sample_id'] = gt_df['Sample_ID'].str.split('_').str[1]

print("FGB data shape:", fgb_df.shape)
print("GT data shape:", gt_df.shape)
print("Sample IDs:", gt_df['sample_id'].values)

# 函数：读取ENVI数据并返回平均DN
def read_envi_avg(hdr_path):
    img = open_image(hdr_path)
    data = img.load()
    # 平均所有像素
    avg_dn = np.mean(data, axis=(0,1))
    return avg_dn

# 样本文件夹
raw_dir = '00-Raw_Spectral'
sample_dirs = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
print("Sample directories:", sample_dirs)

# 存储校准后的反射率
calibrated_data = []

for sample_id in gt_df['sample_id']:
    sample_dir = os.path.join(raw_dir, sample_id)
    if not os.path.exists(sample_dir):
        print(f"Sample {sample_id} directory not found")
        continue
    
    # 读取B (3%), W (30%), P (leaf)
    b_hdr = os.path.join(sample_dir, f'{sample_id}B.hdr')
    w_hdr = os.path.join(sample_dir, f'{sample_id}W.hdr')
    p_hdr = os.path.join(sample_dir, f'{sample_id}P.hdr')
    
    if not all(os.path.exists(f) for f in [b_hdr, w_hdr, p_hdr]):
        print(f"Missing files for {sample_id}")
        continue
    
    dn_3 = read_envi_avg(b_hdr)
    dn_30 = read_envi_avg(w_hdr)
    dn_leaf = read_envi_avg(p_hdr)
    
    # 计算反射率
    R = ((dn_leaf - dn_3) / (dn_30 - dn_3)) * (R30 - R3) + R3
    
    calibrated_data.append({
        'sample_id': sample_id,
        'reflectance': R
    })

print(f"Calibrated {len(calibrated_data)} samples")

# 任务1：绘制光谱曲线
plt.figure(figsize=(12, 8))
for i, data in enumerate(calibrated_data):
    plt.plot(wavelengths, data['reflectance'], label=f'Sample {data["sample_id"]}')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.title('Leaf Reflectance Spectra')
plt.legend()
plt.grid(True)
# 标注特征点
plt.axvline(x=550, color='green', linestyle='--', label='550nm Green Peak')
plt.axvline(x=680, color='red', linestyle='--', label='680nm Absorption Pit')
plt.legend()
plt.savefig('reflectance_curves.png')
# plt.show()  # 移除以避免阻塞

# 合并数据
reflectance_df = pd.DataFrame([d['reflectance'] for d in calibrated_data], 
                              columns=[f'wl_{w:.1f}' for w in wavelengths])
reflectance_df['sample_id'] = [d['sample_id'] for d in calibrated_data]

# 合并生理指标
merged_df = pd.merge(reflectance_df, gt_df, on='sample_id')
merged_df.to_csv('calibrated_phenotype_data.csv', index=False)
print("Saved calibrated_phenotype_data.csv")

# 任务2：光谱预处理
# Savitzky-Golay 平滑
def sg_smooth(spectra, window_length=11, polyorder=2):
    return scipy.signal.savgol_filter(spectra, window_length, polyorder, axis=1)

# 一阶导数
def first_derivative(spectra):
    return np.gradient(spectra, axis=1)

# 多元散射校正 (MSC)
def msc(spectra):
    mean_spectrum = np.mean(spectra, axis=0)
    corrected = np.zeros_like(spectra)
    for i in range(spectra.shape[0]):
        slope, intercept = np.polyfit(mean_spectrum, spectra[i], 1)
        corrected[i] = (spectra[i] - intercept) / slope
    return corrected

spectra = np.array([d['reflectance'] for d in calibrated_data])

# 平滑
smoothed = sg_smooth(spectra)

# 绘制对比
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.plot(wavelengths, spectra[0], label='Original')
plt.plot(wavelengths, smoothed[0], label='Smoothed')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.title('Original vs Smoothed Spectrum')
plt.legend()

# 一阶导数
deriv = first_derivative(spectra)
plt.subplot(1,2,2)
plt.plot(wavelengths, deriv[0], label='First Derivative')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Derivative')
plt.title('First Derivative')
plt.legend()
plt.tight_layout()
plt.savefig('preprocessing_comparison.png')
# plt.show()

# MSC
msc_corrected = msc(spectra)
print("Preprocessing completed")

# 任务3：特征工程与关联分析
# 相关性分析
targets = ['SPAD', 'Pn', 'LCN', 'Ca', 'Cb', 'SLW']
correlations = {}

for target in targets:
    corr = []
    for i in range(len(wavelengths)):
        wl_col = f'wl_{wavelengths[i]:.1f}'
        corr_val = merged_df[wl_col].corr(merged_df[target])
        corr.append(corr_val)
    correlations[target] = np.array(corr)

# 绘制相关性谱图
plt.figure(figsize=(12, 8))
for target in targets:
    plt.plot(wavelengths, correlations[target], label=target)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Pearson Correlation')
plt.title('Correlation Spectra')
plt.legend()
plt.grid(True)
plt.savefig('correlation_spectra.png')
# plt.show()

# 找出敏感波长区域
for target in targets:
    max_idx = np.argmax(np.abs(correlations[target]))
    print(f"{target} sensitive wavelength: {wavelengths[max_idx]:.1f} nm, corr: {correlations[target][max_idx]:.3f}")

# PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(spectra)
explained_var = pca.explained_variance_ratio_
print(f"Explained variance: PC1={explained_var[0]:.3f}, PC2={explained_var[1]:.3f}, PC3={explained_var[2]:.3f}")

# 3D散点图
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_result[:,0], pca_result[:,1], pca_result[:,2])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('PCA 3D Scatter Plot')
plt.savefig('pca_3d.png')
# plt.show()

# 任务4：预测建模
# 选择目标变量，例如SPAD
target = 'SPAD'
X = spectra  # 或smoothed, msc_corrected等
y = merged_df[target].values

# 由于样本少，使用留一交叉验证
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()

# PLSR
plsr = PLSRegression(n_components=5)
plsr_preds = []
for train_idx, test_idx in loo.split(X):
    plsr.fit(X[train_idx], y[train_idx])
    pred = plsr.predict(X[test_idx])
    plsr_preds.append(pred.flatten()[0])  # 修正

plsr_preds = np.array(plsr_preds)
plsr_r2 = r2_score(y, plsr_preds)
plsr_rmse = np.sqrt(mean_squared_error(y, plsr_preds))
y_std = np.std(y)
plsr_rpd = y_std / plsr_rmse

print(f"PLSR - R2: {plsr_r2:.3f}, RMSE: {plsr_rmse:.3f}, RPD: {plsr_rpd:.3f}")

# 随机森林
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf_preds = []
for train_idx, test_idx in loo.split(X):
    rf.fit(X[train_idx], y[train_idx])
    pred = rf.predict(X[test_idx])
    rf_preds.append(pred[0])

rf_preds = np.array(rf_preds)
rf_r2 = r2_score(y, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y, rf_preds))
rf_rpd = y_std / rf_rmse

print(f"RF - R2: {rf_r2:.3f}, RMSE: {rf_rmse:.3f}, RPD: {rf_rpd:.3f}")

# 绘制散点图
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.scatter(y, plsr_preds)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('True SPAD')
plt.ylabel('Predicted SPAD')
plt.title(f'PLSR: R2={plsr_r2:.3f}, RMSE={plsr_rmse:.3f}')
plt.grid(True)

plt.subplot(1,2,2)
plt.scatter(y, rf_preds)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('True SPAD')
plt.ylabel('Predicted SPAD')
plt.title(f'RF: R2={rf_r2:.3f}, RMSE={rf_rmse:.3f}')
plt.grid(True)
plt.tight_layout()
plt.savefig('prediction_scatter.png')
# plt.show()

print("All tasks completed")