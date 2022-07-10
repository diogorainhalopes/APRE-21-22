#%%
from scipy.io import arff
import pandas as pd
import matplotlib.pyplot as plt

cancer = arff.loadarff(r'../data/breast.w.arff')
df = pd.DataFrame(cancer[0])
df.dropna(inplace=True)
class1 = df['Class'][0]
dfc1 = df.loc[df['Class'] == class1]
dfc2 = df.loc[df['Class'] != class1]
n_bins = 10

x1 = [dfc1['Clump_Thickness'], dfc2['Clump_Thickness']]
x2 = [dfc1['Cell_Size_Uniformity'], dfc2['Cell_Size_Uniformity']]
x3 = [dfc1['Cell_Shape_Uniformity'], dfc2['Cell_Shape_Uniformity']]
x4 = [dfc1['Marginal_Adhesion'], dfc2['Marginal_Adhesion']]
x5 = [dfc1['Single_Epi_Cell_Size'], dfc2['Single_Epi_Cell_Size']]
x6 = [dfc1['Bare_Nuclei'], dfc2['Bare_Nuclei']]
x7 = [dfc1['Bland_Chromatin'], dfc2['Bland_Chromatin']]
x8 = [dfc1['Normal_Nucleoli'], dfc2['Normal_Nucleoli']]
x9 = [dfc1['Mitoses'], dfc2['Mitoses']]

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3)

classes = ['benign','malignant']
colors = ['lime', 'green']
ax1.hist(x1, n_bins, histtype='bar', color=colors, stacked=True , label=classes)
ax1.legend()
ax1.set_title('Clump_Thickness')

ax2.hist(x2, n_bins, histtype='bar', color=colors, stacked=True , label=classes)
ax2.legend()
ax2.set_title('Cell_Size_Uniformity')

ax3.hist(x3, n_bins, histtype='bar', color=colors, stacked=True , label=classes)
ax3.legend()
ax3.set_title('Cell_Shape_Uniformity')

ax4.hist(x4, n_bins, histtype='bar', color=colors, stacked=True , label=classes)
ax4.legend()
ax4.set_title('Marginal_Adhesion')

ax5.hist(x5, n_bins, histtype='bar', color=colors, stacked=True , label=classes)
ax5.legend()
ax5.set_title('Single_Epi_Cell_Size')

ax6.hist(x6, n_bins, histtype='bar', color=colors, stacked=True , label=classes)
ax6.legend()
ax6.set_title('Bare_Nuclei')

ax7.hist(x7, n_bins, histtype='bar', color=colors, stacked=True , label=classes)
ax7.legend()
ax7.set_title('Bland_Chromatin')

ax8.hist(x8, n_bins, histtype='bar', color=colors, stacked=True , label=classes)
ax8.legend()
ax8.set_title('Normal_Nucleoli')

ax9.hist(x9, n_bins, histtype='bar', color=colors, stacked=True , label=classes)
ax9.legend()
ax9.set_title('Mitoses')

fig.tight_layout()
plt.show()
#%%