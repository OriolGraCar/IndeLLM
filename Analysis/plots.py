import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font', size=15)
plt.rc('axes', titlesize=15)
plt.rc('axes', labelsize=15)

#Import data
overall = pd.read_csv("overall.csv")
alldata = pd.read_csv("Label_matched.csv")

#Add indel type info
overall['indel'] = overall.apply(lambda row: 'deletion' if row['length_mut'] < row['length_wt'] else 'insertion', axis=1)
overall['length_indel'] = overall.apply(lambda row: row['length_wt'] - row['length_mut'] if row['indel'] == 'deletion' else row['length_mut'] - row['length_wt'], axis=1) 

#Add difference in fitness info, positive value positive effect, negative value negative effect
overall['dif_evo'] = overall['evo_mut'] - overall['evo_wt']
overall['dif_fit'] = overall['fit_mut'] - overall['fit_wt']

#clean up labels
overall['labels_cleaned'] = np.where(overall['label'].str.contains('benign', case=False, na=False), 
                                'Benign', 
                                np.where(overall['label'].str.contains('pathogenic', case=False, na=False), 
                                         'Pathogenic', 
                                         np.nan))

#Creating colors for labels
color_map = {'Pathogenic': 'orange', 'Benign': 'blue'}
colors = overall['labels_cleaned'].map(color_map)
legend_labels = {label: color for label, color in color_map.items() if label in overall['labels_cleaned'].unique()}
handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color)
           for label, color in legend_labels.items()]

#Calculating average dif values per bening and pathogenic
evomeanpath = overall[overall['labels_cleaned'] == 'Pathogenic']['dif_evo'].mean()
evomeanbeg = overall[overall['labels_cleaned'] == 'Benign']['dif_evo'].mean()

fitmeanpath = overall[overall['labels_cleaned'] == 'Pathogenic']['dif_fit'].mean()
fitmeanbeg = overall[overall['labels_cleaned'] == 'Benign']['dif_fit'].mean()

print(evomeanpath)
print(evomeanbeg)
print(fitmeanpath)
print(fitmeanbeg)

#Plotting evo scores
plt.figure(figsize=(10,5))
plt.scatter(overall['pair_id'],overall['dif_evo'], c=colors)
plt.ylabel('Dif_evo')
plt.xlabel('id')
plt.axhline(y=evomeanpath, color = 'orange')
plt.axhline(y=evomeanbeg, color = 'blue')
plt.legend(handles=handles, loc = "best")
plt.savefig('dif_evo.png')

#Plotting fit scores
plt.figure(figsize=(10,5))
plt.scatter(overall['pair_id'],overall['dif_fit'], c=colors)
plt.axhline(y=fitmeanpath, color = 'orange')
plt.axhline(y=fitmeanbeg, color = 'blue')
plt.ylabel('Dif_fit')
plt.xlabel('id')
plt.legend(handles=handles, loc = "best")
plt.savefig('dif_fit.png')

#Plotting evo and fit scores
plt.figure()
plt.scatter(overall['dif_fit'], overall['dif_evo'], c=colors)
plt.xlabel('fit')
plt.ylabel('evo')
plt.axhline(y=evomeanpath, color = 'orange')
plt.axhline(y=evomeanbeg, color = 'blue')
plt.axvline(x=fitmeanpath, color = 'orange')
plt.axvline(x=fitmeanbeg, color = 'blue')
plt.legend(handles=handles, loc = 'best')
plt.savefig('fit_evo.png')

#Comparing to other predictors
keptlabels = alldata[alldata['Id_matching'].isin(overall['pair_id'])]

plt.figure(figsize=(10,5))
plt.scatter(keptlabels['Id_matching'], keptlabels['VEST-indel'], c=colors)
plt.ylabel('VEST')
plt.xlabel('ID')
plt.legend(handles=handles, loc = "best")
plt.savefig('VEST.png')

plt.figure()
sns.boxplot(x = overall['labels_cleaned'], y = keptlabels['VEST-indel'])
plt.savefig('VEST_boxplot.png')



print(keptlabels['FATHMM-indel'])

plt.figure(figsize=(10,5))
plt.scatter(keptlabels['FATHMM-indel'], overall['dif_evo'], c=colors)
plt.ylabel('Dif_evo')
plt.xlabel('FATHMM-indel')
plt.legend(handles=handles, loc = "best")
plt.savefig('FATHMM_evo.png')

plt.figure(figsize=(10,5))
plt.scatter(keptlabels['PROVEAN'],overall['dif_evo'], c=colors)
plt.ylabel('Dif_evo')
plt.xlabel('PROVEAN')
plt.legend(handles=handles, loc = "best")
plt.savefig('PROVEAN_evo.png')


#identifying the low scoring benign
fit_under8 = overall.loc[(overall['labels_cleaned'] == 'Benign') & (overall['dif_fit'] < -8)]
print(fit_under8)

#Partly explained by indel size? removed the largest group, ( > 10)
overall_small = overall[overall['length_indel'] < 11]

#Calculate new mean excluding the large indels
evomeanpathsmall = overall_small[overall_small['labels_cleaned'] == 'Pathogenic']['dif_evo'].mean()
evomeanbegsmall = overall_small[overall_small['labels_cleaned'] == 'Benign']['dif_evo'].mean()
fitmeanpathsmall = overall_small[overall_small['labels_cleaned'] == 'Pathogenic']['dif_fit'].mean()
fitmeanbegsmall = overall_small[overall_small['labels_cleaned'] == 'Benign']['dif_fit'].mean()

print(evomeanpathsmall)
print(evomeanbegsmall)
print(fitmeanpathsmall)
print(fitmeanbegsmall)

colors_small = overall_small['labels_cleaned'].map(color_map)
legend_labels_small = {label: color for label, color in color_map.items() if label in overall_small['labels_cleaned'].unique()}
handles_small = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10, markerfacecolor=color)
           for label, color in legend_labels_small.items()]

#Plotting evo scores for small indels
plt.figure(figsize=(10,5))
plt.scatter(overall_small['pair_id'],overall_small['dif_evo'], c=colors_small)
plt.ylabel('Dif_evo')
plt.xlabel('id')
plt.axhline(y=evomeanpathsmall, color = 'orange')
plt.axhline(y=evomeanbegsmall, color = 'blue')
plt.legend(handles=handles_small, loc = "best")
plt.savefig('dif_evo_smallindels.png')

#Plotting fit scores for small indels
plt.figure(figsize=(10,5))
plt.scatter(overall_small['pair_id'],overall_small['dif_fit'], c=colors_small)
plt.axhline(y=fitmeanpathsmall, color = 'orange')
plt.axhline(y=fitmeanbegsmall, color = 'blue')
plt.ylabel('Dif_fit')
plt.xlabel('id')
plt.legend(handles=handles_small, loc = "best")
plt.savefig('dif_fit_smallindels.png')

#Plotting fit and evo scores for small indels
plt.figure()
plt.scatter(overall_small['dif_fit'], overall_small['dif_evo'], c=colors_small)
plt.xlabel('fit')
plt.ylabel('evo')
plt.axhline(y=evomeanpathsmall, color = 'orange')
plt.axhline(y=evomeanbegsmall, color = 'blue')
plt.axvline(x=fitmeanpathsmall, color = 'orange')
plt.axvline(x=fitmeanbegsmall, color = 'blue')
plt.legend(handles=handles_small, loc = 'best')
plt.savefig('fit_evo_smallindels.png')

#Boxplot
plt.figure()
sns.boxplot(x = overall['labels_cleaned'], y = overall['dif_evo'])
plt.savefig('dif_evo_boxplot.png')

plt.figure()
sns.boxplot(x = overall['labels_cleaned'], y = overall['dif_fit'])
plt.savefig('dif_fit_boxplot.png')

#Splitting the overall after indels sizes. Uing the same sizes as in Cannon et al. 2023 (1, 2-4, 5-10 and 11+)

overall_1 = overall[overall['length_indel'] == 1]
overall_2to4 = overall[overall['length_indel'].between(2, 4)]
overall_5to10 = overall[overall['length_indel'].between(5, 10)]
overall_11pluss = overall[overall['length_indel'] > 10]

evomeanpath_1 = overall_1[overall_1['labels_cleaned'] == 'Pathogenic']['dif_evo'].mean()
evomeanpath_2to4 = overall_2to4[overall_2to4['labels_cleaned'] == 'Pathogenic']['dif_evo'].mean()
evomeanpath_5to10 = overall_5to10[overall_5to10['labels_cleaned'] == 'Pathogenic']['dif_evo'].mean()
evomeanpath_11pluss = overall_11pluss[overall_11pluss['labels_cleaned'] == 'Pathogenic']['dif_evo'].mean()
evomeanbeg_1 = overall_1[overall_1['labels_cleaned'] == 'Benign']['dif_evo'].mean()
evomeanbeg_2to4 = overall_2to4[overall_2to4['labels_cleaned'] == 'Benign']['dif_evo'].mean()
evomeanbeg_5to10 = overall_5to10[overall_5to10['labels_cleaned'] == 'Benign']['dif_evo'].mean()
evomeanbeg_11pluss = overall_11pluss[overall_11pluss['labels_cleaned'] == 'Benign']['dif_evo'].mean()

fitmeanpath_1 = overall_1[overall_1['labels_cleaned'] == 'Pathogenic']['dif_fit'].mean()
fitmeanpath_2to4 = overall_2to4[overall_2to4['labels_cleaned'] == 'Pathogenic']['dif_fit'].mean()
fitmeanpath_5to10 = overall_5to10[overall_5to10['labels_cleaned'] == 'Pathogenic']['dif_fit'].mean()
fitmeanpath_11pluss = overall_11pluss[overall_11pluss['labels_cleaned'] == 'Pathogenic']['dif_fit'].mean()
fitmeanbeg_1 = overall_1[overall_1['labels_cleaned'] == 'Benign']['dif_fit'].mean()
fitmeanbeg_2to4 = overall_2to4[overall_2to4['labels_cleaned'] == 'Benign']['dif_fit'].mean()
fitmeanbeg_5to10 = overall_5to10[overall_5to10['labels_cleaned'] == 'Benign']['dif_fit'].mean()
fitmeanbeg_11pluss = overall_11pluss[overall_11pluss['labels_cleaned'] == 'Benign']['dif_fit'].mean()

print("1 indels")
print("evo", evomeanpath_1)
print("evo", evomeanbeg_1)
print("fit", fitmeanpath_1)
print("fit", fitmeanbeg_1)

print("2-4 indels")
print("evo", evomeanpath_2to4)
print("evo", evomeanbeg_2to4)
print("fit", fitmeanpath_2to4)
print("fit", fitmeanbeg_2to4)

print("5to10 indels")
print("evo", evomeanpath_5to10)
print("evo", evomeanbeg_5to10)
print("fit", fitmeanpath_5to10)
print("fit", fitmeanbeg_5to10)

print("11 pluss indels")
print("evo", evomeanpath_11pluss)
print("evo", evomeanbeg_11pluss)
print("fit", fitmeanpath_11pluss)
print("fit", fitmeanbeg_11pluss)

#plotting boxplots
plt.figure()
sns.boxplot(x = overall_1['labels_cleaned'], y = overall_1['dif_evo'])
plt.savefig('dif_evo_1_boxplot.png')

plt.figure()
sns.boxplot(x = overall_2to4['labels_cleaned'], y = overall_2to4['dif_evo'])
plt.savefig('dif_evo_2to4_boxplot.png')

plt.figure()
sns.boxplot(x = overall_5to10['labels_cleaned'], y = overall_5to10['dif_evo'])
plt.savefig('dif_evo_5to10_boxplot.png')

plt.figure()
sns.boxplot(x = overall_11pluss['labels_cleaned'], y = overall_11pluss['dif_evo'])
plt.savefig('dif_evo_11pluss_boxplot.png')

plt.figure()
sns.boxplot(x = overall_1['labels_cleaned'], y = overall_1['dif_fit'])
plt.savefig('dif_fit_1_boxplot.png')

plt.figure()
sns.boxplot(x = overall_2to4['labels_cleaned'], y = overall_2to4['dif_fit'])
plt.savefig('dif_fit_2to4_boxplot.png')

plt.figure()
sns.boxplot(x = overall_5to10['labels_cleaned'], y = overall_5to10['dif_fit'])
plt.savefig('dif_fit_5to10_boxplot.png')

plt.figure()
sns.boxplot(x = overall_11pluss['labels_cleaned'], y = overall_11pluss['dif_fit'])
plt.savefig('dif_fit_11pluss_boxplot.png')

#Splitting by length of protein
overall_100 = overall[overall['length_wt'] > 100]
overall_100to200 = overall[overall['length_wt'].between(100, 200)]
overall_200to300 = overall[overall['length_wt'].between(200, 300)]
overall_300to400 = overall[overall['length_wt'].between(300, 400)]
overall_400to500 = overall[overall['length_wt'].between(400, 500)]

evomeanpath_100 = overall_100[overall_100['labels_cleaned'] == 'Pathogenic']['dif_evo'].mean()
evomeanpath_100to200 = overall_100to200[overall_100to200['labels_cleaned'] == 'Pathogenic']['dif_evo'].mean()
evomeanpath_200to300 = overall_200to300[overall_200to300['labels_cleaned'] == 'Pathogenic']['dif_evo'].mean()
evomeanpath_300to400 = overall_300to400[overall_300to400['labels_cleaned'] == 'Pathogenic']['dif_evo'].mean()
evomeanpath_400to500 = overall_400to500[overall_400to500['labels_cleaned'] == 'Pathogenic']['dif_evo'].mean()
evomeanbeg_100 = overall_100[overall_100['labels_cleaned'] == 'Benign']['dif_evo'].mean()
evomeanbeg_100to200 = overall_100to200[overall_100to200['labels_cleaned'] == 'Benign']['dif_evo'].mean()
evomeanbeg_200to300 = overall_200to300[overall_200to300['labels_cleaned'] == 'Benign']['dif_evo'].mean()
evomeanbeg_300to400 = overall_300to400[overall_300to400['labels_cleaned'] == 'Benign']['dif_evo'].mean()
evomeanbeg_400to500 = overall_400to500[overall_400to500['labels_cleaned'] == 'Benign']['dif_evo'].mean()

fitmeanpath_100 = overall_100[overall_100['labels_cleaned'] == 'Pathogenic']['dif_fit'].mean()
fitmeanpath_100to200 = overall_100to200[overall_100to200['labels_cleaned'] == 'Pathogenic']['dif_fit'].mean()
fitmeanpath_200to300 = overall_200to300[overall_200to300['labels_cleaned'] == 'Pathogenic']['dif_fit'].mean()
fitmeanpath_300to400 = overall_300to400[overall_300to400['labels_cleaned'] == 'Pathogenic']['dif_fit'].mean()
fitmeanpath_400to500 = overall_400to500[overall_400to500['labels_cleaned'] == 'Pathogenic']['dif_fit'].mean()
fitmeanbeg_100 = overall_100[overall_100['labels_cleaned'] == 'Benign']['dif_fit'].mean()
fitmeanbeg_100to200 = overall_100to200[overall_100to200['labels_cleaned'] == 'Benign']['dif_fit'].mean()
fitmeanbeg_200to300 = overall_200to300[overall_200to300['labels_cleaned'] == 'Benign']['dif_fit'].mean()
fitmeanbeg_300to400 = overall_300to400[overall_300to400['labels_cleaned'] == 'Benign']['dif_fit'].mean()
fitmeanbeg_400to500 = overall_400to500[overall_400to500['labels_cleaned'] == 'Benign']['dif_fit'].mean()

print(" > 100aa proteins")
print("evo", evomeanpath_100)
print("evo", evomeanbeg_100)
print("fit", fitmeanpath_100)
print("fit", fitmeanbeg_100)

print("100-200 aa proteins")
print("evo", evomeanpath_100to200)
print("evo", evomeanbeg_100to200)
print("fit", fitmeanpath_100to200)
print("fit", fitmeanbeg_100to200)

print("200-300 aa proteins")
print("evo", evomeanpath_200to300)
print("evo", evomeanbeg_200to300)
print("fit", fitmeanpath_200to300)
print("fit", fitmeanbeg_200to300)

print("300-400 aa proteins")
print("evo", evomeanpath_300to400)
print("evo", evomeanbeg_300to400)
print("fit", fitmeanpath_300to400)
print("fit", fitmeanbeg_300to400)

print("400-500 aa proteins")
print("evo", evomeanpath_400to500)
print("evo", evomeanbeg_400to500)
print("fit", fitmeanpath_400to500)
print("fit", fitmeanbeg_400to500)

plt.figure()
sns.boxplot(x = overall_100['labels_cleaned'], y = overall_100['dif_evo'])
plt.savefig('dif_evo_100_boxplot.png')

plt.figure()
sns.boxplot(x = overall_100to200['labels_cleaned'], y = overall_100to200['dif_evo'])
plt.savefig('dif_evo_100to200_boxplot.png')

plt.figure()
sns.boxplot(x = overall_200to300['labels_cleaned'], y = overall_200to300['dif_evo'])
plt.savefig('dif_evo_200to300_boxplot.png')

plt.figure()
sns.boxplot(x = overall_300to400['labels_cleaned'], y = overall_300to400['dif_evo'])
plt.savefig('dif_evo_300to400_boxplot.png')

plt.figure()
sns.boxplot(x = overall_400to500['labels_cleaned'], y = overall_400to500['dif_evo'])
plt.savefig('dif_evo_400to500_boxplot.png')

plt.figure()
sns.boxplot(x = overall_100['labels_cleaned'], y = overall_100['dif_fit'])
plt.savefig('dif_fit_100_boxplot.png')

plt.figure()
sns.boxplot(x = overall_100to200['labels_cleaned'], y = overall_100to200['dif_fit'])
plt.savefig('dif_fit_100to200_boxplot.png')

plt.figure()
sns.boxplot(x = overall_200to300['labels_cleaned'], y = overall_200to300['dif_fit'])
plt.savefig('dif_fit_200to300_boxplot.png')

plt.figure()
sns.boxplot(x = overall_300to400['labels_cleaned'], y = overall_300to400['dif_fit'])
plt.savefig('dif_fit_300to400_boxplot.png')

plt.figure()
sns.boxplot(x = overall_400to500['labels_cleaned'], y = overall_400to500['dif_fit'])
plt.savefig('dif_fit_400to500_boxplot.png')

