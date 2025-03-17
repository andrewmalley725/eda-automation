class EDA:
    def __init__(self, df):
        self.df = df

    def calculateUnivariateStatsViz(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        new_df = pd.DataFrame(columns=['Count', 'Unique', 'Data Type', 'Missing', 'Mode', 'Min', '25%', 'Median', '75%', 'Max', 'STD Dev', 'Mean', 'Skew', 'Kurt'])
        for col in self.df:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                f, (ax_box, ax) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (.15, .65)})
                sns.set(style='ticks')
                flierprops = dict(marker='o', markersize=4, markerfacecolor='none', linestyle='none', markeredgecolor='gray')
                sns.boxplot(x=self.df[col], ax=ax_box, fliersize=4, width=.50, linewidth=1, flierprops=flierprops)
                sns.histplot(self.df, x=self.df[col])
                sns.despine(ax=ax)
                sns.despine(ax=ax_box, left=True, bottom=True)
                ax_box.set_title(col, fontsize=14)
                new_df.loc[col] = [self.df[col].count(), round(self.df[col].nunique(), 2), str(self.df[col].dtype), round(self.df[col].isnull().sum(), 2), self.df[col].mode().values[0], round(self.df[col].min(), 2), round(self.df[col].quantile(.25), 2), round(self.df[col].median(), 2), round(self.df[col].quantile(.75), 2), round(self.df[col].max(), 2), round(self.df[col].std(), 2), round(self.df[col].mean(), 2), round(self.df[col].skew(), 2), round(self.df[col].kurt(), 2)]
                text = 'Count: ' + str(self.df[col].count()) + '\n'
                text += 'Unique: ' + str(round(self.df[col].nunique(), 2)) + '\n'
                text += 'Data Type: ' + str(self.df[col].dtype) + '\n'
                text += 'Missing: ' + str(round(self.df[col].isnull().sum(), 2)) + '\n'
                text += 'Mode: ' + str(self.df[col].mode().values[0]) + '\n'
                text += 'Min: ' + str(round(self.df[col].min(), 2)) + '\n'
                text += '25%: ' + str(round(self.df[col].quantile(.25), 2)) + '\n'
                text += 'Median: ' + str(round(self.df[col].median(), 2)) + '\n'
                text += '75%: ' + str(round(self.df[col].quantile(.75), 2)) + '\n'
                text += 'Max: ' + str(round(self.df[col].max(), 2)) + '\n'
                text += 'Std Dev: ' + str(round(self.df[col].std(), 2)) + '\n'
                text += 'Mean: ' + str(round(self.df[col].mean(), 2)) + '\n'
                text += 'Skew: ' + str(round(self.df[col].skew(), 2)) + '\n'
                text += 'Kurt: ' + str(round(self.df[col].kurt(), 2)) + '\n'
                ax.text(.9, .25, text, fontsize=10, transform=plt.gcf().transFigure)
                plt.show()
            else:
                ax_count = sns.countplot(x=col, data=self.df, order=self.df[col].value_counts().index, palette=sns.color_palette('RdBu_r', self.df[col].nunique()))
                sns.despine(ax=ax_count)
                ax_count.set_title(col)
                ax_count.set_xlabel(col)
                ax_count.set_ylabel('')
                new_df.loc[col] = [self.df[col].count(), round(self.df[col].nunique(), 2), str(self.df[col].dtype), round(self.df[col].isnull().sum(), 2), 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA']
                text = 'Count: ' + str(self.df[col].count()) + '\n'
                text += 'Unique: ' + str(round(self.df[col].nunique(), 2)) + '\n'
                text += 'Data Type: ' + str(self.df[col].dtype) + '\n'
                text += 'Missing: ' + str(round(self.df[col].isna().sum(), 2)) + '\n'
                ax_count.text(.9, .5, text, fontsize=10, transform=plt.gcf().transFigure)
                plt.show()
        return new_df

    def calculateTTest(self, feature, label):
        import pandas as pd
        from scipy import stats
        oString = ''
        feats = self.df[feature].unique()
        df1 = self.df[self.df[feature] == feats[0]]
        df2 = self.df[self.df[feature] == feats[1]]
        t, p = stats.ttest_ind(df1[label], df2[label])
        oString += 'T-TEST \nt stat: ' + str(round(t, 2))
        oString += '\np value: ' + str(round(p, 2))
        return oString, t, p

    def calculateANOVA(self, feature, label):
        import pandas as pd
        from scipy import stats
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        oString = ''
        groups = []
        columns = self.df[feature].unique()
        for col in columns:
            groups.append(self.df[self.df[feature] == col][label])
        f, p = stats.f_oneway(*groups)
        tukey = pairwise_tukeyhsd(endog=self.df[label], groups=self.df[feature])
        oString += 'ANOVA \nF stat: ' + str(round(f, 2)) + '\np value: ' + str(round(p, 2))
        return oString, tukey, f, p

    def createBarChart(self, feature, label):
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        groups = self.df[feature].unique()
        if len(groups) > 2:
            result = self.calculateANOVA(feature, label)[0]
            tukey = self.calculateANOVA(feature, label)[1]
            stat = self.calculateANOVA(feature, label)[2]
            p = self.calculateANOVA(feature, label)[3]
        elif len(groups) == 2:
            result = self.calculateTTest(feature, label)[0]
            tukey = ''
            stat = self.calculateTTest(feature, label)[1]
            p = self.calculateTTest(feature, label)[2]
        print(tukey)
        plot = sns.barplot(data=self.df, x=feature, y=label)
        plot.text(1, 0.8, result, fontsize=12, transform=plt.gcf().transFigure)
        plt.show()
        return plot, groups, stat, p

    def numericToNumericStats(self, feature, label):
        import pandas as pd
        from scipy import stats
        import numpy as np
        oString = ''
        r, p = stats.pearsonr(self.df[feature], self.df[label])
        model = np.polyfit(self.df[feature], self.df[label], 1)
        r2 = r ** 2
        equation = 'y = ' + str(round(model[0], 2)) + 'x + ' + str(round(model[1], 2))
        oString += 'r value: ' + str(round(r, 2)) + '\np value: ' + str(round(p, 2))
        oString += '\nLinear Regression Equation: ' + equation + '\nr squared: ' + str(round(r2, 2))
        return [oString, [r, p, r2, equation]]

    def createScatterPlot(self, feature, label):
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        text = self.numericToNumericStats(feature, label)[0]
        plot = sns.jointplot(x=self.df[feature], y=self.df[label], kind='reg')
        plot.figure.text(1, 0.8, text, fontsize=12, transform=plt.gcf().transFigure)
        plt.show()
        return plot

    def calculateBivariateStatsViz(self, label):
        import pandas as pd
        import seaborn as sns
        from scipy.stats import chi2_contingency
        import matplotlib.pyplot as plt
        new_df = pd.DataFrame(columns=['Stat', '+/-', 'Effect Size', 'p-value'])
        if pd.api.types.is_numeric_dtype(self.df[label]):
            for col in self.df:
                if col != label:
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        r, p, r2, eq = self.numericToNumericStats(col, label)[1]
                        sign = '+' if r > 0 else '-'
                        new_df.loc[col] = ['r', sign, round(abs(r), 2), round(p, 2)]
                        stat = 'r'
                        plot = self.createScatterPlot(col, label)
                        plot.figure.suptitle(col)
                    else:
                        plot, groups, val, p = self.createBarChart(col, label)
                        plot.set_title(col)
                        stat = 'F' if len(groups) > 2 else 'T'
                        new_df.loc[col] = [stat, ' ', round(val, 2), round(p, 2)]
        else:
            for col in self.df:
                if col != label:
                    if pd.api.types.is_numeric_dtype(self.df[col]):
                        plot, groups, val, p = self.createBarChart(label, col)
                        plot.set_title(col)
                        stat = 'F' if len(groups) > 2 else 'T'
                        new_df.loc[col] = [stat, ' ', round(val, 2), round(p, 2)]
                    else:
                        contingency_table = pd.crosstab(self.df[col], self.df[label])
                        chi2, p, dof, ex = chi2_contingency(contingency_table)
                        ax_count = sns.countplot(x=col, hue=label, data=self.df, palette='RdBu_r')
                        sns.despine(ax=ax_count)
                        ax_count.set_title(col)
                        ax_count.set_xlabel(col)
                        ax_count.set_ylabel('Count')
                        text = f'Chi2: {chi2:.2f}\np-value: {p:.2f}'
                        ax_count.text(1, 0.8, text, fontsize=12, transform=plt.gcf().transFigure)
                        new_df.loc[col] = ['Chi2', ' ', round(chi2, 2), round(p, 2)]
                        plt.show()
        return new_df
