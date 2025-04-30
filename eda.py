import pandas as pd
import numpy as np

class EDA:
    def __init__(self, df=None, label=None):
        self.df = df
        self.label = label
        self.label_is_numeric = pd.api.types.is_numeric_dtype(df[label]) if label is not None else False

    def set_df(self, df):
        self.df = df

    def set_label(self, label, is_numeric=False):
        self.label = label
        self.label_is_numeric = is_numeric

    def _cramer_v(self, chi2_stat, n, r, k):
        return np.sqrt(chi2_stat / (n * (min(r,k)) - 1))
    
    def _cohen_d(self, x1, x2):
        n1, n2 = len(x1), len(x2)
        mean1, mean2 = np.mean(x1), np.mean(x2)
        std1, std2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
        return (mean1 - mean2) / pooled_std
    
    def _eta_squared(self, feature, target):
        group_means = self.df.groupby(feature)[target].mean()
        overall_mean = self.df[target].mean()
        ss_between = sum(
            self.df[feature].value_counts()[group] * (mean - overall_mean) ** 2
            for group, mean in group_means.items()
        )
        ss_total = sum((self.df[target] - overall_mean) ** 2)
        return ss_between / ss_total

    def bivariate_stats(self):
        from scipy import stats
        from scipy.stats import chi2_contingency
        final_data = []
        df = self.df.copy()
        for col in df.drop(self.label, axis=1).columns:
            record = {
                'Column Name': col
            }
            if self.label_is_numeric:
                if pd.api.types.is_numeric_dtype(df[col]):
                    record['Type'] = 'Numeric'
                    r, p = stats.pearsonr(df[col], df[self.label])
                    r2 = r ** 2
                    record['Stat'] = 'R'
                    record['Value'] = r
                    record['P-Value'] = p
                    record['Effect Size Type'] = 'Pearson R'
                    record['Effect Size Value'] = r
                    record['R2'] = r2
                    final_data.append(record)
                else:
                    record['Type'] = 'Categorical'
                    groups = df[col].nunique()
                    if groups > 2:
                        record['Stat'] = 'F'
                        f, p = stats.f_oneway(*[df[self.label][df[col] == group] for group in df[col].unique()])
                        record['R2'] = self._eta_squared(col, self.label)
                        record['Value'] = f
                        record['P-Value'] = p
                        record['Effect Size Type'] = 'Eta^2'
                        record['Effect Size Value'] = record['R2']
                        final_data.append(record)
                    elif groups == 2:
                        record['Stat'] = 'T'
                        values = df[col].unique()
                        g1 = values[0]
                        g2 = values[1]
                        x1 = df[self.label][df[col] == g1]
                        x2 = df[self.label][df[col] == g2]
                        record['R2'] = (self._cohen_d(x1, x2) ** 2) / (1 + self._cohen_d(x1, x2) ** 2)
                        t, p = stats.ttest_ind(x1, x2)
                        record['Value'] = t
                        record['P-Value'] = p
                        record['Effect Size Type'] = 'Cohen D'
                        record['Effect Size Value'] = self._cohen_d(x1, x2)
                        final_data.append(record)
            else:
                label_groups = df[self.label].nunique()
                if pd.api.types.is_numeric_dtype(df[col]):
                    record['Type'] = 'Numeric'
                    if label_groups > 2:
                        record['Stat'] = 'F'
                        f, p = stats.f_oneway(*[df[col][df[self.label] == group] for group in df[self.label].unique()])
                        record['R2'] = self._eta_squared(col, self.label)
                        record['Value'] = f
                        record['P-Value'] = p
                        record['Effect Size Type'] = 'Eta^2'
                        record['Effect Size Value'] = record['R2']
                        final_data.append(record)
                    elif label_groups == 2:
                        record['Stat'] = 'T'
                        values = df[self.label].unique()
                        g1 = values[0]
                        g2 = values[1]
                        x1 = df[col][df[self.label] == g1]
                        x2 = df[col][df[self.label] == g2]
                        record['R2'] = (self._cohen_d(x1, x2) ** 2) / (1 + self._cohen_d(x1, x2) ** 2)
                        t, p = stats.ttest_ind(x1,x2)
                        record['Value'] = t
                        record['P-Value'] = p
                        record['Effect Size Type'] = 'Cohen D'
                        record['Effect Size Value'] = self._cohen_d(x1, x2)
                        final_data.append(record)
                else:
                    record['Type'] = 'Categorical'
                    record['Stat'] = 'CHI2'
                    contingency_table = pd.crosstab(df[col], df[self.label])
                    n = contingency_table.sum().sum()
                    r,k = contingency_table.shape
                    chi2, p, _, _ = chi2_contingency(contingency_table)
                    cramer_v = self._cramer_v(chi2, n, r, k)
                    record['Value'] = chi2
                    record['R2'] = cramer_v ** 2
                    record['P-Value'] = p
                    record['Effect Size Type'] = 'Cramer V'
                    record['Effect Size Value'] = cramer_v
                    final_data.append(record)
        final_data = pd.DataFrame(final_data)
        for col in final_data.columns:
            if pd.api.types.is_numeric_dtype(final_data[col]):
                final_data[col] = final_data[col].round(2)
        final_data['Abs_Value'] = final_data['R2'].abs()
        final_data.sort_values(by='Abs_Value', ascending=False, inplace=True)
        final_data.drop(columns=['Abs_Value'], inplace=True)
        final_data.reset_index(drop=True, inplace=True)
        final_data.index = final_data['Column Name']
        final_data.drop(columns=['Column Name'], inplace=True)
        final_data = final_data[['Type', 'Stat', 'Value', 'P-Value', 'Effect Size Type', 'Effect Size Value', 'R2']]
        return final_data

    def bivariate_viz(self, feature, ax=None):
        import seaborn as sns
        import matplotlib.pyplot as plt
        df = self.df.copy()
        feature_is_numeric = pd.api.types.is_numeric_dtype(df[feature])
        if self.label_is_numeric and feature_is_numeric:
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x=feature, y=self.label, ax=ax, color='blue', alpha=0.7)
            ax.set_title(f'Scatter plot of {feature} vs {self.label}', fontsize=16, fontweight='bold')
            ax.set_xlabel(feature, fontsize=14)
            ax.set_ylabel(self.label, fontsize=14)
            if ax is None:
                plt.tight_layout()
                plt.show()

        elif self.label_is_numeric and not feature_is_numeric:
            num_groups = df[feature].nunique()
            if num_groups <= 15:
                ordered_df = df.groupby(feature)[self.label].mean().reset_index()
                ordered_df.sort_values(by=self.label, ascending=False, inplace=True)
                if ax is None:
                    fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=ordered_df, x=feature, y=self.label, ax=ax, palette='viridis')
                ax.set_title(f'Bar plot of {feature} vs average {self.label}', fontsize=16, fontweight='bold')
                ax.set_xlabel(feature, fontsize=14)
                ax.set_ylabel(self.label, fontsize=14)
                if ax is None:
                    plt.tight_layout()
                    plt.show()
            else:
                print(f'Cardinality of {feature} is too high for bar plot.')
        elif not self.label_is_numeric and feature_is_numeric:
            num_groups = df[self.label].nunique()
            if num_groups <= 15:
                ordered_df = df.groupby(self.label)[feature].mean().reset_index()
                ordered_df.sort_values(by=feature, ascending=False, inplace=True)
                if ax is None:
                    fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=ordered_df, x=self.label, y=feature, ax=ax, palette='viridis')
                ax.set_title(f'Bar plot of {self.label} vs average {feature}', fontsize=16, fontweight='bold')
                ax.set_xlabel(self.label, fontsize=14)
                ax.set_ylabel(feature, fontsize=14)
                if ax is None:
                    plt.tight_layout()
                    plt.show()
            else:
                print(f'Cardinality of {self.label} is too high for bar plot.')
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(data=df, x=feature, hue=self.label, ax=ax, palette='viridis')
            ax.set_title(f'Count plot of {feature} vs {self.label}', fontsize=16, fontweight='bold')
            ax.set_xlabel(feature, fontsize=14)
            ax.set_ylabel('Count', fontsize=14)
            ax.legend(title=self.label)
            if ax is None:
                plt.tight_layout()
                plt.show()

    def univariate_stats(self):
        df = self.df.copy()
        new_df = pd.DataFrame(columns=['Count', 'Unique', 'Data Type', 'Missing', 'Mode', 'Min', '25%', 'Median', '75%', 'Max', 'STD Dev', 'Mean', 'Skew', 'Kurt'])
        for col in df:
            if pd.api.types.is_numeric_dtype(df[col]):
                new_df.loc[col] = [df[col].count(), 
                                round(df[col].nunique(), 2), 
                                str(df[col].dtype), 
                                round(df[col].isnull().sum(), 2), 
                                df[col].mode().values[0], 
                                round(df[col].min(), 2), 
                                round(df[col].quantile(.25), 2), 
                                round(df[col].median(), 2), 
                                round(df[col].quantile(.75), 2), 
                                round(df[col].max(), 2), 
                                round(df[col].std(), 2), 
                                round(df[col].mean(), 2), 
                                round(df[col].skew(), 2), 
                                round(df[col].kurt(), 2)]
            else:
                new_df.loc[col] = [df[col].count(), 
                                round(df[col].nunique(), 2), 
                                str(df[col].dtype), 
                                round(df[col].isnull().sum(), 2), 
                                'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA']
        return new_df
    
    def univariate_viz(self, col, axi=None):
        import seaborn as sns
        import matplotlib.pyplot as plt
        df = self.df.copy()
        if axi is None:
            fig, axi = plt.subplots(figsize=(10, 6))
        if col == self.label:
            if self.label_is_numeric == False:
                sns.countplot(x=col, data=df, ax=axi, order=df[col].value_counts().index, palette='viridis')
                axi.set_title(col, fontsize=16, fontweight='bold')
                axi.set_xlabel(col)
                axi.set_ylabel('Count')
                axi.set_xticklabels(axi.get_xticklabels())
            else:
                sns.boxplot(x=df[col], ax=axi, color='white', width=0.5)
                sns.histplot(df[col], kde=True, ax=axi.twinx(), color='blue', bins=30, alpha=0.6)
                axi.set_title(col, fontsize=16, fontweight='bold')
                axi.set_xlabel(col)
                axi.set_ylabel('Count')
        elif pd.api.types.is_numeric_dtype(df[col]):
            sns.boxplot(x=df[col], ax=axi, color='white', width=0.5)
            sns.histplot(df[col], kde=True, ax=axi.twinx(), color='blue', bins=30, alpha=0.6)
            axi.set_title(col, fontsize=16, fontweight='bold')
            axi.set_xlabel(col)
            axi.set_ylabel('Count')
        else:
            sns.countplot(x=col, data=df, ax=axi, order=df[col].value_counts().index, palette='viridis')
            axi.set_title(col, fontsize=16, fontweight='bold')
            axi.set_xlabel(col)
            axi.set_ylabel('Count')
            axi.set_xticklabels(axi.get_xticklabels())
        if axi is None:
            plt.tight_layout()
            plt.show()

    def uni_viz_all(self):
        import matplotlib.pyplot as plt
        import math
        df = self.df.copy()
        num_features = len(df.columns)
        cols = 3
        rows = math.ceil(num_features / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 6 * rows))
        axes = axes.flatten()
        for i, col in enumerate(df.columns):
            self.univariate_viz(col, axi=axes[i])
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.show()

    def biv_viz_all(self):
        import matplotlib.pyplot as plt
        import math
        df = self.df.copy()
        num_features = len(df.columns) - 1
        cols = 3
        rows = math.ceil(num_features / cols) 
        fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 6 * rows))
        axes = axes.flatten()
        for i, col in enumerate(df.columns):
            if col != self.label:
                self.bivariate_viz(col, ax=axes[i])
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.show()