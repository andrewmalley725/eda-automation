class EDA:
    def __init__(self):
        pass

    def bivariate_stats(self, data, label):
        from scipy import stats
        import pandas as pd
        import numpy as np
        from scipy.stats import chi2_contingency
        final_data = []
        df = data.copy()
        label_is_numeric = pd.api.types.is_numeric_dtype(df[label])
        for col in df.drop(label, axis=1).columns:
            record = {
                'Column Name': col
            }
            if label_is_numeric:
                if pd.api.types.is_numeric_dtype(df[col]):
                    record['Type'] = 'Numeric'
                    r, p = stats.pearsonr(df[col], df[label])
                    r2 = r ** 2
                    record['Stat'] = 'R'
                    record['Value'] = r
                    record['P-Value'] = p
                    record['R-Squared'] = r2
                    final_data.append(record)
                else:
                    record['Type'] = 'Categorical'
                    groups = df[col].nunique()
                    if groups > 2:
                        record['Stat'] = 'F'
                        f, p = stats.f_oneway(*[df[label][df[col] == group] for group in df[col].unique()])
                        record['Value'] = f
                        record['P-Value'] = p
                        record['R-Squared'] = np.nan
                        final_data.append(record)
                    elif groups == 2:
                        record['Stat'] = 'T'
                        values = df[col].unique()
                        g1 = values[0]
                        g2 = values[1]
                        t, p = stats.ttest_ind(df[label][df[col] == g1], df[label][df[col] == g2])
                        record['Value'] = t
                        record['P-Value'] = p
                        record['R-Squared'] = np.nan
                        final_data.append(record)
            else:
                label_groups = df[label].nunique()
                if pd.api.types.is_numeric_dtype(df[col]):
                    record['Type'] = 'Numeric'
                    if label_groups > 2:
                        record['Stat'] = 'F'
                        f, p = stats.f_oneway(*[df[col][df[label] == group] for group in df[label].unique()])
                        record['Value'] = f
                        record['P-Value'] = p
                        record['R-Squared'] = np.nan
                        final_data.append(record)
                    elif label_groups == 2:
                        record['Stat'] = 'T'
                        values = df[label].unique()
                        g1 = values[0]
                        g2 = values[1]
                        t, p = stats.ttest_ind(df[col][df[label] == g1], df[col][df[label] == g2])
                        record['Value'] = t
                        record['P-Value'] = p
                        record['R-Squared'] = np.nan
                        final_data.append(record)
                else:
                    record['Type'] = 'Categorical'
                    record['Stat'] = 'CHI2'
                    contingency_table = pd.crosstab(df[col], df[label])
                    chi2, p, _, _ = chi2_contingency(contingency_table)
                    record['Value'] = chi2
                    record['P-Value'] = p
                    record['R-Squared'] = np.nan
                    final_data.append(record)
        final_data = pd.DataFrame(final_data)
        for col in final_data.columns:
            if pd.api.types.is_numeric_dtype(final_data[col]):
                final_data[col] = final_data[col].round(2)
        final_data['Abs_Value'] = final_data['Value'].abs()
        final_data.sort_values(by='Abs_Value', ascending=False, inplace=True)
        final_data.drop(columns=['Abs_Value'], inplace=True)
        return final_data

    def bivariate_viz(self, data, feature, label, ax=None):
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        df = data.copy()
        label_is_numeric = pd.api.types.is_numeric_dtype(df[label])
        feature_is_numeric = pd.api.types.is_numeric_dtype(df[feature])
        if label_is_numeric and feature_is_numeric:
            if ax is None:
                plt.title(f'Scatter plot of {feature} vs {label}', fontsize=16, fontweight='bold')
            else:
                ax.set_title(f'Scatter plot of {feature} vs {label}', fontsize=16, fontweight='bold')
            plt.xlabel(feature)
            plt.ylabel(label)
            sns.scatterplot(data=df, x=feature, y=label, ax=ax)
            if ax is None:
                plt.show()

        elif label_is_numeric and not feature_is_numeric:
            num_groups = df[feature].nunique()
            if num_groups <= 15:
                ordered_df = df.groupby(feature)[label].mean().reset_index()
                ordered_df.sort_values(by=label, ascending=False, inplace=True)
                if ax is None:
                    plt.title(f'Bar plot of {feature} vs average {label}', fontsize=16, fontweight='bold')
                else:
                    ax.set_title(f'Bar plot of {feature} vs average {label}', fontsize=16, fontweight='bold')
                plt.xlabel(feature)
                plt.ylabel(label)
                sns.barplot(data=ordered_df, x=feature, y=label, hue=label, ax=ax)
                if ax is None:
                    plt.show()
            else:
                print(f'Cardinality of {feature} is too high for bar plot.')
        elif not label_is_numeric and feature_is_numeric:
            num_groups = df[label].nunique()
            if num_groups <= 15:
                ordered_df = df.groupby(label)[feature].mean().reset_index()
                ordered_df.sort_values(by=feature, ascending=False, inplace=True)
                if ax is None:
                    plt.title(f'Bar plot of {label} vs average {feature}', fontsize=16, fontweight='bold')
                else:
                    ax.set_title(f'Bar plot of {label} vs average {feature}', fontsize=16, fontweight='bold')
                plt.xlabel(label)
                plt.ylabel(feature)
                sns.barplot(data=ordered_df, x=label, y=feature, hue=feature, ax=ax)
                if ax is None:
                    plt.show()
            else:
                print(f'Cardinality of {label} is too high for bar plot.')
        else:
            if ax is None:
                plt.title(f'Count plot of {feature} vs {label}', fontsize=16, fontweight='bold')
            else:
                ax.set_title(f'Count plot of {feature} vs {label}', fontsize=16, fontweight='bold')
            plt.xlabel(feature)
            plt.ylabel('Count')
            plt.legend(title=label)
            sns.countplot(data=df, x=feature, hue=label, ax=ax)
            if ax is None:
                plt.show()

    def univariate_stats(self, df):
        import pandas as pd
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
    
    def univariate_viz(self, df, col, axi=None):
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        if axi is None:
            fig, axi = plt.subplots(figsize=(10, 6))
        if pd.api.types.is_numeric_dtype(df[col]):
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

    def uni_viz_all(self, df):
        import matplotlib.pyplot as plt
        import math
        num_features = len(df.columns)
        cols = 3
        rows = math.ceil(num_features / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 6 * rows))
        axes = axes.flatten()
        for i, col in enumerate(df.columns):
            self.univariate_viz(df, col, axi=axes[i])
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.show()

    def biv_viz_all(self, df, label):
        import matplotlib.pyplot as plt
        import math
        num_features = len(df.columns) - 1
        cols = 3
        rows = math.ceil(num_features / cols) 
        fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 6 * rows))
        axes = axes.flatten()
        for i, col in enumerate(df.columns):
            if col != label:
                self.bivariate_viz(df, col, label, ax=axes[i])
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.show()