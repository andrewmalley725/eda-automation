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


    def univariate_viz(self, df):
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        new_df = pd.DataFrame(columns=['Count', 'Unique', 'Data Type', 'Missing', 'Mode', 'Min', '25%', 'Median', '75%', 'Max', 'STD Dev', 'Mean', 'Skew', 'Kurt'])
        for col in df:
            if pd.api.types.is_numeric_dtype(df[col]):
                f, (ax_box, ax) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (.15, .65)})
                sns.set(style='ticks')
                flierprops = dict(marker='o', markersize=4, markerfacecolor='none', linestyle='none', markeredgecolor='gray')
                sns.boxplot(x=df[col], ax=ax_box, fliersize=4, width=.50, linewidth=1, flierprops=flierprops)
                sns.histplot(df, x=df[col])
                sns.despine(ax=ax)
                sns.despine(ax=ax_box, left=True, bottom=True)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                ax_box.set_title(col, fontsize=14)
                new_df.loc[col] = [df[col].count(), round(df[col].nunique(), 2), str(df[col].dtype), round(df[col].isnull().sum(), 2), df[col].mode().values[0], round(df[col].min(), 2), round(df[col].quantile(.25), 2), round(df[col].median(), 2), round(df[col].quantile(.75), 2), round(df[col].max(), 2), round(df[col].std(), 2), round(df[col].mean(), 2), round(df[col].skew(), 2), round(df[col].kurt(), 2)]
                text = 'Count: ' + str(df[col].count()) + '\n'
                text += 'Unique: ' + str(round(df[col].nunique(), 2)) + '\n'
                text += 'Data Type: ' + str(df[col].dtype) + '\n'
                text += 'Missing: ' + str(round(df[col].isnull().sum(), 2)) + '\n'
                text += 'Mode: ' + str(df[col].mode().values[0]) + '\n'
                text += 'Min: ' + str(round(df[col].min(), 2)) + '\n'
                text += '25%: ' + str(round(df[col].quantile(.25), 2)) + '\n'
                text += 'Median: ' + str(round(df[col].median(), 2)) + '\n'
                text += '75%: ' + str(round(df[col].quantile(.75), 2)) + '\n'
                text += 'Max: ' + str(round(df[col].max(), 2)) + '\n'
                text += 'Std Dev: ' + str(round(df[col].std(), 2)) + '\n'
                text += 'Mean: ' + str(round(df[col].mean(), 2)) + '\n'
                text += 'Skew: ' + str(round(df[col].skew(), 2)) + '\n'
                text += 'Kurt: ' + str(round(df[col].kurt(), 2)) + '\n'
                ax.text(.9, .25, text, fontsize=10, transform=plt.gcf().transFigure)
                plt.show()
            else:
                ax_count = sns.countplot(x=col, data=df, order=df[col].value_counts().index, palette=sns.color_palette('RdBu_r', df[col].nunique()))
                sns.despine(ax=ax_count)
                ax_count.set_title(col)
                ax_count.set_xlabel(col)
                ax_count.set_ylabel('')
                ax_count.set_xticklabels(ax_count.get_xticklabels(), rotation=45)
                new_df.loc[col] = [df[col].count(), round(df[col].nunique(), 2), str(df[col].dtype), round(df[col].isnull().sum(), 2), 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA']
                text = 'Count: ' + str(df[col].count()) + '\n'
                text += 'Unique: ' + str(round(df[col].nunique(), 2)) + '\n'
                text += 'Data Type: ' + str(df[col].dtype) + '\n'
                text += 'Missing: ' + str(round(df[col].isna().sum(), 2)) + '\n'
                ax_count.text(.9, .5, text, fontsize=10, transform=plt.gcf().transFigure)
                plt.show()
        return new_df

    def bivariate_viz(self, data, feature, label):
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        df = data.copy()
        label_is_numeric = pd.api.types.is_numeric_dtype(df[label])
        feature_is_numeric = pd.api.types.is_numeric_dtype(df[feature])
        if label_is_numeric and feature_is_numeric:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x=feature, y=label)
            plt.title(f'Scatter plot of {feature} vs {label}')
            plt.xlabel(feature)
            plt.ylabel(label)
            plt.show()
        elif label_is_numeric and not feature_is_numeric:
            num_groups = df[feature].nunique()
            if num_groups <= 15:
                ordered_df = df.groupby(feature)[label].mean().reset_index()
                ordered_df.sort_values(by=label, ascending=False, inplace=True)
                plt.figure(figsize=(10, 6))
                sns.barplot(data=ordered_df, x=feature, y=label, hue=label)
                plt.title(f'Bar plot of {feature} vs Average {label}')
                plt.xticks(rotation=45)
                plt.xlabel(feature)
                plt.ylabel(label)
                plt.show()
            else:
                print(f'Cardinality of {feature} is too high for bar plot.')
        elif not label_is_numeric and feature_is_numeric:
            num_groups = df[label].nunique()
            if num_groups <= 15:
                ordered_df = df.groupby(label)[feature].mean().reset_index()
                ordered_df.sort_values(by=feature, ascending=False, inplace=True)
                plt.figure(figsize=(10, 6))
                sns.barplot(data=ordered_df, x=label, y=feature, hue=label)
                plt.title(f'Bar plot of {label} vs Average {feature}')
                plt.xticks(rotation=45)
                plt.xlabel(label)
                plt.ylabel(feature)
                plt.show()
            else:
                print(f'Cardinality of {label} is too high for bar plot.')
        else:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df, x=feature, hue=label)
            plt.title(f'Count plot of {feature} vs {label}')
            plt.xlabel(feature)
            plt.ylabel('Count')
            plt.legend(title=label)
            plt.show()