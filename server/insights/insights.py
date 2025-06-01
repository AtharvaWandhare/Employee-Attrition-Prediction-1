# module that finds relationship between different features of the dataset.
"""
Employee Attrition Analysis

This script performs exploratory data analysis on HR-Employee-Attrition dataset
to understand relationships between various features and attrition.
"""

import pandas as pd
import numpy as np # Keep for potential future use or if statistical functions require it
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import sys

# Add parent directory to path to allow imports (consider if truly necessary or adjust data_path)
# This line is often more relevant when importing custom modules from a parent directory.
# For a CSV file, it's generally better to provide a direct or relative path from the script's location.
# For demonstration purposes, I'll keep it as it assumes a specific project structure.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set plot style globally (or integrate into a plotting utility function/class method)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

class AttritionAnalyzer:
    """
    A class to analyze and visualize attrition patterns in HR data.
    """

    def __init__(self, csv_path):
        """Initialize with the path to the CSV data file."""
        try:
            self.data = pd.read_csv(csv_path)
            self._preprocess_data()
        except FileNotFoundError:
            print(f"Error: The file at {csv_path} was not found.")
            sys.exit(1) # Exit if the data file isn't found
        except Exception as e:
            print(f"An error occurred while loading or preprocessing data: {e}")
            sys.exit(1)

    def _preprocess_data(self):
        """Preprocess the data for analysis."""
        # Convert 'Yes'/'No' values in Attrition to 1/0
        self.data['AttritionBinary'] = self.data['Attrition'].apply(
            lambda x: 1 if x == 'Yes' else 0
        )

        # Calculate overall attrition rate
        self.attrition_rate = self.data['AttritionBinary'].mean() * 100
        print(f"Overall Attrition Rate: {self.attrition_rate:.2f}%")

    def summary_statistics(self):
        """Print summary statistics of the dataset."""
        print("\n===== Dataset Summary =====")
        print(f"Total employees: {len(self.data)}")
        print(f"Number who left: {self.data['AttritionBinary'].sum()}")
        print(f"Number who stayed: {len(self.data) - self.data['AttritionBinary'].sum()}")

        print("\n===== Feature Statistics =====")
        # Select a subset of interesting columns for summary
        interesting_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'DistanceFromHome',
                            'JobSatisfaction', 'WorkLifeBalance', 'OverTime']

        print(self.data[interesting_cols].describe())

    def plot_categorical_relationships(self, features):
        """
        Plot relationship between categorical features and attrition.

        Args:
            features (list): List of categorical feature names to analyze
        """
        num_features = len(features)
        fig, axes = plt.subplots(num_features, 1, figsize=(12, 5 * num_features))

        if num_features == 1:
            axes = [axes]

        for i, feature in enumerate(features):
            ax = axes[i]

            # Create the count plot
            sns.countplot(x=feature, hue='Attrition', data=self.data, ax=ax, palette="Set2")

            # Add percentage annotations based on each category
            # This is the corrected logic for annotations
            for container in ax.containers:
                for j, p in enumerate(container.patches):
                    height = p.get_height()
                    if height > 0:
                        # Get the total count for the current category (e.g., 'Male', 'Female')
                        # and the specific hue (e.g., 'Yes', 'No')
                        category_total = self.data[self.data[feature] == p.get_x()][feature].count()
                        # Get the count for the current category and hue
                        count_for_hue = height
                        percentage = (count_for_hue / category_total) * 100 if category_total > 0 else 0

                        # Adjust text placement to avoid overlap
                        ax.text(
                            p.get_x() + p.get_width() / 2.,
                            height, # Position text at the top of the bar
                            f'{percentage:.1f}%',
                            ha='center',
                            va='bottom', # Align text to the bottom of the height for better spacing
                            fontsize=8,
                            color='black'
                        )

            ax.set_title(f'Attrition by {feature}', fontsize=14)
            ax.set_xlabel(feature, fontsize=12)
            ax.set_ylabel('Count', fontsize=12)

            # Adjust x-labels for better readability if too many categories
            if len(self.data[feature].unique()) > 5:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            # Add chi-square test
            contingency_table = pd.crosstab(self.data[feature], self.data['Attrition'])
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

            significance = "significant" if p < 0.05 else "not significant"
            ax.text(0.05, 0.95, f'Chi-square test: p = {p:.4f} ({significance})',
                    transform=ax.transAxes, ha='left', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(f'categorical_relationships.png')
        plt.close()

    def plot_numerical_relationships(self, features):
        """
        Plot relationship between numerical features and attrition.

        Args:
            features (list): List of numerical feature names to analyze
        """
        num_features = len(features)
        fig, axes = plt.subplots(num_features, 2, figsize=(16, 4 * num_features))

        if num_features == 1:
            axes = axes.reshape(1, 2)

        for i, feature in enumerate(features):
            # Box plot
            sns.boxplot(x='Attrition', y=feature, data=self.data, ax=axes[i, 0], palette="Set2")
            axes[i, 0].set_title(f'Attrition by {feature} (Box Plot)', fontsize=14)
            axes[i, 0].set_xlabel('Attrition', fontsize=12)
            axes[i, 0].set_ylabel(feature, fontsize=12)

            # Histogram
            for attrition, group in self.data.groupby('Attrition'):
                sns.histplot(group[feature], kde=True, ax=axes[i, 1],
                             label=attrition, alpha=0.6, stat='density', common_norm=False) # Use stat='density' for comparison
            axes[i, 1].set_title(f'Distribution of {feature} by Attrition', fontsize=14)
            axes[i, 1].set_xlabel(feature, fontsize=12)
            axes[i, 1].set_ylabel('Density', fontsize=12)
            axes[i, 1].legend(title='Attrition')

            # Add t-test or Mann-Whitney U test
            attrition_yes = self.data[self.data['Attrition'] == 'Yes'][feature].dropna()
            attrition_no = self.data[self.data['Attrition'] == 'No'][feature].dropna()

            # Ensure there's enough data for statistical tests
            if len(attrition_yes) > 1 and len(attrition_no) > 1:
                # Normality test (Shapiro-Wilk for smaller samples, otherwise rely on CLT or visual inspection)
                # For very large samples, Shapiro-Wilk can be computationally expensive and overly sensitive.
                # Here, we'll proceed with the assumption of CLT for large samples or use non-parametric.
                # A more robust check for normality in large datasets might involve visual inspection or D'Agostino's K^2 test.
                
                # Check for normality for both groups (if sample size is reasonable for Shapiro-Wilk)
                is_normal_yes = False
                if len(attrition_yes) > 3: # Shapiro-Wilk requires at least 3 data points
                    try:
                        _, p_shapiro_yes = stats.shapiro(attrition_yes)
                        is_normal_yes = p_shapiro_yes > 0.05
                    except ValueError: # Handle cases where data is constant or not enough points for Shapiro-Wilk
                        is_normal_yes = False 
                
                is_normal_no = False
                if len(attrition_no) > 3: # Shapiro-Wilk requires at least 3 data points
                    try:
                        _, p_shapiro_no = stats.shapiro(attrition_no)
                        is_normal_no = p_shapiro_no > 0.05
                    except ValueError: # Handle cases where data is constant or not enough points for Shapiro-Wilk
                        is_normal_no = False

                if is_normal_yes and is_normal_no:
                    stat, p = stats.ttest_ind(attrition_yes, attrition_no)
                    test_name = "Independent T-test"
                else:
                    stat, p = stats.mannwhitneyu(attrition_yes, attrition_no)
                    test_name = "Mann-Whitney U test"

                significance = "significant" if p < 0.05 else "not significant"
                axes[i, 0].text(0.05, 0.95, f'{test_name}: p = {p:.4f} ({significance})',
                                transform=axes[i, 0].transAxes, ha='left', fontsize=10,
                                bbox=dict(facecolor='white', alpha=0.8))
            else:
                axes[i, 0].text(0.05, 0.95, 'Not enough data for statistical test',
                                transform=axes[i, 0].transAxes, ha='left', fontsize=10,
                                bbox=dict(facecolor='white', alpha=0.8))


        plt.tight_layout()
        plt.savefig(f'numerical_relationships.png')
        plt.close()

    def plot_correlation_matrix(self):
        """Plot correlation matrix for numerical features."""
        # Select numerical columns only
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()

        # Create correlation matrix
        corr_matrix = self.data[numerical_cols].corr()

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Create heatmap
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, linewidths=0.5, cbar_kws={"shrink": .8})

        plt.title('Correlation Matrix of Numerical Features', fontsize=16)
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        plt.close()

    def plot_feature_importance(self):
        """Plot feature importance using a simple model."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split # Added for better practice

        # Prepare the data
        # Ensure 'Attrition' is not considered a categorical feature for encoding if it's the target
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        if 'Attrition' in categorical_cols:
            categorical_cols.remove('Attrition')  # Remove target variable if it's in object type

        # Encode categorical variables
        df_encoded = self.data.copy()
        # label_encoders = {} # Not used, so removed

        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(self.data[col])
            # label_encoders[col] = le # No need to store if not used later

        # Features and target
        # Ensure 'AttritionBinary' is not in features
        features_to_drop = ['Attrition', 'AttritionBinary']
        # Filter out columns that might not exist after encoding or selection
        X = df_encoded.drop(columns=[col for col in features_to_drop if col in df_encoded.columns], axis=1)
        y = df_encoded['AttritionBinary']
        
        # It's good practice to split data into training and testing sets,
        # but for simple feature importance plotting, it might be skipped to use all data.
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a simple random forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y) # Fit on full data for feature importance

        # Plot feature importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance for Attrition Prediction', fontsize=16)

        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

        # Print top 10 features
        top_features = [(X.columns[i], importances[i]) for i in indices[:10]]
        print("\n===== Top 10 Features for Attrition Prediction =====")
        for feature, importance in top_features:
            print(f"{feature}: {importance:.4f}")

    def analyze_specific_feature(self, feature, is_categorical=False):
        """
        Perform detailed analysis of a specific feature.

        Args:
            feature (str): Feature name to analyze
            is_categorical (bool): Whether the feature is categorical
        """
        print(f"\n===== Detailed Analysis of {feature} =====")

        # Create figure
        plt.figure(figsize=(14, 8))

        if is_categorical:
            # Group by feature and calculate attrition rate
            grouped = self.data.groupby(feature)['AttritionBinary'].agg(['count', 'sum', 'mean'])
            grouped['attrition_rate'] = grouped['mean'] * 100

            # Sort by attrition rate for better visualization
            grouped = grouped.sort_values('attrition_rate', ascending=False)

            # Bar plot of attrition rate by category
            plt.subplot(1, 2, 1)
            bars = plt.bar(grouped.index, grouped['attrition_rate'], color=sns.color_palette("Set2")[0])
            plt.title(f'Attrition Rate by {feature}', fontsize=14)
            plt.xlabel(feature, fontsize=12)
            plt.ylabel('Attrition Rate (%)', fontsize=12)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                         f'{height:.1f}%', ha='center', fontsize=9)

            # Adjust x-labels for better readability if too many categories
            if len(grouped.index) > 5:
                plt.xticks(rotation=45, ha='right')

            # Add count information as a table
            plt.subplot(1, 2, 2)
            plt.axis('off')  # Turn off axis

            # Create table data
            table_data = []
            for category in grouped.index:
                count = int(grouped.loc[category, 'count'])
                left = int(grouped.loc[category, 'sum'])
                rate = grouped.loc[category, 'attrition_rate']
                table_data.append([category, count, left, f"{rate:.1f}%"])

            table = plt.table(cellText=table_data,
                              colLabels=[feature, 'Count', 'Left', 'Attrition Rate'],
                              loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)

            # Print chi-square test results
            contingency_table = pd.crosstab(self.data[feature], self.data['Attrition'])
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            print(f"Chi-square test: chi2 = {chi2:.4f}, p = {p:.4f}")
            plt.text(0.5, 0.05, f'Chi-square test: p = {p:.4f}', transform=plt.gca().transAxes,
                     ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))


        else:  # Numerical feature
            # Calculate statistics by attrition group
            stats_by_group = self.data.groupby('Attrition')[feature].describe()
            print(stats_by_group)

            # Create figure with three subplots
            plt.subplot(1, 3, 1)
            sns.boxplot(x='Attrition', y=feature, data=self.data, palette="Set2")
            plt.title(f'{feature} by Attrition Status', fontsize=14)

            plt.subplot(1, 3, 2)
            for attrition, group in self.data.groupby('Attrition'):
                sns.histplot(group[feature], kde=True, label=attrition, alpha=0.6, stat='density', common_norm=False)
            plt.title(f'Distribution of {feature} by Attrition', fontsize=14)
            plt.legend(title='Attrition')

            # Add a scatter plot with jittering for visualization
            plt.subplot(1, 3, 3)
            sns.stripplot(x='Attrition', y=feature, data=self.data,
                          jitter=0.2, alpha=0.5, palette="Set2") # Added jitter amount for consistency
            plt.title(f'Individual Data Points of {feature}', fontsize=14)

            # Perform statistical test
            attrition_yes = self.data[self.data['Attrition'] == 'Yes'][feature].dropna()
            attrition_no = self.data[self.data['Attrition'] == 'No'][feature].dropna()

            # Ensure there's enough data for statistical tests
            if len(attrition_yes) > 1 and len(attrition_no) > 1:
                # Normality test
                # Shapiro-Wilk test for samples up to 5000. For larger, consider other tests or rely on CLT.
                p_yes = 0
                if len(attrition_yes) > 3 and len(attrition_yes) <= 5000:
                    try:
                        _, p_yes = stats.shapiro(attrition_yes)
                    except ValueError: # Handle constant data
                        pass
                
                p_no = 0
                if len(attrition_no) > 3 and len(attrition_no) <= 5000:
                    try:
                        _, p_no = stats.shapiro(attrition_no)
                    except ValueError: # Handle constant data
                        pass

                # Choose appropriate test based on normality (p-value > 0.05 for normality)
                if p_yes > 0.05 and p_no > 0.05:  # Both normally distributed
                    stat, p = stats.ttest_ind(attrition_yes, attrition_no)
                    test_name = "Independent T-test"
                else:  # Non-parametric test
                    stat, p = stats.mannwhitneyu(attrition_yes, attrition_no)
                    test_name = "Mann-Whitney U test"

                print(f"{test_name}: statistic = {stat:.4f}, p = {p:.4f}")
                plt.text(0.5, 0.95, f'{test_name}: p = {p:.4f}', transform=plt.gca().transAxes,
                         ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            else:
                print("Not enough data to perform statistical test.")
                plt.text(0.5, 0.95, 'Not enough data for statistical test', transform=plt.gca().transAxes,
                         ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))


        plt.tight_layout()
        plt.savefig(f'{feature}_analysis.png')
        plt.close()

    def run_complete_analysis(self):
        """Run a complete analysis on the dataset."""
        # Print basic statistics
        self.summary_statistics()

        # Analyze categorical features
        categorical_features = ['Department', 'OverTime', 'MaritalStatus', 'JobRole',
                                'Gender', 'EducationField', 'BusinessTravel']
        self.plot_categorical_relationships(categorical_features)

        # Analyze numerical features
        numerical_features = ['Age', 'MonthlyIncome', 'DistanceFromHome',
                              'YearsAtCompany', 'TotalWorkingYears', 'JobLevel']
        self.plot_numerical_relationships(numerical_features)

        # Plot correlation matrix
        self.plot_correlation_matrix()

        # Calculate and plot feature importance
        self.plot_feature_importance()

        # Detail analysis of specific important features
        self.analyze_specific_feature('OverTime', is_categorical=True)
        self.analyze_specific_feature('Age', is_categorical=False)
        self.analyze_specific_feature('MonthlyIncome', is_categorical=False)
        self.analyze_specific_feature('JobRole', is_categorical=True)
        self.analyze_specific_feature('YearsAtCompany', is_categorical=False)

        print("\nAnalysis complete! Visualization images saved.")

if __name__ == "__main__":
    # Path to the dataset
    # Consider adjusting this path to be relative to the script's location
    # or providing a full path as a command-line argument for flexibility.
    data_path = "../data/HR-Employee-Attrition.csv"

    # Create analyzer instance
    analyzer = AttritionAnalyzer(data_path)

    # Run complete analysis
    analyzer.run_complete_analysis()