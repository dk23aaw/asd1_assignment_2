import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_worldbank_data(filename):
    """
    Read data from a CSV file and perform necessary preprocessing.

    Parameters:
    - filename (str): The path to the CSV file.

    Returns:
    - df_years (pd.DataFrame): DataFrame with years as columns.
    - df_countries (pd.DataFrame): DataFrame with countries as columns.
    """

    # Step 1: Read data from the CSV file, skipping the first 4 rows.
    df = pd.read_csv(filename, skiprows=4)

    # Step 2: Drop unnecessary columns.
    cols_to_drop = ['Country Code', 'Indicator Code', 'Unnamed: 66']
    df = df.drop(cols_to_drop, axis=1)

    # Step 3: Rename remaining columns for clarity.
    df = df.rename(columns={'Country Name': 'Country'})

    # Step 4: Melt the dataframe to convert years to a single column.
    df = df.melt(id_vars=['Country', 'Indicator Name'],
                 var_name='Year', value_name='Value')

    # Step 5: Convert year column to integer and value column to float.
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # Step 6: Separate dataframes with years and countries as columns.
    df_years = df.pivot_table(
        index=['Country', 'Indicator Name'], columns='Year', values='Value')
    df_countries = df.pivot_table(
        index=['Year', 'Indicator Name'], columns='Country', values='Value')

    # Step 7: Clean the data by removing columns with all NaN values.
    df_years = df_years.dropna(how='all', axis=1)
    df_countries = df_countries.dropna(how='all', axis=1)

    # Step 8: Return the preprocessed dataframes.
    return df_years, df_countries




def filter_and_transpose_data(df_years, countries, indicators, start_year, end_year):
    """
    Subsets the data to include only the selected countries, indicators, and specified year range.
    Returns the subsetted data as a new DataFrame.
    """
    # Create a boolean mask for the specified year range
    mask_year = (df_years.columns.get_level_values('Year').astype(int) >= start_year) & (df_years.columns.get_level_values('Year').astype(int) <= end_year)

    # Apply masks to subset the data
    df = df_years.loc[(countries, indicators), mask_year].transpose()

    return df


def calculate_correlations(df):
    """
    Calculates the correlations between the indicators in the input DataFrame.
    Returns the correlation matrix as a new DataFrame.
    """
    corr = df.corr()
    return corr



def plot_selected_years_line_chart(df_years, selected_years, indicator):
    """
    Plot a line chart for the specified indicator for selected years and countries.

    Parameters:
    - df_years (pd.DataFrame): DataFrame with years as columns.
    - selected_years (list): List of years to be plotted.
    - indicator (str): The indicator to plot.

    Returns:
    - None
    """

    # List of countries to include in the plot
    country_list = ['Bangladesh', 'India', 'Germany', 'China', 'United Kingdom']

    # Set Seaborn style with a dark background
    sns.set(style="darkgrid")

    # Use a specific dark color
    dark_color = '#3498db'  # You can replace this with any dark color of your choice

    # Set the line styles
    line_styles = ['-', '--', '-.', ':']

    # Set the figure size
    fig, ax = plt.subplots(figsize=(16, 10))

    # Iterate over each country and plot the line chart
    for i, country in enumerate(country_list):
        df_subset = df_years.loc[(country, indicator), selected_years]
        ax.plot(df_subset.index, df_subset.values, label=country, linestyle=line_styles[i % len(line_styles)], color=dark_color)
        # Print results for the selected years
        print(f"\nResults for {country} - {indicator} for the selected years:")
        print(df_subset)


    # Set labels and title with increased text size
    ax.set_xlabel('Year', fontsize=18)
    ax.set_ylabel(indicator, fontsize=18)
    ax.set_title(f'{indicator} form 2000-2010 Years', fontsize=20)

    # Increase legend font size
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')

    # Add a grid to the plot
    ax.grid(True, linestyle='--', alpha=0.6)

    # Display the plot
    plt.show()


def plot_custom_bar_chart(df_years, indicator):
    """
    Plot a bar chart for a specific indicator over the years and countries.

    Parameters:
    - df_years (pd.DataFrame): DataFrame with years as columns.
    - indicator (str): The indicator to plot.

    Returns:
    - None
    """

    # List of countries to include in the plot
    country_list = ['Bangladesh', 'India', 'Germany', 'China', 'United Kingdom']
    
    # List of years for which the bar chart will be plotted
    years = [2000,2001,2002,2003,2004, 2005,2006,2008,2009,2010]
    
    # Generate x-axis positions
    x = np.arange(len(country_list))
    
    # Width of each bar
    width = 0.35

    # Set Seaborn style with a dark background
    sns.set(style="darkgrid")

    # Set a dark color palette
    dark_palette = sns.color_palette("dark:#69d", n_colors=len(years))

    # Create a subplot with specified size
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Iterate over each year and plot a bar for each country
    for i, year in enumerate(years):
        indicator_values = []
        
        # Iterate over each country to get the indicator value for the specified year
        for country in country_list:
            value = df_years.loc[(country, indicator), year]
            indicator_values.append(value)

            # Print the data used for each bar (optional)
            print(f"{country} - {indicator} ({year}): {value}")

        # Plot a bar for each country with the specified color
        rects1 = ax.bar(x - width/2 + i*width/len(years), indicator_values,
                        width/len(years), label=str(year)+" ", color=dark_palette[i])

    # Set labels and title with increased text size
    ax.set_xlabel('Country', fontsize=16)
    ax.set_ylabel('Value', fontsize=16)
    ax.set_title(f'{indicator} over the years', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(country_list, fontsize=14)
    ax.legend(fontsize='large')

    # Adjust layout and display the plot
    fig.tight_layout()
    plt.show()


def plot_custom_pie_chart(df_years, selected_year, custom_indicator, country_list=None, color_palette='husl'):
    """
    Plot a pie chart for a custom indicator for a selected year.

    Parameters:
    - df_years (pd.DataFrame): DataFrame with years as columns.
    - selected_year (int): The specific year to plot.
    - custom_indicator (str): The custom indicator to plot.
    - country_list (list): List of countries to include in the plot.
    - color_palette (str): Seaborn color palette name.

    Returns:
    - None
    """

    # If country_list is not provided, use the default list
    if country_list is None:
       country_list = ['Bangladesh', 'India','Germany', 'China','United Kingdom']

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Set the color palette
    colors = sns.color_palette(color_palette, n_colors=len(country_list))

    # Create a subplot with specified size
    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract values for the selected year
    values = [df_years.loc[(country, custom_indicator), selected_year] for country in country_list]
    
    # Filter out NaN values and corresponding countries
    valid_data = [(country, value) for country, value in zip(country_list, values) if not pd.isna(value)]
    valid_countries, valid_values = zip(*valid_data)

    # Check if there are valid data points
    if not valid_data:
        raise ValueError("No valid data points for the specified custom indicator and year.")

    # Plot the pie chart with valid data
    ax.pie(valid_values, labels=valid_countries, autopct='%1.1f%%', colors=colors, startangle=90)

    # Equal aspect ratio ensures the pie chart is circular.
    ax.axis('equal')

    # Set title with increased text size
    plt.title(f'{custom_indicator} Distribution for {selected_year}', fontsize=16)
    
    # Display the pie chart
    plt.show()

    # Print the results to the console
    print(f"\nResults for {custom_indicator} for the selected year {selected_year}:")
    for country, value in valid_data:
        print(f"{country}: {value:.2f}%")



def explore_indicators(df_years, countries, indicators):
    """
    Explore statistical properties of indicators for individual countries and cross-compare.

    Parameters:
    - df_years (pandas.DataFrame): DataFrame containing the yearly data for indicators.
    - countries (list of str): List of countries for analysis.
    - indicators (list of str): List of indicators to explore.
    """
    # Create a dictionary to store summary statistics
    summary_stats = {}
   
    # Explore indicators for individual countries
    for country in countries:
        for indicator in indicators:
            # Get data for the specific country and indicator
            data = df_years.loc[(country, indicator), :]

            # Calculate summary statistics using .describe() and two additional statistical methods
            stats = {
                'describe': data.describe(),
                'median': data.median(),
                'std_dev': data.std(),
            }

            # Store the statistics in the dictionary
            summary_stats[f'{country} - {indicator}'] = stats

    # Explore indicators for aggregated regions or categories
    for indicator in indicators:
        # Get data for the world (you can modify this for other regions/categories)
        data_world = df_years.loc[('World', indicator), :]

        # Calculate summary statistics using .describe() and two additional statistical methods
        stats_world = {
            'describe': data_world.describe(),
            'median': data_world.median(),
            'std_dev': data_world.std(),
        }

        # Store the statistics in the dictionary
        summary_stats[f'World - {indicator}'] = stats_world

    # Print the summary statistics
    for key, stats in summary_stats.items():
        print(f"Summary Statistics for {key}:")
        print(stats['describe'])
        print(f"Median: {stats['median']}")
        print(f"Standard Deviation: {stats['std_dev']}")
        print("\n" + "=" * 50 + "\n")

def explore_correlations(df_years, countries, indicators, start_year, end_year):
    """
    Explore and understand correlations between indicators within countries and across time.

    Parameters:
    - df_years (pandas.DataFrame): DataFrame containing the yearly data for indicators.
    - countries (list of str): List of countries for analysis.
    - indicators (list of str): List of indicators to explore.
    - start_year (int): Start year for the analysis.
    - end_year (int): End year for the analysis.
    """
    # Subset the data for the specified year range
    df_filtered = filter_and_transpose_data(df_years, countries, indicators, start_year, end_year)

    # Calculate correlations
    corr_matrix = calculate_correlations(df_filtered)

    # Visualize correlations as a heatmap
    plot_correlation_heatmap(corr_matrix)

def plot_correlation_heatmap(corr):
    """
    Plots the correlation matrix as a heatmap using Seaborn with improved styles.

    Parameters:
    - corr (pandas.DataFrame): Correlation matrix.
    """
    # Set Seaborn style
    sns.set_theme(style="whitegrid")

    # Set the color palette
    cmap = sns.color_palette("twilight")

    # Set the font scale and size
    sns.set(font_scale=1.4)
    
    # Set the figure size
    plt.figure(figsize=(12, 10))

    # Plot the heatmap
    sns.heatmap(corr, cmap=cmap, annot=True, square=True, linewidths=.5, cbar_kws={"shrink": 0.75})

    # Set title and show the plot
    plt.title('Correlation Matrix of Indicators', fontsize=16)
    plt.show()

    
def main():
    # Read World Bank data from the specified file
    df_years, df_countries = load_worldbank_data(r"C:\Users\)

    # Define indicators and countries for heatmap analysis
    indicators_heatmap1 = ['Urban population (% of total population)', 'Urban population growth (annual %)']
    countries_list = ['Bangladesh', 'India','Germany', 'China','United Kingdom']
    start_year = 2000
    end_year = 2010

    # Explore correlations for the first set of indicators
   # explore_correlations(df_years, countries_list, indicators_heatmap1, start_year, end_year)

    # Explore correlations for the second set of indicators
    indicators_heatmap2 = ['Population, total', 'Population growth (annual %)']
   # explore_correlations(df_years, countries_list, indicators_heatmap2, start_year, end_year)

    # Explore individual indicators for the first set
    #explore_indicators(df_years, countries_list, indicators_heatmap1)

    # Explore individual indicators for the second set
    #explore_indicators(df_years, countries_list, indicators_heatmap2)
 # Plot bar charts for urban population growth indicators
    selected_indicator_urban_population_growth = 'Electricity production from hydroelectric sources (% of total)'
   # plot_custom_bar_chart(df_years, selected_indicator_urban_population_growth)

    selected_indicator_urban_population_growth = 'Electricity production from coal sources (% of total)'
   # plot_custom_bar_chart(df_years, selected_indicator_urban_population_growth)
    
    # Plot pie charts for CO2 emissions in selected years
    selected_year_pie = 2000
    selected_indicator_pie = 'Access to electricity (% of population)'
    color_palette = 'Set2'
   # plot_custom_pie_chart(df_years, selected_year_pie, selected_indicator_pie, color_palette=color_palette)

    selected_year_pie = 2015
    color_palette = 'deep'
   # plot_custom_pie_chart(df_years, selected_year_pie, selected_indicator_pie, color_palette=color_palette)

    # Define selected years for line charts
    selected_years = [2000,2001,2002,2003,2004, 2005,2006,2008,2009,2010]

    # Plot line chart for the first selected indicator
    line_selected_indicator1 = 'Renewable energy consumption (% of total final energy consumption)'
    plot_selected_years_line_chart(df_years, selected_years, line_selected_indicator1)

    # Plot line chart for the second selected indicator
    line_selected_indicator2 = 'Renewable electricity output (% of total electricity output)'
    plot_selected_years_line_chart(df_years, selected_years, line_selected_indicator2)

   

# Execute the main function when the script is run
if __name__ == '__main__':
    main()