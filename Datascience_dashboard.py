import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

#Loading the dataset
df = pd.read_excel('MingerCleaned_data.xlsx')

#Streamlit app title
st.title(" :chart_with_upwards_trend: Minger Analysis Dashboard")

# Creating two columns layout
col1, col2 = st.columns(2)

# Pie chart showing sum of sales by category
pie_chart_data = df['Category'].value_counts().reset_index()
pie_chart_data.columns = ['Category', 'Sales']
fig1 = px.pie(pie_chart_data, values='Sales', names='Category', title='Sum of Sales by Category')
col1.plotly_chart(fig1, use_container_width=True)

# Pie chart showing count of orders by Market
market_count = df['Market'].value_counts().reset_index()
market_count.columns = ['Market', 'Count']
fig8 = px.pie(market_count, values='Count', names='Market', title='Count of Orders by Market')
col2.plotly_chart(fig8, use_container_width=True)


# Line chart showing trends of shipping cost over time
line_chart_data = df.groupby('Order Date')['Shipping Cost'].sum().reset_index()
fig2 = px.line(line_chart_data, x='Order Date', y='Shipping Cost', title='Trends of Shipping Cost by year')
col1.plotly_chart(fig2, use_container_width=True)

# Scatter plot of relationships between sales and profit
total_rows = len(df)
sample_size = min(3000, total_rows)
scatter_data = df.sample(sample_size, replace=True)  # Setting replace=True for large samples
fig3 = px.scatter(scatter_data, x='Sales', y='Profit', color='Category', title='Relationship between Sales and Profit')
col2.plotly_chart(fig3, use_container_width=True)

# Set background color and font color
plt.style.use('dark_background')
plt.rcParams['text.color'] = 'white'

# Bar chart comparing different categories
bar_chart_data = df.groupby('Sub-Category')['Profit'].sum().reset_index()
fig4 = px.bar(bar_chart_data, x='Sub-Category', y='Profit', title='Profit by Sub Category')
col2.plotly_chart(fig4, use_container_width=True)

# Box plot of Profit by Category
fig5 = px.box(df, x='Category', y='Profit', title='Profit Distribution by Category')
col1.plotly_chart(fig5, use_container_width=True)

# Ribbon chart showing Sales and Profit
import plotly.graph_objects as go
fig12 = go.Figure(data=[
    go.Scatter(
        name='Sales',
        x=df['Order Date'],
        y=df['Sales'],
        mode='lines',
        line=dict(color='blue'),
        stackgroup='one'
    ),
    go.Scatter(
        name='Profit',
        x=df['Order Date'],
        y=df['Profit'],
        mode='lines',
        line=dict(color='green'),
        stackgroup='two'
    )
])

fig12.update_layout(title='Ribbon Chart of Sales and Profit', xaxis_title='Date', yaxis_title='Value')
col1.plotly_chart(fig12, use_container_width=True)

# Summary Card for Total Profit
total_profit = df['Profit'].sum()
col2.metric("Total Profit", f"${total_profit:,.2f}")

# Area Chart of Sales Over Region
cumulative_sales = df.groupby('Region')['Sales'].sum().cumsum().reset_index()
fig6 = px.area(cumulative_sales, x='Region', y='Sales', title='Cumulative Sales Over Time')
col2.plotly_chart(fig6, use_container_width=True)


#MBA 
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Read the cleaned dataset
MingerCleaned_data = pd.read_excel("MingerCleaned_data.xlsx")

# Group by 'Order ID' and 'Sub-Category' and count the occurrences of each combination
basket = (MingerCleaned_data.groupby(['Order ID', 'Sub-Category'])['Row ID']
          .count().unstack().reset_index().fillna(0)
          .set_index('Order ID'))

# Convert the occurrence counts to binary values (0 or 1)
basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)

#Generating frequent itemsets using the Apriori algorithm (Market Basket Analysis)
frequent_itemsets = apriori(basket_sets, min_support=0.001, use_colnames=True)

# Generate association rules using the frequent itemsets and specify the metric and minimum threshold
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules = rules[['antecedents', 'consequents', 'antecedent support', 'consequent support', 'support', 'confidence', 'lift']]


# Creating an empty DataFrame 
binary_subcategories = pd.DataFrame(index=basket.index)

# Iterate over unique sub-categories
for sub_category in MingerCleaned_data['Sub-Category'].unique():
    # Create a binary column indicating presence of sub-category
    binary_subcategories[sub_category] = (basket[sub_category] > 0).astype(int)


#Heatmap of the the association rules

import seaborn as sns
import matplotlib.pyplot as plt

# Pivot the rules DataFrame to create a table for the heatmap
HeatMapTable = rules.pivot(index='consequents', columns='antecedents', values='lift')

# Create the heatmap using seaborn
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(HeatMapTable, cmap='YlGnBu', annot=True, fmt=".2f", cbar=True)

# Set plot titles and labels
plt.title('Heatmap of lift for Association Rules')
plt.xlabel('Antecedents')
plt.ylabel('Consequents')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Display the heatmap directly in Streamlit
st.pyplot(plt) 

#Barplot of support for Sub-category items 
# Assuming you have already defined `frequent_itemsets` DataFrame
sorted_itemsets = frequent_itemsets.sort_values(by='support', ascending=False).head(10)

# Convert frozensets to strings
sorted_itemsets['itemsets_str'] = sorted_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(sorted_itemsets['itemsets_str'], sorted_itemsets['support'], color='teal')
plt.xlabel('Itemsets')
plt.ylabel('Support')
plt.title('Top 10 Frequent Itemsets')
plt.xticks(rotation=45)


# Display the bar plot directly in Streamlit
st.pyplot(plt)
















