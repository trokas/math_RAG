import streamlit as st
import plotly.express as px
import pandas as pd

df = pd.read_csv('problems_t_sne.csv')

fig = px.scatter(
    df,
    x='x',
    y='y',
    color='area', 
    hover_data={'problem_text': True, 'area': True, 'x': False, 'y': False}
)

# Update legend: move to top & center, horizontal orientation
fig.update_layout(
    width=1000,
    height=800,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        itemwidth=70
    )
)

fig.update_traces(hovertemplate='%{customdata[0]}<extra></extra>')

st.title('Having fun with math problem embeddings')

st.markdown("""
To generate the dataset, I used [Mathematics dataset](https://github.com/google-deepmind/mathematics_dataset).
It produced 1214 problems from 8 different areas and 3 difficulty levels. My goal was to see if I could produce embeddings 
that cluster math problems into similar areas even without any prior knowledge.

Initially, I fed the problems directly into *GPT-o1*, asking it to pick an area, but I only got around a 50% hit rate. 
As you'll see below, with a proper embedding, we can easily reach 95%.
""")

st.header("t-SNE Visualization (Hover to see problem)")

st.markdown("""
This visualization shows the obtained embeddings reduced to 2D. Colors are based on the original dataset, though in some 
cases, the embedding might group tasks even better. Hover over any point and see for yourself. You'll notice the groupings 
are meaningful, and similar problems (requiring similar skills) fall close to each other.
""")

st.plotly_chart(fig, use_container_width=False)

st.header("Methodology")

st.markdown("""
I initially tried feeding the problem directly into *text-embedding-3-large* to get embeddings, but that didn't lead to 
decent results. Instead, I realized that using *gpt-4o-mini* and first asking for a summary works much better. 
It improved the hit rate from *90% to 95%*.

Template I used before clustering:
```python
Restate the problem using natural language and then describe its key elements.
Problem: {problem}
What math area does this belong to?
How hard is it?
What else is interesting about it?
```
The execution itself was done using LangChain and OpenAI, with monitoring in LangSmith.
""")
