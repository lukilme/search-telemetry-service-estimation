import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.title("Exemplo de Plot Matplotlib no Streamlit")

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title("Seno de X")
ax.set_xlabel("X")
ax.set_ylabel("sin(X)")

st.pyplot(fig)
